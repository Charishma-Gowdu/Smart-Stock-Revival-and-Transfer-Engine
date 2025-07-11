# inventory/ml/engine.py

import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ───────────── Config ─────────────
TODAY = datetime.now()
RUN_ID = TODAY.strftime("%Y%m%d_%H%M%S")
DEFAULT_REL_RATIO = 0.10
DEFAULT_ABS_DAYS = 2

# ───────────── Core Functions ─────────────

def load_data(inv_path: str, dist_path: str):
    inv = pd.read_csv(inv_path, parse_dates=["expiry_date"])
    dist = pd.read_csv(dist_path)

    if "price" in inv.columns and "MRP" not in inv.columns:
        inv.rename(columns={"price": "MRP"}, inplace=True)

    required = {"store_id", "product_id", "stock", "expiry_date",
                "shelf_life_days", "avg_daily_sales", "MRP"}
    missing = required - set(inv.columns)
    if missing:
        raise ValueError(f"Inventory CSV missing columns: {missing}")

    return inv, dist


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce')
    df["days_to_expiry"] = (df["expiry_date"] - pd.Timestamp(TODAY)).dt.days.clip(lower=0)
    df["remaining_ratio"] = df["days_to_expiry"] / df["shelf_life_days"].clip(lower=1)
    df["expected_sales"] = df["days_to_expiry"] * df["avg_daily_sales"]
    return df


def train_demand_model(df: pd.DataFrame):
    df = df.copy()
    df['days_to_expiry'] = df['days_to_expiry'].fillna(0)

    X = df[["stock", "avg_daily_sales", "days_to_expiry"]]
    y = df["expected_sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    df["predicted_demand"] = model.predict(X)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"[ML MODEL] Mean Squared Error: {mse:.2f}")

    return model, df


def compute_discount(row, base=0.10, max_disc=0.40):
    unsold_units = max(row["stock"] - row["predicted_demand"], 0)
    unsold_ratio = unsold_units / row["stock"] if row["stock"] else 0
    urgency_rel = 1 - row["remaining_ratio"]
    urgency_abs = max((DEFAULT_ABS_DAYS - row["days_to_expiry"] + 1), 0) / DEFAULT_ABS_DAYS
    urgency = max(urgency_rel, urgency_abs)
    dynamic = 0.5 * unsold_ratio * urgency
    return min(max_disc, base + dynamic)


def apply_pricing(df: pd.DataFrame):
    df = df.copy()
    df["discount"] = df.apply(compute_discount, axis=1)
    df["final_price"] = df["MRP"] * (1 - df["discount"])
    return df


def suggest_transfers(inv_df: pd.DataFrame, dist_df: pd.DataFrame,
                      ratio_thr: float, days_thr: int):
    transfers = []

    for (pid, exp) in inv_df[["product_id", "expiry_date"]].drop_duplicates().itertuples(index=False):
        batch = inv_df[(inv_df["product_id"] == pid) & (inv_df["expiry_date"] == exp)].copy()

        donors = batch[(batch["stock"] - batch["predicted_demand"] > 0) & (
            (batch["remaining_ratio"] <= ratio_thr) |
            (batch["days_to_expiry"] <= days_thr)
        )]
        receivers = batch[(batch["stock"] - batch["predicted_demand"]) < 0]

        if donors.empty or receivers.empty:
            continue

        donors = donors.copy()
        receivers = receivers.copy()

        donors.loc[:, "surplus"] = donors["stock"] - donors["predicted_demand"]
        receivers.loc[:, "surplus"] = receivers["stock"] - receivers["predicted_demand"]


        donors = donors.assign(
            risk=lambda d: (d["surplus"] * d["MRP"]) / d["days_to_expiry"].clip(lower=1)
        ).sort_values("risk", ascending=False)
        receivers = receivers.sort_values("surplus")

        for _, donor in donors.iterrows():
            for _, rec in receivers.iterrows():
                qty = int(min(donor["surplus"], -rec["surplus"]))
                if qty <= 0:
                    continue

                drow = dist_df[(dist_df["from_store"] == donor["store_id"]) &
                               (dist_df["to_store"] == rec["store_id"])]
                if drow.empty:
                    continue

                transfers.append({
                    "run_id": RUN_ID,
                    "product_id": pid,
                    "expiry_date": exp,
                    "from_store": donor["store_id"],
                    "to_store": rec["store_id"],
                    "quantity": qty,
                    "distance_km": drow["distance_km"].values[0],
                    "remaining_ratio": donor["remaining_ratio"],
                    "days_to_expiry": donor["days_to_expiry"],
                })

                donor["surplus"] -= qty
                rec["surplus"] += qty
                if donor["surplus"] <= 0:
                    break

    return pd.DataFrame(transfers)


# ───────────── Entry Function ─────────────

def run_engine(inventory_path, distance_path):
    inv, dist = load_data(inventory_path, distance_path)
    print(inv)
    inv = add_features(inv)
    _, inv = train_demand_model(inv)
    inv = apply_pricing(inv)
    inv["run_id"] = RUN_ID
    inv["surplus"] = inv["stock"] - inv["predicted_demand"]

    transfers = suggest_transfers(inv, dist, ratio_thr=DEFAULT_REL_RATIO, days_thr=DEFAULT_ABS_DAYS)

    return inv, transfers
