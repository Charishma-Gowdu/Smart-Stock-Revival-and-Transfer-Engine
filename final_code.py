"""
Stock-revival engine that flags donor batches when **either** condition holds:
    1. Remaining shelf-life percentage ≤ `--ratio`  (e.g. ≤10%)
    2. Absolute days-to-expiry  ≤ `--days`  (e.g. ≤2days)
"""

import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np # Import numpy for random number generation

# ───────────── Config ─────────────
TODAY = datetime.now()
RUN_ID = TODAY.strftime("%Y%m%d_%H%M%S")
DEFAULT_REL_RATIO = 0.10  # 10 % of shelf life
DEFAULT_ABS_DAYS = 2      # 2 days or fewer

# ───────────── Data Loading ─────────────

def load_data(inv_path: str, dist_path: str):
    """Load inventory & distance CSVs, fix column names, basic validation."""
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

# ───────────── Feature Engineering ─────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce') # Added this line
    df["days_to_expiry"]  = (df["expiry_date"] - pd.Timestamp(TODAY)).dt.days.clip(lower=0)
    df["remaining_ratio"] = df["days_to_expiry"] / df["shelf_life_days"].clip(lower=1)
    df["expected_sales"]  = df["days_to_expiry"] * df["avg_daily_sales"]

    # --- Simulate actual sales data for demonstration ---
    # In a real scenario, you would load and merge your actual sales data here.
    # This simulation creates random sales data based on average daily sales and days to expiry,
    # with some randomness added.
    np.random.seed(42) # for reproducibility
    df["actual_sales"] = (df["avg_daily_sales"] * df["days_to_expiry"] * np.random.uniform(0.8, 1.2, size=len(df))).round().clip(lower=0)
    # --- End of simulation ---


    return df

# ───────────── Demand Prediction ─────────────


def train_demand_model(df: pd.DataFrame):
    df = df.copy()  # Avoid modifying the original DataFrame
    df['days_to_expiry'] = df['days_to_expiry'].fillna(0)
    # Ensure 'actual_sales' column exists and handle potential NaNs if not simulated
    if "actual_sales" not in df.columns:
         raise ValueError("'actual_sales' column not found. Please load or simulate actual sales data.")
    df['actual_sales'] = df['actual_sales'].fillna(df['actual_sales'].median()) # Fill NaNs in simulated data


    # Define features and target
    X = df[["stock", "avg_daily_sales", "days_to_expiry"]]
    # Change the target variable to 'actual_sales'
    y = df["actual_sales"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict demand on the entire dataset
    # The model now predicts 'actual_sales' but we store it as 'predicted_demand'
    df["predicted_demand"] = model.predict(X)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set (predicting actual_sales): {mse}")

    return model, df

# ───────────── Pricing ─────────────

def compute_discount(row, base=0.10, max_disc=0.40):
    """Scale discount by both surplus ratio and urgency (1 - remaining_ratio)."""
    # Use predicted_demand (which is now predicting actual sales)
    unsold_units = max(row["stock"] - row["predicted_demand"], 0)
    unsold_ratio = unsold_units / row["stock"] if row["stock"] else 0
    urgency_rel  = 1 - row["remaining_ratio"]
    urgency_abs  = max((DEFAULT_ABS_DAYS - row["days_to_expiry"] + 1), 0) / DEFAULT_ABS_DAYS
    urgency      = max(urgency_rel, urgency_abs)  # whichever is higher
    dynamic      = 0.5 * unsold_ratio * urgency
    return min(max_disc, base + dynamic)

def apply_pricing(df: pd.DataFrame):
    df = df.copy()
    df["discount"]    = df.apply(compute_discount, axis=1)
    df["final_price"] = df["MRP"] * (1 - df["discount"])
    return df

# ───────────── Transfer Suggestion ─────────────

def suggest_transfers(inv_df: pd.DataFrame, dist_df: pd.DataFrame, ratio_thr: float, days_thr: int):
    transfers = []

    # print("--- Inside suggest_transfers ---")
    # print(f"Initial inventory dataframe shape: {inv_df.shape}")
    # print(f"Initial distance dataframe shape: {dist_df.shape}")
    # print(f"Ratio threshold: {ratio_thr}, Days threshold: {days_thr}")

    for (pid, exp) in inv_df[["product_id", "expiry_date"]].drop_duplicates().itertuples(index=False):
        # print(f"\nProcessing product_id: {pid}, expiry_date: {exp}")
        batch = inv_df[(inv_df["product_id"] == pid) & (inv_df["expiry_date"] == exp)].copy()
        # batch["surplus"] = batch["stock"] - batch["predicted_demand"] # Moved this calculation outside the function

        # print(f"Batch shape: {batch.shape}")
        # print(f"Batch head:\n{batch.head()}")


        donors = batch[(batch["surplus"] > 0) & (
            (batch["remaining_ratio"] <= ratio_thr) |
            (batch["days_to_expiry"]  <= days_thr)
        )]
        receivers = batch[batch["surplus"] < 0]

        # print(f"Donors shape: {donors.shape}")
        # print(f"Receivers shape: {receivers.shape}")

        if donors.empty or receivers.empty:
            # print("No donors or receivers found for this batch. Skipping.")
            continue

        donors = donors.assign(
            risk=lambda d: (d["surplus"] * d["MRP"]) / d["days_to_expiry"].clip(lower=1)
        ).sort_values("risk", ascending=False)
        receivers = receivers.sort_values("surplus")

        # print(f"Donors head (sorted by risk):\n{donors.head()}")
        # print(f"Receivers head (sorted by surplus):\n{receivers.head()}")


        for _, donor in donors.iterrows():
            for _, rec in receivers.iterrows():
                qty = int(min(donor["surplus"], -rec["surplus"]))
                # print(f"  Attempting transfer: donor={donor['store_id']}, receiver={rec['store_id']}, quantity={qty}")
                if qty <= 0:
                    # print("    Quantity is zero or negative. Skipping.")
                    continue

                drow = dist_df[(dist_df["from_store"] == donor["store_id"]) &
                               (dist_df["to_store"]   == rec["store_id"])]
                if drow.empty:
                    # print(f"    No distance information found for transfer between {donor['store_id']} and {rec['store_id']}. Skipping.")
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
                # print(f"    Added transfer: {transfers[-1]}")


                donor["surplus"] -= qty
                rec["surplus"] += qty
                if donor["surplus"] <= 0:
                    # print(f"    Donor {donor['store_id']} surplus exhausted. Moving to next donor.")
                    break

    # print(f"\n--- End of suggest_transfers ---")
    # print(f"Total transfers suggested: {len(transfers)}")
    return pd.DataFrame(transfers)

# ───────────── Main ─────────────

# Sample test case - replace with your actual file paths
inventory_file = "inventory_large_with_shelf_life.csv"
distance_file = "store_distance_large.csv"
relative_threshold = DEFAULT_REL_RATIO  # Using the default relative threshold
absolute_days_threshold = DEFAULT_ABS_DAYS # Using the default absolute days threshold

inv, dist = load_data(inventory_file, distance_file)
inv  = add_features(inv) # This now includes simulated 'actual_sales'
model, inv = train_demand_model(inv) # Model trained on 'actual_sales'
inv  = apply_pricing(inv)
inv["run_id"] = RUN_ID
inv["surplus"] = inv["stock"] - inv["predicted_demand"] # Added surplus calculation here

transfers = suggest_transfers(inv, dist, ratio_thr=relative_threshold, days_thr=absolute_days_threshold)

# Display feature importance
print("\nFeature Importance:")
for feature, importance in zip(["stock", "avg_daily_sales", "days_to_expiry"], model.feature_importances_):
    print(f"{feature}: {importance:.4f}")


inv.to_csv(f"inventory_with_predictions.csv", index=False)
transfers.to_csv(f"transfer_suggestions.csv", index=False)
print(f"\n[✔] Outputs saved (run_id) — ratio ≤{relative_threshold}, days ≤{absolute_days_threshold}")



