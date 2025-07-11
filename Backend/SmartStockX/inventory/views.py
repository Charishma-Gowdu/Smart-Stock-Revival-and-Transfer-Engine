from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from .ml.engine import run_engine
from .models import Inventory, Transfer
from .serializers import InventorySerializer, TransferSerializer

from rest_framework.generics import ListAPIView
import tempfile


class RunSmartStockX(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        print("Hi")
        inventory_file = request.FILES.get('inventory_file')
        distance_file = request.FILES.get('distance_file')

        if not inventory_file or not distance_file:
            return Response({"error": "Both inventory and distance files are required."},
                            status=status.HTTP_400_BAD_REQUEST)


        try:
            # Save the uploaded inventory file to a temp .csv
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as inv_temp:
                for chunk in inventory_file.chunks():
                    inv_temp.write(chunk)
                inv_path = inv_temp.name

            # Save the uploaded distance file to a temp .csv
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as dist_temp:
                for chunk in distance_file.chunks():
                    dist_temp.write(chunk)
                dist_path = dist_temp.name

            # Run SmartStockX engine
            inventory_df, transfers_df = run_engine(inv_path, dist_path)
            
            # DELETE OLD DATA
            try:
                Transfer.objects.all().delete()
            except Exception as e:
                print(f"Skipping Transfer delete: {e}")

            try:
                Inventory.objects.all().delete()
            except Exception as e:
                print(f"Skipping Inventory delete: {e}")


            # INSERT INVENTORY DATA
            inventory_objs = [
                Inventory(
                    run_id=row["run_id"],
                    store_id=row["store_id"],
                    product_id=row["product_id"],
                    product_name=row["product_name"],
                    stock=row["stock"],
                    expiry_date=row["expiry_date"],
                    shelf_life_days=row["shelf_life_days"],
                    avg_daily_sales=row["avg_daily_sales"],
                    MRP=row["MRP"],
                    days_to_expiry=row["days_to_expiry"],
                    remaining_ratio=row["remaining_ratio"],
                    expected_sales=row["expected_sales"],
                    predicted_demand=row["predicted_demand"],
                    discount=row["discount"],
                    final_price=row["final_price"],
                )
                for _, row in inventory_df.iterrows()
            ]
            Inventory.objects.bulk_create(inventory_objs)

            # INSERT TRANSFER DATA
            transfer_objs = [
                Transfer(
                    run_id=row["run_id"],
                    product_id=row["product_id"],
                    expiry_date=row["expiry_date"],
                    from_store=row["from_store"],
                    to_store=row["to_store"],
                    quantity=row["quantity"],
                    distance_km=row["distance_km"],
                    remaining_ratio=row["remaining_ratio"],
                    days_to_expiry=row["days_to_expiry"],
                )
                for _, row in transfers_df.iterrows()
            ]
            Transfer.objects.bulk_create(transfer_objs)

            return Response({
                "inventory": inventory_df.to_dict(orient="records"),
                "transfers": transfers_df.to_dict(orient="records")
            })

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# GET APIs for frontend

class InventoryListView(ListAPIView):
    queryset = Inventory.objects.all()
    serializer_class = InventorySerializer

class TransferListView(ListAPIView):
    queryset = Transfer.objects.all()
    serializer_class = TransferSerializer
