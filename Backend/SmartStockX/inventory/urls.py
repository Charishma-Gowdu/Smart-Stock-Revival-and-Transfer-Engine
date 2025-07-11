from django.urls import path, include
from .views import RunSmartStockX, InventoryListView, TransferListView

urlpatterns = [
    path('run/', RunSmartStockX.as_view(), name='run_smartstockx'),
    path('inventory/', InventoryListView.as_view(), name='inventory_list'),
    path('transfers/', TransferListView.as_view(), name='transfer_list'),
]