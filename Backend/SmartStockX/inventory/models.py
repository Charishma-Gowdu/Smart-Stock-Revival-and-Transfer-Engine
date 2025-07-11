from django.db import models

class Inventory(models.Model):
    run_id = models.CharField(max_length=50)
    store_id = models.CharField(max_length=20)
    product_id = models.CharField(max_length=20)
    product_name = models.CharField(max_length=100, default="Unknown")
    stock = models.FloatField()
    expiry_date = models.DateField()
    shelf_life_days = models.IntegerField()
    avg_daily_sales = models.FloatField()
    MRP = models.FloatField()
    days_to_expiry = models.IntegerField()
    remaining_ratio = models.FloatField()
    expected_sales = models.FloatField()
    predicted_demand = models.FloatField()
    discount = models.FloatField()
    final_price = models.FloatField()

class Transfer(models.Model):
    run_id = models.CharField(max_length=50)
    product_id = models.CharField(max_length=20)
    expiry_date = models.DateField()
    from_store = models.CharField(max_length=20)
    to_store = models.CharField(max_length=20)
    quantity = models.IntegerField()
    distance_km = models.FloatField()
    remaining_ratio = models.FloatField()
    days_to_expiry = models.IntegerField()
