from django.db import models

class StockData(models.Model):
    ticker = models.CharField(max_length=10) # Örn: NVDA, THYAO
    date = models.DateField()
    close_price = models.FloatField()
    volume = models.BigIntegerField()
    sentiment_score = models.FloatField(default=0.0) # FinBERT'ten gelen skor

    class Meta:
        unique_together = ('ticker', 'date') # Aynı günün verisi tekrar etmesin

class PredictionResult(models.Model):
    ticker = models.CharField(max_length=10)
    prediction_date = models.DateField(auto_now_add=True)
    predicted_price = models.FloatField()
    actual_price = models.FloatField(null=True, blank=True) # Yarın belli olacak
    direction = models.CharField(max_length=10) # Artış / Azalış