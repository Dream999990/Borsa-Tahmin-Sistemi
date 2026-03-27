from django.shortcuts import render
import os
import pandas as pd
import numpy as np
from .services.data_fetcher import get_stock_data
from .services.sentiment import analyze_sentiment
from .ml_models.predict import get_prediction
from .ml_models.train import train_model 

def index(request):
    context = {}
    if request.method == "POST":
        ticker = request.POST.get('ticker').upper()
        df = get_stock_data(ticker)
        
        if df is not None and not df.empty:
            # 1. Model Kontrolü ve Eğitimi
            model_path = f"prediction/ml_models/saved_models/{ticker}_model.h5"
            if not os.path.exists(model_path):
                print(f"🚀 {ticker} için model eğitiliyor, lütfen bekleyin...")
                if len(df) > 60:
                    train_model(df, ticker) # Burada eğitim başlar
                else:
                    context['error'] = "Yeterli veri yok (En az 60 gün gerekli)."

            # 2. Fiyat Verisini Temizleme
            try:
                # Veriyi düzleştirip (flatten) son geçerli fiyatı alıyoruz
                all_close_prices = df['Close'].values.flatten()
                last_price_raw = all_close_prices[~np.isnan(all_close_prices)][-1]
                last_price = round(float(last_price_raw), 2)
            except Exception as e:
                print(f"⚠️ Fiyat çekme hatası: {e}")
                last_price = 0.0
            
            # 3. Analizleri Çalıştır ve Context Oluştur
            # Bu kısım try-except bloğunun DIŞINDA olmalı ki her zaman çalışsın
            sentiment_score = analyze_sentiment(ticker)
            prediction = get_prediction(ticker, df)
            
            context = {
                'ticker': ticker,
                'last_price': last_price,
                'sentiment': sentiment_score,
                'prediction': prediction,
            }
        else:
            context = {'error': 'Hisse verisi çekilemedi. Sembolü kontrol edin.'}

    return render(request, 'index.html', context)