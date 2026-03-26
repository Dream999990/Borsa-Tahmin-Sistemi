from django.shortcuts import render
from .services.data_fetcher import get_stock_data
from .services.sentiment import analyze_sentiment
from .ml_models.predict import get_prediction
import pandas as pd

def index(request):
    context = {}
    
    if request.method == "POST":
        ticker = request.POST.get('ticker').upper()
        
        # 1. Adım: Güncel Borsa Verilerini Çek
        df = get_stock_data(ticker)
        
        if df is not None and not df.empty:
            last_price = round(df['Close'].iloc[-1], 2)
            
            # 2. Adım: Haber Duygu Analizi (FinBERT)
            # Not: Bu işlem ilk seferde biraz yavaş olabilir
            sentiment_score = analyze_sentiment(ticker)
            
            # 3. Adım: Yapay Zeka Fiyat Tahmini (LSTM)
            # Not: Eğer saved_models klasöründe model yoksa "Model Eğitilmedi" döner
            prediction = get_prediction(ticker, df)
            
            # 4. Adım: Verileri Arayüze (HTML) Gönder
            context = {
                'ticker': ticker,
                'last_price': last_price,
                'sentiment': sentiment_score,
                'prediction': prediction,
            }
        else:
            context = {'error': 'Hisse verisi bulunamadı. Lütfen kodu kontrol edin.'}

    return render(request, 'prediction/index.html', context)