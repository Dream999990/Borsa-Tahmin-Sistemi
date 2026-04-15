import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import numpy as np

# NewsAPI'den aldığın anahtarı buraya yapıştır
NEWS_API_KEY = "SENIN_API_KEYIN"

def get_stock_data(ticker_symbol):
    """Geçmiş 5 yıllık fiyat ve teknik göstergeleri çeker."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1825)
        
        # Veriyi indir
        df = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        if df is None or df.empty:
            return None

        # --- TEKNİK GÖSTERGELER EKLEME ---
        
        # 1. Hareketli Ortalama (SMA 20)
        # rolling().mean() ile son 20 günün ortalamasını her satıra hesaplar
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # 2. RSI (Göreceli Güç Endeksi) - Alım/Satım doygunluğu
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # İhtiyacımız olan sütunları döndür (Close, Volume ve yeniler)
        return df[['Close', 'Volume', 'SMA_20', 'RSI']]
        
    except Exception as e:
        print(f"Borsa verisi çekilemedi: {e}")
        return None

def get_stock_news(ticker_name):
    """Hisse hakkında son haberleri çeker."""
    url = f'https://newsapi.org/v2/everything?q={ticker_name}&language=en&apiKey={NEWS_API_KEY}'
    try:
        response = requests.get(url)
        articles = response.json().get('articles', [])
        return [(a['title'], a['publishedAt']) for a in articles[:15]]
    except Exception as e:
        print(f"Haberler çekilemedi: {e}")
        return []