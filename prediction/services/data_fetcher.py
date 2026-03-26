import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests

# NewsAPI'den aldığın anahtarı buraya yapıştır
NEWS_API_KEY = "SENIN_API_KEYIN"

def get_stock_data(ticker_symbol):
    """Geçmiş 5 yıllık fiyat verilerini çeker."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1825)
        df = yf.download(ticker_symbol, start=start_date, end=end_date)
        return df[['Close', 'Volume']]
    except Exception as e:
        print(f"Borsa verisi çekilemedi: {e}")
        return None

def get_stock_news(ticker_name):
    """Hisse hakkında son haberleri çeker."""
    url = f'https://newsapi.org/v2/everything?q={ticker_name}&language=en&apiKey={NEWS_API_KEY}'
    try:
        response = requests.get(url)
        articles = response.json().get('articles', [])
        # Sadece başlık ve tarihi döndür
        return [(a['title'], a['publishedAt']) for a in articles[:15]]
    except Exception as e:
        print(f"Haberler çekilemedi: {e}")
        return []