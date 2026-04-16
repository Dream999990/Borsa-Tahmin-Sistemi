from django.shortcuts import render
import os
import pandas as pd
import numpy as np
import glob
from .services.data_fetcher import get_stock_data, get_stock_news
from .services.sentiment import analyze_sentiment
from .ml_models.predict import get_prediction
from .ml_models.train import train_model 

def limit_saved_models(max_models=3):
    """Sistemdeki model sayısını sınırlar, en eski olanı siler."""
    model_dir = "prediction/ml_models/saved_models/"
    
    # Klasör yoksa fonksiyondan çık
    if not os.path.exists(model_dir):
        return

    # Klasördeki tüm .h5 dosyalarını listele
    models = glob.glob(os.path.join(model_dir, "*.h5"))
    
    # Eğer model sayısı sınırı aşıyorsa (Sınırı max_models - 1 yapıyoruz ki yeniye yer açılsın)
    if len(models) >= max_models:
        # Dosyaları oluşturulma zamanına göre sırala (en eski en başta)
        models.sort(key=os.path.getmtime)
        
        # Sınırı aşan miktarda eski modeli sil
        # Örn: 3 model var, sınır 3. 1 tanesini silmeliyiz.
        num_to_delete = len(models) - max_models + 1
        for i in range(num_to_delete):
            oldest_model = models[i]
            ticker_to_delete = os.path.basename(oldest_model).split('_')[0]
            
            try:
                os.remove(oldest_model) # .h5 siler
                scaler_path = oldest_model.replace(".h5", ".pkl")
                if os.path.exists(scaler_path):
                    os.remove(scaler_path) # .pkl siler
                print(f"♻️ Hafıza yönetimi: En eski model ({ticker_to_delete}) temizlendi.")
            except Exception as e:
                print(f"⚠️ Temizlik hatası: {e}")

def index(request):
    context = {}
    if request.method == "POST":
        ticker = request.POST.get('ticker').upper()
        df = get_stock_data(ticker)
        
        if df is not None and not df.empty:
            # 1. Model Kontrolü ve Eğitimi
            model_path = f"prediction/ml_models/saved_models/{ticker}_model.h5"
            
            if not os.path.exists(model_path):
                # YENİ: Model eğitilmeden hemen önce temizlik yap
                limit_saved_models(max_models=3) 
                
                if len(df) > 60:
                    print(f"🧠 {ticker} için yeni model eğitiliyor...")
                    train_model(df, ticker)
                else:
                    context['error'] = "Yeterli veri yok (En az 60 gün gerekli)."

            # 2. Fiyat Verisini Temizleme
            try:
                all_close_prices = df['Close'].values.flatten()
                last_price_raw = all_close_prices[~np.isnan(all_close_prices)][-1]
                last_price = round(float(last_price_raw), 2)
            except Exception as e:
                print(f"⚠️ Fiyat çekme hatası: {e}")
                last_price = 0.0
            
            # 3. Grafik İçin Veri Hazırlama
            recent_df = df.tail(30)
            chart_dates = recent_df.index.strftime('%Y-%m-%d').tolist()
            chart_prices = [round(float(p), 2) for p in recent_df['Close'].values.flatten()]
            chart_sma = [round(float(s), 2) if not np.isnan(s) else None for s in recent_df['SMA_20'].values.flatten()]

            # 4. Haberleri Çekme
            raw_news = get_stock_news(ticker)

            sentiment_score = analyze_sentiment(ticker)
            prediction = get_prediction(ticker, df)
            
            context = {
                'ticker': ticker,
                'last_price': last_price,
                'sentiment': sentiment_score,
                'prediction': prediction,
                'chart_dates': chart_dates,
                'chart_prices': chart_prices,
                'chart_sma': chart_sma,
                'news_list': raw_news[:5],
            }
        else:
            context = {'error': 'Hisse verisi çekilemedi. Sembolü kontrol edin.'}

    return render(request, 'index.html', context)