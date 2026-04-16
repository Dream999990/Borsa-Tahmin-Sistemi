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
    
    # Klasör yoksa oluştur, hata alma
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return

    # Klasördeki tüm .h5 dosyalarını listele
    models = glob.glob(os.path.join(model_dir, "*.h5"))
    
    # Eğer model sayısı sınırı aşıyorsa yeniye yer açmak için en eskiyi sil
    if len(models) >= max_models:
        models.sort(key=os.path.getmtime)
        
        # Sınırı aşanları temizle
        num_to_delete = len(models) - max_models + 1
        for i in range(num_to_delete):
            oldest_model = models[i]
            ticker_to_delete = os.path.basename(oldest_model).split('_')[0]
            
            try:
                os.remove(oldest_model) # .h5 siler
                scaler_path = oldest_model.replace(".h5", ".pkl")
                acc_path = oldest_model.replace(".h5", ".accuracy.txt")
                
                if os.path.exists(scaler_path):
                    os.remove(scaler_path) # .pkl siler
                if os.path.exists(acc_path):
                    os.remove(acc_path) # .txt siler
                    
                print(f"♻️ Hafıza yönetimi: {ticker_to_delete} modeli temizlendi.")
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
                # Yeni model eğitimi öncesi hafıza temizliği
                limit_saved_models(max_models=3) 
                
                if len(df) > 60:
                    print(f"🧠 {ticker} için Multivariate LSTM eğitiliyor...")
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
            
            # 3. Grafik İçin Veri Hazırlama (Son 30 İş Günü)
            recent_df = df.tail(30)
            chart_dates = recent_df.index.strftime('%Y-%m-%d').tolist()
            chart_prices = [round(float(p), 2) for p in recent_df['Close'].values.flatten()]
            chart_sma = [round(float(s), 2) if not np.isnan(s) else None for s in recent_df['SMA_20'].values.flatten()]

            # 4. Haber Analizi
            raw_news = get_stock_news(ticker)
            sentiment_score = analyze_sentiment(ticker)

            # 5. Gelişmiş Tahmin ve Sinyal Mekanizması
            # prediction: Tahmin fiyatı
            # accuracy: Modelin başarı yüzdesi
            # signal_text: Al/Sat mesajı
            # signal_class: Bootstrap renk sınıfı (success, danger vb.)
            prediction, accuracy, signal_text, signal_class = get_prediction(ticker, df)
            
            context = {
                'ticker': ticker,
                'last_price': last_price,
                'sentiment': sentiment_score,
                'prediction': prediction,
                'accuracy': accuracy,
                'signal_text': signal_text,
                'signal_class': signal_class,
                'chart_dates': chart_dates,
                'chart_prices': chart_prices,
                'chart_sma': chart_sma,
                'news_list': raw_news[:5],
            }
        else:
            context = {'error': 'Hisse verisi çekilemedi. Sembolü kontrol edin.'}

    return render(request, 'index.html', context)