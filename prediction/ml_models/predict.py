import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

def get_prediction(ticker, df):
    """
    Eğitilmiş LSTM modelini yükler ve bir sonraki günün fiyatını tahmin eder.
    """
    # Modelin kaydedildiği klasör yolu
    model_path = f"prediction/ml_models/saved_models/{ticker}_model.h5"
    
    # 1. Kontrol: Bu hisse için eğitilmiş bir model var mı?
    if not os.path.exists(model_path):
        return "Model Eğitilmedi"

    try:
        # 2. Modeli dosyadan yükle
        model = load_model(model_path)
        
        # 3. Veriyi ölçeklendir (Model 0-1 arası sayılarla eğitildiği için)
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Sadece 'Close' (Kapanış) fiyatlarını alıyoruz
        closing_prices = df[['Close']].values
        scaled_data = scaler.fit_transform(closing_prices)
        
        # 4. Tahmin için son 60 günü hazırla
        # LSTM modelleri (Batch_size, Time_steps, Features) şeklinde veri bekler
        last_60_days = scaled_data[-60:].reshape(1, 60, 1)
        
        # 5. Tahmini yap
        predicted_price_scaled = model.predict(last_60_days, verbose=0)
        
        # 6. Tahmini gerçek fiyat değerine geri döndür
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        
        return round(float(predicted_price[0][0]), 2)
        
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return "Hata Oluştu"