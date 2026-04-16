import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

def get_prediction(ticker, df):
    try:
        model_path = f"prediction/ml_models/saved_models/{ticker}_model.h5"
        scaler_path = f"prediction/ml_models/saved_models/{ticker}_scaler.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return "Model Eğitilmedi"

        model = load_model(model_path)
        scaler = joblib.load(scaler_path) # Kaydettiğimiz özel ölçekleyiciyi yüklüyoruz
        
        # 1. Özellikleri Hazırla (Eğitimdekiyle aynı sıra: Close, Volume, SMA_20, RSI)
        # NaN değerleri temizliyoruz (özellikle en güncel veride NaN olmamalı)
        features = df[['Close', 'Volume', 'SMA_20', 'RSI']].ffill().bfill()
        
        if len(features) < 60:
            return "Yetersiz Veri"

        # 2. Veriyi Ölçeklendir
        # Artık 4 sütun üzerinden ölçekleme yapılıyor
        scaled_data = scaler.transform(features.values)

        # 3. Son 60 Günü Hazırla
        last_60_days = scaled_data[-60:]
        X_test = np.array([last_60_days]) # Shape: (1, 60, 4)
        
        # 4. Tahmin Yap
        pred_scaled = model.predict(X_test, verbose=0)
        
        # 5. Ters Ölçekleme (Inverse Transform)
        # Model tek bir değer döndürür ama scaler 4 değer bekler.
        # Bu yüzden dummy (sahte) bir matris oluşturup fiyata denk gelen kısmı geri çeviriyoruz.
        dummy_matrix = np.zeros((1, 4))
        dummy_matrix[0, 0] = pred_scaled[0, 0] # Tahmini ilk sütuna koy (Close)
        inverse_result = scaler.inverse_transform(dummy_matrix)
        
        final_prediction = round(float(inverse_result[0, 0]), 2)
        
        return final_prediction
    except Exception as e:
        print(f"⚠️ Tahmin hatası ({ticker}): {e}")
        return "Hata Oluştu"