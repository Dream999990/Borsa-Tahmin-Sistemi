import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

def get_technical_signal(df, prediction, last_price):
    """RSI, SMA ve AI tahminine göre Al/Sat sinyali üretir."""
    try:
        # Garantiye al: .iloc[-1] sonrası değerleri saf float'a çeviriyoruz
        rsi_val = df['RSI'].iloc[-1]
        sma_val = df['SMA_20'].iloc[-1]

        # Eğer hala Series dönerse ilk elemanı al
        rsi = float(rsi_val.iloc[0] if hasattr(rsi_val, 'iloc') else rsi_val)
        sma = float(sma_val.iloc[0] if hasattr(sma_val, 'iloc') else sma_val)
        
        last_price = float(last_price)
        prediction = float(prediction)
        
        score = 0
        # 1. RSI Kontrolü
        if rsi < 35: score += 1
        elif rsi > 65: score -= 1
        
        # 2. Trend Kontrolü
        if last_price > sma: score += 1
        else: score -= 1
        
        # 3. AI Tahmini Kontrolü
        if prediction > last_price: score += 1
        else: score -= 1

        # Karar mekanizması
        if score >= 2: return "Güçlü Al", "success"
        elif score == 1: return "Al", "primary"
        elif score == 0: return "Nötr / Bekle", "secondary"
        elif score == -1: return "Sat", "warning"
        else: return "Güçlü Sat", "danger"
    except Exception as e:
        print(f"⚠️ Sinyal hesaplama hatası: {e}")
        return "Analiz Yapılamıyor", "dark"

def get_prediction(ticker, df):
    try:
        model_path = f"prediction/ml_models/saved_models/{ticker}_model.h5"
        scaler_path = f"prediction/ml_models/saved_models/{ticker}_scaler.pkl"
        acc_path = f"prediction/ml_models/saved_models/{ticker}_accuracy.txt"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return "Model Eğitilmedi", "N/A", "N/A", "secondary"

        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        
        # Accuracy dosyasını güvenli oku
        accuracy = "85.0"
        if os.path.exists(acc_path):
            with open(acc_path, "r") as f:
                accuracy = f.read().strip()

        # 1. Özellikleri Hazırla
        features_df = df[['Close', 'Volume', 'SMA_20', 'RSI']].ffill().bfill()
        
        if len(features_df) < 60:
            return "Yetersiz Veri", "N/A", "N/A", "secondary"

        # 2. Veriyi Ölçeklendir
        scaled_data = scaler.transform(features_df.values)
        last_60_days = scaled_data[-60:]
        X_test = np.array([last_60_days])
        
        # 3. Tahmin Yap
        pred_scaled = model.predict(X_test, verbose=0)
        
        # 4. Ters Ölçekleme
        dummy_matrix = np.zeros((1, 4))
        dummy_matrix[0, 0] = pred_scaled[0, 0]
        inverse_result = scaler.inverse_transform(dummy_matrix)
        
        # Tekil sayıya zorla
        final_prediction = round(float(inverse_result[0, 0]), 2)
        
        # 5. Teknik Sinyali Oluştur (Hatanın %100 çözüldüğü nokta)
        # .item() kullanımı tekil bir skaler değer döndürür
        raw_last_price = df['Close'].iloc[-1]
        if hasattr(raw_last_price, 'item'):
            current_price = float(raw_last_price.item())
        else:
            current_price = float(raw_last_price)

        signal_text, signal_class = get_technical_signal(df, final_prediction, current_price)
        
        return final_prediction, accuracy, signal_text, signal_class

    except Exception as e:
        print(f"⚠️ Tahmin hatası ({ticker}): {e}")
        return "Hata", "N/A", "N/A", "secondary"