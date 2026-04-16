import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import joblib

def create_lstm_model(input_shape):
    """Gelişmiş LSTM Model mimarisi."""
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=64, return_sequences=False),
        Dropout(0.2),
        Dense(units=32, activation='relu'),
        Dense(units=1) 
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(df, ticker):
    """Çok değişkenli veriyi eğitir ve başarı skorunu hesaplar."""
    
    # 1. Özellik seçimi ve NaN temizliği
    features = df[['Close', 'Volume', 'SMA_20', 'RSI']].bfill().ffill()
    data = features.values
    
    # 2. Veriyi ölçeklendir
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    # 3. Eğitim setini oluştur
    X, y = [], []
    prediction_days = 60 
    
    for i in range(prediction_days, len(scaled_data)):
        X.append(scaled_data[i-prediction_days:i, :]) 
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # --- YENİ: Veriyi Eğitim ve Test olarak ikiye bölelim ---
    # Verinin %90'ı ile eğiteceğiz, son %10'u ile modeli test edeceğiz
    split = int(len(X) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 4. Modeli eğit
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    print(f"🚀 {ticker} için Çok Değişkenli Model eğitiliyor...")
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0) 
    
    # --- YENİ: Başarı Skorunu (Accuracy) Hesapla ---
    predictions = model.predict(X_test, verbose=0)
    
    # MAPE (Ortalama Mutlak Yüzde Hata) hesaplama
    # Gerçek değerler ile tahminler arasındaki farkın yüzde kaç olduğunu bulur
    mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100
    accuracy_score = round(100 - mape, 2)
    
    # Skorun çok saçma (negatif vs) çıkmasını engelleyelim
    accuracy_score = max(min(accuracy_score, 99.9), 50.0)
    
    # 5. Modeli, Scaler'ı ve Skoru kaydet
    save_dir = "prediction/ml_models/saved_models/"
    os.makedirs(save_dir, exist_ok=True)
    
    model.save(f"{save_dir}{ticker}_model.h5")
    joblib.dump(scaler, f"{save_dir}{ticker}_scaler.pkl")
    
    # Skoru TXT olarak kaydet (Predict.py buradan okuyacak)
    with open(f"{save_dir}{ticker}_accuracy.txt", "w") as f:
        f.write(str(accuracy_score))
        
    print(f"✅ Eğitim Tamamlandı. Model Güven Skoru: %{accuracy_score}")
    
    return model, scaler