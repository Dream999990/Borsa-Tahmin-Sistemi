import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def create_lstm_model(input_shape):
    """LSTM Model mimarisini kurar."""
    model = Sequential([
        # İlk katman: Verideki zaman serisi kalıplarını yakalar
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2), # Ezberlemeyi (overfitting) önlemek için %20 nöronu kapat
        
        # İkinci katman: Daha derin özellikler
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        
        # Çıkış katmanı: Tahmin edilen fiyat
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(df, ticker):
    """Veriyi hazırlar ve modeli eğitip kaydeder."""
    # Sadece kapanış fiyatlarını (Close) kullanıyoruz
    data = df[['Close']].values
    
    # Veriyi ölçeklendir (Yapay zeka 0-1 arasını daha iyi anlar)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    # Son 60 günü kullanarak bir sonraki günü tahmin etme yapısı
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Modeli oluştur ve eğit
    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=1, epochs=5) # Şimdilik hızlı olması için 5 epoch
    
    # Modeli kaydet
    save_path = f"prediction/ml_models/saved_models/{ticker}_model.h5"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    
    return model, scaler