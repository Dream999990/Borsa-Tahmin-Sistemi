import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def create_lstm_model(input_shape):
    """Gelişmiş LSTM Model mimarisi."""
    model = Sequential([
        # input_shape artık (60, 4) olacak (Close, Volume, SMA, RSI)
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        LSTM(units=64, return_sequences=False),
        Dropout(0.2),
        
        Dense(units=32, activation='relu'),
        Dense(units=1) # Sonuç yine tek: Gelecek fiyat
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(df, ticker):
    """Çok değişkenli veriyi hazırlar ve modeli eğitir."""
    
    # 1. Özellik seçimi (Feature Selection)
    # NaN değerleri (SMA ve RSI ilk günlerde NaN olur) temizleyelim
    features = df[['Close', 'Volume', 'SMA_20', 'RSI']].bfill().ffill()
    data = features.values
    
    # 2. Veriyi ölçeklendir
    # ÖNEMLİ: Her sütun kendi içinde 0-1 arasına çekilir
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    # 3. Eğitim setini oluştur (X: Tüm özellikler, y: Sadece Close fiyatı)
    X_train, y_train = [], []
    prediction_days = 60 # Geçmiş 60 gün
    
    for i in range(prediction_days, len(scaled_data)):
        # X_train: Son 60 günün 4 verisi (Close, Vol, SMA, RSI)
        X_train.append(scaled_data[i-prediction_days:i, :]) 
        # y_train: Sadece hedef günün Close fiyatı (0. sütun)
        y_train.append(scaled_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # 4. Modeli eğit
    # X_train shape: (Örnek Sayısı, 60, 4)
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    print(f"🚀 {ticker} için Çok Değişkenli Model eğitiliyor...")
    model.fit(X_train, y_train, batch_size=32, epochs=10) # Daha stabil olması için batch_size 32
    
    # 5. Modeli ve Scaler'ı kaydet
    # ÖNEMLİ: Scaler'ı da kaydetmeliyiz çünkü tahminde aynı ölçek lazım
    save_dir = "prediction/ml_models/saved_models/"
    os.makedirs(save_dir, exist_ok=True)
    
    model.save(f"{save_dir}{ticker}_model.h5")
    
    # Scaler'ı bir dosyaya kaydedelim (Tahmin sırasında kullanmak için)
    import joblib
    joblib.dump(scaler, f"{save_dir}{ticker}_scaler.pkl")
    
    return model, scaler