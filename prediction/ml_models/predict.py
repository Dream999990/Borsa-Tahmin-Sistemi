import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def get_prediction(ticker, df):
    try:
        model_path = f"prediction/ml_models/saved_models/{ticker}_model.h5"
        model = load_model(model_path)
        
        # Sadece Kapanış fiyatını al ve boş olmadığını doğrula
        data = df[['Close']].values
        if len(data) < 60:
            return "Yetersiz Veri"

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data) # Hata tam burada oluyordu

        # Son 60 günü hazırla
        last_60_days = scaled_data[-60:]
        X_test = np.array([last_60_days])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Tahmin yap
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        
        return round(float(pred_price[0][0]), 2)
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return "Model Eğitilmedi"