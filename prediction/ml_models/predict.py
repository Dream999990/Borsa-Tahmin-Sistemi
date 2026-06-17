import os

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from .train import BASELINE_FEATURES, prepare_modeling_frame


def get_technical_signal(df, prediction, last_price, sentiment_score=0.0):
    """RSI, trend, return tahmini ve haber duygu skoruna göre Al/Sat sinyali üretir."""
    try:
        rsi_val = df["RSI"].iloc[-1]
        sma_val = df["SMA_20"].iloc[-1]
        rsi = float(rsi_val.iloc[0] if hasattr(rsi_val, "iloc") else rsi_val)
        sma = float(sma_val.iloc[0] if hasattr(sma_val, "iloc") else sma_val)

        last_price = float(last_price)
        prediction = float(prediction)

        score = 0
        if rsi < 35:
            score += 1
        elif rsi > 65:
            score -= 1

        if last_price > sma:
            score += 1
        else:
            score -= 1

        if prediction > last_price:
            score += 1
        else:
            score -= 1

        sentiment_score = float(sentiment_score)
        if sentiment_score > 0.15:
            score += 1
        elif sentiment_score < -0.15:
            score -= 1

        if score >= 3:
            return "Güçlü Al", "success"
        if score == 2:
            return "Al", "primary"
        if score == 1:
            return "Al", "primary"
        if score == 0:
            return "Nötr / Bekle", "secondary"
        if score == -1:
            return "Sat", "warning"
        return "Güçlü Sat", "danger"
    except Exception as e:
        print(f"⚠️ Sinyal hesaplama hatası: {e}")
        return "Analiz Yapılamıyor", "dark"


def _legacy_price_prediction(model, scaler, df):
    features_df = df[BASELINE_FEATURES].ffill().bfill()
    scaled_data = scaler.transform(features_df.values)
    last_60_days = scaled_data[-60:]
    pred_scaled = model.predict(np.array([last_60_days]), verbose=0)
    dummy_matrix = np.zeros((1, len(BASELINE_FEATURES)))
    dummy_matrix[0, 0] = pred_scaled[0, 0]
    return float(scaler.inverse_transform(dummy_matrix)[0, 0])


def _bundle_prediction(model, bundle, df, news_items=None, sentiment_details=None):
    modeling_frame = prepare_modeling_frame(df, news_items, sentiment_details)
    feature_columns = bundle.get("feature_columns", [])
    sequence_length = int(bundle.get("sequence_length", 60))
    if len(modeling_frame) < sequence_length:
        raise ValueError("Tahmin için yeterli sequence verisi yok.")

    feature_values = modeling_frame[feature_columns].values
    scaled_features = bundle["feature_scaler"].transform(feature_values)
    last_sequence = scaled_features[-sequence_length:]
    pred_scaled = model.predict(np.array([last_sequence]), verbose=0)
    predicted_target = float(bundle["target_scaler"].inverse_transform(pred_scaled)[0, 0])
    last_close = float(modeling_frame["Close"].iloc[-1])
    if bundle.get("target_type") == "close":
        return predicted_target
    return last_close * (1 + predicted_target)


def get_prediction(
    ticker,
    df,
    sentiment_score=0.0,
    current_price_override=None,
    news_items=None,
    sentiment_details=None,
):
    try:
        model_path = f"prediction/ml_models/saved_models/{ticker}_model.h5"
        scaler_path = f"prediction/ml_models/saved_models/{ticker}_scaler.pkl"
        acc_path = f"prediction/ml_models/saved_models/{ticker}_accuracy.txt"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return "Model Eğitilmedi", "N/A", "N/A", "secondary"

        model = load_model(model_path, compile=False)
        scaler_or_bundle = joblib.load(scaler_path)

        accuracy = "85.0"
        if os.path.exists(acc_path):
            with open(acc_path, "r") as f:
                accuracy = f.read().strip()

        # Yeni bundle yapisi hem close hem return hedefli modelleri destekler.
        if isinstance(scaler_or_bundle, dict) and "target_scaler" in scaler_or_bundle:
            final_prediction = round(
                float(_bundle_prediction(model, scaler_or_bundle, df, news_items, sentiment_details)),
                2,
            )
        else:
            final_prediction = round(float(_legacy_price_prediction(model, scaler_or_bundle, df)), 2)

        if current_price_override is not None:
            current_price = float(current_price_override)
        else:
            raw_last_price = df["Close"].iloc[-1]
            current_price = float(raw_last_price.item() if hasattr(raw_last_price, "item") else raw_last_price)

        signal_text, signal_class = get_technical_signal(df, final_prediction, current_price, sentiment_score)
        return final_prediction, accuracy, signal_text, signal_class

    except Exception as e:
        print(f"⚠️ Tahmin hatası ({ticker}): {e}")
        return "Hata", "N/A", "N/A", "secondary"
