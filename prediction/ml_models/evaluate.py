import json
import os

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from .train import (
    BASELINE_FEATURES,
    _build_sequences,
    _calculate_price_metrics,
    _evaluate_predictions,
    _calculate_naive_baseline,
    audit_sentiment_pipeline,
    prepare_modeling_frame,
)


def evaluate_lstm_model(ticker, df, news_items=None, sentiment_details=None):
    model_path = f"prediction/ml_models/saved_models/{ticker}_model.h5"
    scaler_path = f"prediction/ml_models/saved_models/{ticker}_scaler.pkl"
    metrics_path = f"prediction/ml_models/saved_models/{ticker}_metrics.json"

    empty_result = {
        "error": "Bu sembol için kayitli LSTM modeli bulunamadi. Once ana ekranda analiz calistirin.",
        "metrics": {},
        "comparison": [],
        "ablation": [],
        "feature_contribution": [],
        "best_model": {},
        "academic_report": {},
        "sentiment_alignment": {},
        "sentiment_audit": {},
        "sentiment_distribution": {},
        "chart": {"dates": [], "actual": [], "predicted": [], "naive": []},
    }

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return empty_result

    bundle = joblib.load(scaler_path)
    if not isinstance(bundle, dict):
        return _evaluate_legacy_lstm(ticker, df, model_path, bundle)

    feature_columns = bundle.get("feature_columns", [])
    sequence_length = int(bundle.get("sequence_length", 60))
    target_column = bundle.get("target_column", "target_return")
    target_type = bundle.get("target_type", "return")
    modeling_frame = prepare_modeling_frame(df, news_items, sentiment_details)
    split_row = int(len(modeling_frame) * 0.9)

    if not feature_columns or len(modeling_frame) <= sequence_length + 5:
        empty_result["error"] = "LSTM performans testi için yeterli test verisi yok."
        return empty_result

    scaled_features = bundle["feature_scaler"].transform(modeling_frame[feature_columns].values)
    scaled_target = bundle["target_scaler"].transform(modeling_frame[[target_column]].values).reshape(-1)
    _, _, X_test, _, test_target_indexes = _build_sequences(
        scaled_features,
        scaled_target,
        sequence_length,
        split_row,
    )

    if len(X_test) == 0:
        empty_result["error"] = "LSTM performans testi için yeterli test verisi yok."
        return empty_result

    model = load_model(model_path, compile=False)
    pred_scaled = model.predict(X_test, verbose=0)
    predicted_target = bundle["target_scaler"].inverse_transform(pred_scaled).reshape(-1)
    metrics = _evaluate_predictions(modeling_frame, predicted_target, test_target_indexes, target_type)
    metrics.update({
        "variant_key": bundle.get("variant_key", "lstm"),
        "variant_name": bundle.get("variant_name", "LSTM"),
        "sequence_length": sequence_length,
        "feature_count": len(feature_columns),
        "target_type": target_type,
    })

    comparison = []
    ablation = []
    feature_contribution = []
    sentiment_alignment = {}
    sentiment_audit = audit_sentiment_pipeline(df, news_items, sentiment_details)
    sentiment_distribution = {}
    academic_report = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as file:
                saved_metrics = json.load(file)
            comparison = saved_metrics.get("comparison", [])
            ablation = saved_metrics.get("ablation", [])
            feature_contribution = saved_metrics.get("feature_contribution", [])
            sentiment_alignment = saved_metrics.get("sentiment_alignment", {})
            sentiment_audit = saved_metrics.get("sentiment_audit", sentiment_audit)
            sentiment_distribution = saved_metrics.get("sentiment_distribution", {})
            academic_report = saved_metrics.get("academic_report", {})
            metrics.update({
                "training_performance_score": saved_metrics.get(
                    "performance_score",
                    metrics["performance_score"],
                ),
                "training_mape": saved_metrics.get("mape", metrics["mape"]),
                "training_direction_accuracy": saved_metrics.get(
                    "direction_accuracy",
                    metrics["direction_accuracy"],
                ),
                "performance_score_formula": saved_metrics.get(
                    "performance_score_formula",
                    metrics["performance_score_formula"],
                ),
            })
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    dates = modeling_frame.iloc[test_target_indexes].index.strftime("%Y-%m-%d").tolist()
    tail = min(80, len(dates))
    return {
        "error": "",
        "metrics": metrics,
        "comparison": comparison,
        "ablation": ablation,
        "feature_contribution": feature_contribution,
        "best_model": {
            "name": metrics.get("variant_name"),
            "sequence_length": metrics.get("sequence_length"),
            "feature_count": metrics.get("feature_count"),
        },
        "academic_report": academic_report,
        "sentiment_alignment": sentiment_alignment,
        "sentiment_audit": sentiment_audit,
        "sentiment_distribution": sentiment_distribution,
        "chart": {
            "dates": dates[-tail:],
            "actual": metrics["actual_prices"][-tail:],
            "predicted": metrics["predicted_prices"][-tail:],
            "naive": metrics["naive_baseline"]["predicted_prices"][-tail:],
            "residuals": metrics["residuals"][-tail:],
            "direction_hits": metrics["direction_report"]["correct_series"][-tail:],
            "actual_direction": metrics["direction_report"]["actual_direction_series"][-tail:],
            "predicted_direction": metrics["direction_report"]["predicted_direction_series"][-tail:],
        },
    }


def _evaluate_legacy_lstm(ticker, df, model_path, scaler):
    features = df[BASELINE_FEATURES].bfill().ffill()
    split_row = int(len(features) * 0.9)
    scaled_data = scaler.transform(features.values)
    X_test, actual_prices, predicted_prices, dates = [], [], [], []

    sequence_length = 60
    for i in range(sequence_length, len(scaled_data)):
        if i < split_row:
            continue
        X_test.append(scaled_data[i - sequence_length:i, :])
        actual_prices.append(float(features["Close"].iloc[i]))
        dates.append(features.index[i].strftime("%Y-%m-%d"))

    if not X_test:
        return {
            "error": "Eski LSTM modeli için yeterli test verisi yok.",
            "metrics": {},
            "comparison": [],
            "ablation": [],
            "feature_contribution": [],
            "best_model": {},
            "academic_report": {},
            "sentiment_alignment": {},
            "sentiment_audit": {},
            "sentiment_distribution": {},
            "chart": {"dates": [], "actual": [], "predicted": []},
        }

    model = load_model(model_path, compile=False)
    pred_scaled = model.predict(np.array(X_test), verbose=0)
    dummy = np.zeros((len(pred_scaled), len(BASELINE_FEATURES)))
    dummy[:, 0] = pred_scaled.reshape(-1)
    predicted_prices = scaler.inverse_transform(dummy)[:, 0]
    actual_prices = np.array(actual_prices)
    metrics = _calculate_price_metrics(actual_prices, predicted_prices)
    previous_prices = np.array([float(features["Close"].iloc[i - 1]) for i in range(sequence_length, len(scaled_data)) if i >= split_row])
    earlier_prices = np.array([float(features["Close"].iloc[i - 2]) for i in range(sequence_length, len(scaled_data)) if i >= split_row])
    naive_baseline = _calculate_naive_baseline(actual_prices, previous_prices, earlier_prices)
    metrics.update({
        "performance_score": 0.0,
        "accuracy": 0.0,
        "direction_accuracy": 0.0,
        "test_sample_count": int(len(actual_prices)),
        "variant_name": "Eski Baseline LSTM",
        "sequence_length": sequence_length,
        "feature_count": len(BASELINE_FEATURES),
        "performance_score_formula": "Legacy model formatinda hesaplanmadi.",
        "naive_baseline": naive_baseline,
        "lag_analysis": {},
        "return_analysis": {},
        "alignment_report": {},
    })

    tail = min(80, len(dates))
    return {
        "error": "",
        "metrics": metrics,
        "comparison": [],
        "ablation": [],
        "feature_contribution": [],
        "best_model": {
            "name": "Eski Baseline LSTM",
            "sequence_length": sequence_length,
            "feature_count": len(BASELINE_FEATURES),
        },
        "academic_report": {
            "academic_reliability": "Bu kayit eski model formatinda; guncel karsilastirma icin yeniden egitim gerekir."
        },
        "sentiment_alignment": {
            "legacy_notice": "Bu kayit eski model formatindadir; yeni karsilastirma icin model yeniden egitilmelidir.",
        },
        "sentiment_audit": {},
        "sentiment_distribution": {},
        "chart": {
            "dates": dates[-tail:],
            "actual": [round(float(v), 2) for v in actual_prices[-tail:]],
            "predicted": [round(float(v), 2) for v in predicted_prices[-tail:]],
            "naive": naive_baseline["predicted_prices"][-tail:],
            "residuals": [round(float(pred - act), 4) for pred, act in zip(predicted_prices[-tail:], actual_prices[-tail:])],
            "direction_hits": [],
            "actual_direction": [],
            "predicted_direction": [],
        },
    }
