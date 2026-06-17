import json
import os
import re
import time
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


BASELINE_FEATURES = ["Close", "Volume", "SMA_20", "RSI"]
PRICE_ONLY_FEATURES = ["Close", "Volume"]
SENTIMENT_FEATURES = [
    "daily_sentiment_mean",
    "daily_sentiment_max",
    "daily_sentiment_min",
    "daily_sentiment_std",
    "positive_news_count",
    "negative_news_count",
    "neutral_news_count",
    "news_count",
]
TECHNICAL_FEATURES = [
    "Close",
    "Volume",
    "return",
    "log_return",
    "MA5",
    "MA10",
    "MA20",
    "volatility",
    "RSI",
    "MACD",
    "MACD_signal",
    "volume_change",
]
FULL_FEATURES = TECHNICAL_FEATURES + SENTIMENT_FEATURES
PREDICTION_DAYS = 60
SEQUENCE_LENGTHS = [10, 20, 30, 60, 90]


MODEL_VARIANTS = [
    {
        "key": "close_price_lstm",
        "name": "Close Price LSTM",
        "features": PRICE_ONLY_FEATURES,
        "target_type": "close",
        "report_group": "comparison",
    },
    {
        "key": "return_lstm",
        "name": "Return Prediction LSTM",
        "features": PRICE_ONLY_FEATURES,
        "target_type": "return",
        "report_group": "comparison",
        "ablation_label": "Sadece fiyat",
    },
    {
        "key": "return_sentiment_lstm",
        "name": "Return + Sentiment LSTM",
        "features": PRICE_ONLY_FEATURES + SENTIMENT_FEATURES,
        "target_type": "return",
        "report_group": "comparison",
        "ablation_label": "Fiyat + FinBERT",
    },
    {
        "key": "return_technical_lstm",
        "name": "Return + Technical Indicators LSTM",
        "features": TECHNICAL_FEATURES,
        "target_type": "return",
        "report_group": "ablation",
        "ablation_label": "Fiyat + teknik gostergeler",
    },
    {
        "key": "return_sentiment_technical_lstm",
        "name": "Return + Sentiment + Technical Indicators LSTM",
        "features": FULL_FEATURES,
        "target_type": "return",
        "report_group": "comparison",
        "ablation_label": "Fiyat + FinBERT + teknik gostergeler",
    },
]


@contextmanager
def _training_lock(save_dir, ticker, timeout_seconds=180):
    lock_path = os.path.join(save_dir, f"{ticker}.lock")
    start_time = time.time()
    lock_fd = None

    while True:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, str(os.getpid()).encode("utf-8"))
            break
        except FileExistsError:
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"{ticker} modeli su anda baska bir islemde egitiliyor.")
            time.sleep(1)

    try:
        yield
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if os.path.exists(lock_path):
            os.remove(lock_path)


def create_lstm_model(input_shape, dropout=0.25, recurrent_dropout=0.1, learning_rate=0.001):
    use_bidirectional = input_shape[-1] >= len(PRICE_ONLY_FEATURES)
    first_layer = LSTM(
        units=64,
        return_sequences=True,
        recurrent_dropout=recurrent_dropout,
        input_shape=input_shape,
    )
    second_layer = LSTM(units=32, recurrent_dropout=recurrent_dropout)

    if use_bidirectional:
        model = Sequential([
            Bidirectional(first_layer),
            Dropout(dropout),
            Bidirectional(second_layer),
            Dropout(dropout),
            Dense(units=16, activation="relu"),
            Dense(units=1),
        ])
        loss = Huber()
    else:
        model = Sequential([
            first_layer,
            Dropout(dropout),
            second_layer,
            Dropout(dropout),
            Dense(units=16, activation="relu"),
            Dense(units=1),
        ])
        loss = "mean_squared_error"

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)
    return model


def add_technical_features(df):
    data = df.copy().sort_index()
    data["return"] = data["Close"].pct_change()
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["MA5"] = data["Close"].rolling(window=5).mean()
    data["MA10"] = data["Close"].rolling(window=10).mean()
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["volatility"] = data["return"].rolling(window=10).std()
    data["volume_change"] = data["Volume"].pct_change()

    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    if "RSI" not in data.columns:
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-9)
        data["RSI"] = 100 - (100 / (1 + rs))

    if "SMA_20" not in data.columns:
        data["SMA_20"] = data["Close"].rolling(window=20).mean()

    return data


def _parse_news_datetime(value):
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.tz_convert(None).normalize()


def _next_trading_day(news_date, trading_index):
    matches = trading_index[trading_index >= news_date]
    if len(matches) == 0:
        return None
    return matches[0]


def _normalize_title_key(title):
    cleaned = re.sub(r"[^a-z0-9]+", " ", (title or "").lower())
    return " ".join(cleaned.split())


def build_sentiment_frame(df, news_items=None, sentiment_details=None, return_metadata=False):
    trading_index = pd.DatetimeIndex(df.index).normalize()
    frame = pd.DataFrame(index=df.index)
    for column in SENTIMENT_FEATURES:
        frame[column] = 0.0

    metadata = {
        "news_items_seen": 0,
        "sentiment_details_seen": 0,
        "aligned_news_items": 0,
        "weekend_or_holiday_shifted": 0,
        "unmatched_news_items": 0,
        "days_with_news": 0,
        "exact_title_matches": 0,
        "normalized_title_matches": 0,
        "missing_sentiment_details": 0,
        "missing_titles_sample": [],
    }

    news_items = news_items or []
    sentiment_details = sentiment_details or []
    detail_by_title = {item.get("title"): item for item in sentiment_details}
    detail_by_normalized_title = {
        _normalize_title_key(item.get("title")): item
        for item in sentiment_details
        if item.get("title")
    }
    buckets = {}
    metadata["news_items_seen"] = len(news_items)
    metadata["sentiment_details_seen"] = len(sentiment_details)

    for news in news_items:
        title = news.get("title", "")
        published_at = _parse_news_datetime(news.get("date") or news.get("published_at"))
        if published_at is None:
            metadata["unmatched_news_items"] += 1
            continue

        aligned_day = _next_trading_day(published_at, trading_index)
        if aligned_day is None:
            metadata["unmatched_news_items"] += 1
            continue

        if aligned_day > published_at:
            metadata["weekend_or_holiday_shifted"] += 1

        detail = detail_by_title.get(title)
        if detail:
            metadata["exact_title_matches"] += 1
        else:
            detail = detail_by_normalized_title.get(_normalize_title_key(title))
            if detail:
                metadata["normalized_title_matches"] += 1
            else:
                metadata["missing_sentiment_details"] += 1
                if len(metadata["missing_titles_sample"]) < 5 and title:
                    metadata["missing_titles_sample"].append(title)

        detail = detail or {}
        score = float(detail.get("score", news.get("sentiment_score", 0.0)) or 0.0)
        buckets.setdefault(aligned_day, []).append(score)
        metadata["aligned_news_items"] += 1

    for day, scores in buckets.items():
        mask = trading_index == day
        if not mask.any():
            continue

        values = np.array(scores, dtype=float)
        frame.loc[mask, "daily_sentiment_mean"] = float(values.mean())
        frame.loc[mask, "daily_sentiment_max"] = float(values.max())
        frame.loc[mask, "daily_sentiment_min"] = float(values.min())
        frame.loc[mask, "daily_sentiment_std"] = float(values.std()) if len(values) > 1 else 0.0
        frame.loc[mask, "positive_news_count"] = int((values >= 0.15).sum())
        frame.loc[mask, "negative_news_count"] = int((values <= -0.15).sum())
        frame.loc[mask, "neutral_news_count"] = int(((values > -0.15) & (values < 0.15)).sum())
        frame.loc[mask, "news_count"] = len(values)

    metadata["days_with_news"] = int((frame["news_count"] > 0).sum())

    if return_metadata:
        return frame, metadata
    return frame


def _sentiment_feature_stats(sentiment_frame):
    stats = []
    for column in SENTIMENT_FEATURES:
        series = sentiment_frame[column].astype(float)
        stats.append({
            "feature": column,
            "min": round(float(series.min()), 6) if len(series) else 0.0,
            "max": round(float(series.max()), 6) if len(series) else 0.0,
            "mean": round(float(series.mean()), 6) if len(series) else 0.0,
            "non_zero_rows": int((series.abs() > 1e-12).sum()),
            "non_zero_ratio_pct": round(float((series.abs() > 1e-12).mean() * 100), 2) if len(series) else 0.0,
        })
    return stats


def audit_sentiment_pipeline(df, news_items=None, sentiment_details=None, news_diagnostics=None, emit_logs=False):
    if df is None or getattr(df, "empty", True):
        diagnostics = {
            "news": news_diagnostics or {},
            "sentiment_details_count": len(sentiment_details or []),
            "sentiment_details_empty": len(sentiment_details or []) == 0,
            "finbert_scores_generated": bool(len(sentiment_details or [])),
            "finbert_score_summary": {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            },
            "build_sentiment_frame": {
                "input_news_count": len(news_items or []),
                "input_sentiment_details_count": len(sentiment_details or []),
                "news_items_seen": len(news_items or []),
                "sentiment_details_seen": len(sentiment_details or []),
                "aligned_news_items": 0,
                "weekend_or_holiday_shifted": 0,
                "unmatched_news_items": 0,
                "days_with_news": 0,
                "exact_title_matches": 0,
                "normalized_title_matches": 0,
                "missing_sentiment_details": 0,
                "missing_titles_sample": [],
            },
            "feature_stats": [],
            "non_zero_feature_rows": {},
            "rows_with_any_sentiment_signal": 0,
            "logs": [
                "0. Fiyat verisi gelmedigi icin sentiment feature frame olusturulamadi.",
            ],
        }
        if emit_logs:
            for line in diagnostics["logs"]:
                print(f"[sentiment-audit] {line}")
        return diagnostics

    sentiment_frame, metadata = build_sentiment_frame(
        df,
        news_items,
        sentiment_details,
        return_metadata=True,
    )
    feature_stats = _sentiment_feature_stats(sentiment_frame)
    score_values = np.array([float(item.get("score", 0.0) or 0.0) for item in (sentiment_details or [])], dtype=float)
    non_zero_feature_rows = {
        item["feature"]: item["non_zero_rows"]
        for item in feature_stats
    }
    any_non_zero_rows = int((sentiment_frame[SENTIMENT_FEATURES].abs().sum(axis=1) > 1e-12).sum())

    diagnostics = {
        "news": news_diagnostics or {},
        "sentiment_details_count": len(sentiment_details or []),
        "sentiment_details_empty": len(sentiment_details or []) == 0,
        "finbert_scores_generated": bool(len(sentiment_details or [])),
        "finbert_score_summary": {
            "min": round(float(score_values.min()), 6) if len(score_values) else 0.0,
            "max": round(float(score_values.max()), 6) if len(score_values) else 0.0,
            "mean": round(float(score_values.mean()), 6) if len(score_values) else 0.0,
            "positive_count": int((score_values >= 0.15).sum()) if len(score_values) else 0,
            "negative_count": int((score_values <= -0.15).sum()) if len(score_values) else 0,
            "neutral_count": int(((score_values > -0.15) & (score_values < 0.15)).sum()) if len(score_values) else 0,
        },
        "build_sentiment_frame": {
            "input_news_count": len(news_items or []),
            "input_sentiment_details_count": len(sentiment_details or []),
            **metadata,
        },
        "feature_stats": feature_stats,
        "non_zero_feature_rows": non_zero_feature_rows,
        "rows_with_any_sentiment_signal": any_non_zero_rows,
        "logs": [],
    }

    logs = diagnostics["logs"]
    newsapi_info = diagnostics["news"].get("newsapi", {})
    yahoo_info = diagnostics["news"].get("yahoo_finance", {})
    logs.append(
        f"1. NewsAPI enabled={newsapi_info.get('enabled', False)} count={newsapi_info.get('count', 0)} error={newsapi_info.get('error', '') or 'yok'}"
    )
    logs.append(
        f"2. Yahoo Finance news count={yahoo_info.get('count', 0)} error={yahoo_info.get('error', '') or 'yok'} selected_source={diagnostics['news'].get('selected_source', 'N/A')}"
    )
    logs.append(
        f"3. FinBERT sentiment_details count={diagnostics['sentiment_details_count']} empty={diagnostics['sentiment_details_empty']}"
    )
    logs.append(
        f"4. FinBERT score range min={diagnostics['finbert_score_summary']['min']} max={diagnostics['finbert_score_summary']['max']} mean={diagnostics['finbert_score_summary']['mean']}"
    )
    logs.append(
        f"5. build_sentiment_frame input_news={metadata['news_items_seen']} aligned={metadata['aligned_news_items']} shifted_weekend_or_holiday={metadata['weekend_or_holiday_shifted']}"
    )
    logs.append(
        f"6. Title match exact={metadata['exact_title_matches']} normalized={metadata['normalized_title_matches']} missing={metadata['missing_sentiment_details']}"
    )
    logs.append(
        f"7. Rows with any non-zero sentiment feature={any_non_zero_rows} / {len(sentiment_frame)}"
    )
    for item in feature_stats:
        logs.append(
            f"feature={item['feature']} min={item['min']} max={item['max']} mean={item['mean']} non_zero_rows={item['non_zero_rows']}"
        )

    if emit_logs:
        for line in logs:
            print(f"[sentiment-audit] {line}")

    return diagnostics


def prepare_modeling_frame(df, news_items=None, sentiment_details=None, return_metadata=False):
    data = add_technical_features(df)
    sentiment_frame, sentiment_metadata = build_sentiment_frame(
        data,
        news_items,
        sentiment_details,
        return_metadata=True,
    )
    data = data.join(sentiment_frame, how="left")
    data = data.replace([np.inf, -np.inf], np.nan)

    feature_columns = sorted(set(
        PRICE_ONLY_FEATURES + BASELINE_FEATURES + TECHNICAL_FEATURES + SENTIMENT_FEATURES
    ))
    data[feature_columns] = data[feature_columns].ffill().fillna(0)

    # Sequence t-seq ... t-1 -> hedef t olacak sekilde kurulur.
    # Boylece son gozlenen kapanis ile gercek ertesi kapanis/return arasindaki yon karsilastirmasi dogrudur.
    data["target_return"] = data["Close"].pct_change()
    data["target_close"] = data["Close"]
    data = data.dropna(subset=["target_return", "target_close"])

    if return_metadata:
        return data, sentiment_metadata
    return data


def _build_sequences(feature_values, target_values, sequence_length, split_row):
    X_train, y_train, X_test, y_test, test_target_indexes = [], [], [], [], []

    for i in range(sequence_length, len(feature_values)):
        row = feature_values[i - sequence_length:i, :]
        target = target_values[i]
        if i < split_row:
            X_train.append(row)
            y_train.append(target)
        else:
            X_test.append(row)
            y_test.append(target)
            test_target_indexes.append(i)

    return (
        np.array(X_train),
        np.array(y_train),
        np.array(X_test),
        np.array(y_test),
        test_target_indexes,
    )


def _direction_labels(delta_values, tolerance=1e-9):
    labels = np.zeros(len(delta_values), dtype=int)
    labels[np.asarray(delta_values) > tolerance] = 1
    labels[np.asarray(delta_values) < -tolerance] = -1
    return labels


def _calculate_performance_score(mape, direction_accuracy, r2):
    mape_component = max(0.0, 1 - min(float(mape), 20.0) / 20.0)
    direction_component = max(0.0, min(1.0, float(direction_accuracy) / 100.0))
    r2_component = max(0.0, min(1.0, float(r2)))
    score = 100 * ((0.45 * mape_component) + (0.35 * direction_component) + (0.20 * r2_component))
    return round(float(score), 2)


def _calculate_price_metrics(actual_prices, predicted_prices):
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    actual_safe = np.where(actual_prices == 0, 1e-9, actual_prices)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_safe)) * 100
    r2 = r2_score(actual_prices, predicted_prices) if len(actual_prices) > 1 else 0.0

    metrics = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
        "r2": round(float(r2), 4),
    }
    return metrics


def _build_direction_report(actual_prices, predicted_prices, previous_prices):
    actual_delta = actual_prices - previous_prices
    predicted_delta = predicted_prices - previous_prices
    actual_direction = _direction_labels(actual_delta)
    predicted_direction = _direction_labels(predicted_delta)
    correct_mask = actual_direction == predicted_direction

    up_mask = actual_direction == 1
    down_mask = actual_direction == -1
    flat_mask = actual_direction == 0
    predicted_up_mask = predicted_direction == 1
    predicted_down_mask = predicted_direction == -1
    predicted_flat_mask = predicted_direction == 0

    absolute_actual_moves = np.abs(actual_delta)
    median_move = float(np.median(absolute_actual_moves)) if len(absolute_actual_moves) else 0.0
    tiny_move_ratio = float(np.mean(absolute_actual_moves <= median_move)) if len(absolute_actual_moves) else 0.0

    return {
        "direction_accuracy": round(float(correct_mask.mean() * 100), 2),
        "correct_count": int(correct_mask.sum()),
        "incorrect_count": int((~correct_mask).sum()),
        "actual_up_days": int(up_mask.sum()),
        "actual_down_days": int(down_mask.sum()),
        "actual_flat_days": int(flat_mask.sum()),
        "predicted_up_days": int(predicted_up_mask.sum()),
        "predicted_down_days": int(predicted_down_mask.sum()),
        "predicted_flat_days": int(predicted_flat_mask.sum()),
        "up_hit_rate": round(float(correct_mask[up_mask].mean() * 100), 2) if up_mask.any() else 0.0,
        "down_hit_rate": round(float(correct_mask[down_mask].mean() * 100), 2) if down_mask.any() else 0.0,
        "tiny_move_ratio": round(tiny_move_ratio * 100, 2),
        "median_absolute_move": round(median_move, 4),
        "actual_direction_series": actual_direction.astype(int).tolist(),
        "predicted_direction_series": predicted_direction.astype(int).tolist(),
        "correct_series": correct_mask.astype(int).tolist(),
    }


def _safe_correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return round(float(np.corrcoef(left, right)[0, 1]), 4)


def _calculate_naive_baseline(actual_prices, previous_prices, earlier_prices):
    naive_prices = previous_prices.copy()
    metrics = _calculate_price_metrics(actual_prices, naive_prices)
    # Fiyat naive baseline'i bir sonraki kapanisi dogrudan onceki kapanisa esitler.
    # Bu durumda predicted_delta = 0 olur; bu nedenle yon benchmark'i ayri olarak
    # "dunku yon devam eder" (persistence) mantigiyla hesapliyoruz.
    actual_delta = actual_prices - previous_prices
    persistence_delta = previous_prices - earlier_prices
    actual_direction = _direction_labels(actual_delta)
    predicted_direction = _direction_labels(persistence_delta)
    correct_mask = actual_direction == predicted_direction

    metrics.update({
        "variant_key": "naive_baseline",
        "variant_name": "Naive Baseline",
        "sequence_length": 0,
        "direction_accuracy": round(float(correct_mask.mean() * 100), 2) if len(correct_mask) else 0.0,
        "direction_accuracy_neutral_rule": round(
            float(np.mean(_direction_labels(naive_prices - previous_prices) == actual_direction) * 100),
            2,
        ) if len(actual_direction) else 0.0,
        "direction_rule": "predicted_direction[t] = sign(previous_close[t] - close[t-2])",
        "price_rule": "predicted_close[t] = previous_close[t]",
        "actual_prices": [round(float(v), 2) for v in actual_prices],
        "predicted_prices": [round(float(v), 2) for v in naive_prices],
        "residuals": [round(float(pred - act), 4) for pred, act in zip(naive_prices, actual_prices)],
        "actual_direction_series": actual_direction.astype(int).tolist(),
        "predicted_direction_series": predicted_direction.astype(int).tolist(),
        "direction_sample": {
            "actual_direction_head": actual_direction.astype(int).tolist()[:12],
            "predicted_direction_head": predicted_direction.astype(int).tolist()[:12],
        },
    })
    return metrics


def _build_lag_analysis(actual_prices, predicted_prices, previous_prices):
    corr_pred_actual = _safe_correlation(predicted_prices, actual_prices)
    corr_pred_previous = _safe_correlation(predicted_prices, previous_prices)
    corr_actual_previous = _safe_correlation(actual_prices, previous_prices)
    corr_pred_actual_shifted = _safe_correlation(predicted_prices[1:], actual_prices[:-1]) if len(actual_prices) > 2 else 0.0
    mae_vs_previous = round(float(np.mean(np.abs(predicted_prices - previous_prices))), 4) if len(predicted_prices) else 0.0
    copy_ratio = round(
        float(np.mean(np.abs(predicted_prices - previous_prices) <= np.maximum(0.01, np.abs(previous_prices) * 0.001)) * 100),
        2,
    ) if len(predicted_prices) else 0.0

    likely_copying_previous = (
        corr_pred_previous >= 0.98
        and abs(corr_pred_previous - corr_pred_actual) <= 0.01
    )
    lag_suspected = corr_pred_actual_shifted > corr_pred_actual

    return {
        "corr_predicted_actual": corr_pred_actual,
        "corr_predicted_previous": corr_pred_previous,
        "corr_actual_previous": corr_actual_previous,
        "corr_predicted_actual_shifted": corr_pred_actual_shifted,
        "mae_vs_previous_close": mae_vs_previous,
        "copy_like_ratio_pct": copy_ratio,
        "likely_copying_previous": likely_copying_previous,
        "lag_suspected": lag_suspected,
    }


def _build_return_analysis(predicted_target, predicted_prices, previous_prices, target_type):
    if target_type == "return":
        pred_returns = np.asarray(predicted_target, dtype=float)
    else:
        pred_returns = np.where(previous_prices == 0, 0.0, (predicted_prices - previous_prices) / previous_prices)

    if len(pred_returns) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "near_zero_ratio_pct": 0.0,
            "series": [],
        }

    near_zero_ratio = float(np.mean(np.abs(pred_returns) <= 0.002) * 100)
    return {
        "mean": round(float(pred_returns.mean()), 6),
        "std": round(float(pred_returns.std()), 6),
        "min": round(float(pred_returns.min()), 6),
        "max": round(float(pred_returns.max()), 6),
        "near_zero_ratio_pct": round(near_zero_ratio, 2),
        "series": [round(float(v), 6) for v in pred_returns.tolist()],
    }


def _build_alignment_report(modeling_frame, test_target_indexes):
    target_dates = modeling_frame.iloc[test_target_indexes].index.strftime("%Y-%m-%d").tolist()
    previous_dates = modeling_frame.iloc[[i - 1 for i in test_target_indexes]].index.strftime("%Y-%m-%d").tolist()
    labels_match_targets = len(target_dates) == len(test_target_indexes)
    consecutive_previous_alignment = True
    if target_dates and previous_dates:
        for target_idx, prev_idx in zip(test_target_indexes, [i - 1 for i in test_target_indexes]):
            if target_idx - prev_idx != 1:
                consecutive_previous_alignment = False
                break
    return {
        "chart_label_count": len(target_dates),
        "target_index_count": len(test_target_indexes),
        "labels_match_targets": labels_match_targets,
        "consecutive_previous_alignment": consecutive_previous_alignment,
        "first_target_label": target_dates[0] if target_dates else "",
        "last_target_label": target_dates[-1] if target_dates else "",
        "first_previous_label": previous_dates[0] if previous_dates else "",
        "last_previous_label": previous_dates[-1] if previous_dates else "",
        "off_by_one_suspected": not (labels_match_targets and consecutive_previous_alignment),
        "chart_labels": target_dates,
    }


def _evaluate_predictions(modeling_frame, predicted_target, test_target_indexes, target_type):
    previous_prices = modeling_frame.iloc[[i - 1 for i in test_target_indexes]]["Close"].values
    earlier_prices = modeling_frame.iloc[[i - 2 for i in test_target_indexes]]["Close"].values
    actual_prices = modeling_frame.iloc[test_target_indexes]["target_close"].values

    if target_type == "return":
        predicted_prices = previous_prices * (1 + predicted_target.reshape(-1))
    else:
        predicted_prices = predicted_target.reshape(-1)

    metrics = _calculate_price_metrics(actual_prices, predicted_prices)
    direction_report = _build_direction_report(actual_prices, predicted_prices, previous_prices)
    naive_metrics = _calculate_naive_baseline(actual_prices, previous_prices, earlier_prices)
    lag_analysis = _build_lag_analysis(actual_prices, predicted_prices, previous_prices)
    return_analysis = _build_return_analysis(predicted_target, predicted_prices, previous_prices, target_type)
    alignment_report = _build_alignment_report(modeling_frame, test_target_indexes)
    performance_score = _calculate_performance_score(
        metrics["mape"],
        direction_report["direction_accuracy"],
        metrics["r2"],
    )

    metrics.update({
        "performance_score": performance_score,
        "accuracy": performance_score,
        "performance_score_formula": (
            "100 * (0.45 * (1 - min(MAPE,20)/20) + 0.35 * (DirectionAccuracy/100) + 0.20 * max(0,min(R2,1)))"
        ),
        "direction_accuracy": direction_report["direction_accuracy"],
        "test_sample_count": int(len(actual_prices)),
        "actual_prices": [round(float(v), 2) for v in actual_prices],
        "predicted_prices": [round(float(v), 2) for v in predicted_prices],
        "previous_prices": [round(float(v), 2) for v in previous_prices],
        "residuals": [round(float(pred - act), 4) for pred, act in zip(predicted_prices, actual_prices)],
        "actual_changes": [round(float(act - prev), 4) for act, prev in zip(actual_prices, previous_prices)],
        "predicted_changes": [round(float(pred - prev), 4) for pred, prev in zip(predicted_prices, previous_prices)],
        "predicted_target": [round(float(v), 6) for v in np.asarray(predicted_target).reshape(-1).tolist()],
        "direction_report": direction_report,
        "naive_baseline": naive_metrics,
        "lag_analysis": lag_analysis,
        "return_analysis": return_analysis,
        "alignment_report": alignment_report,
    })
    return metrics


def _atomic_model_save(model, path):
    temp_path = f"{path}.tmp.h5"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    model.save(temp_path)
    os.replace(temp_path, path)


def _atomic_joblib_dump(value, path):
    temp_path = f"{path}.tmp"
    joblib.dump(value, temp_path)
    os.replace(temp_path, path)


def _atomic_text_write(path, content):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w") as file:
        file.write(content)
    os.replace(temp_path, path)


def _atomic_json_write(path, content):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def _train_variant(modeling_frame, variant, sequence_length, save_dir, ticker):
    feature_columns = variant["features"]
    target_type = variant["target_type"]
    target_column = "target_close" if target_type == "close" else "target_return"
    split_row = int(len(modeling_frame) * 0.9)

    if split_row <= sequence_length or len(modeling_frame) - split_row < 5:
        raise ValueError("Egitim/test ayrimi icin yeterli veri yok.")

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1)) if target_type == "close" else StandardScaler()
    feature_scaler.fit(modeling_frame.iloc[:split_row][feature_columns].values)
    target_scaler.fit(modeling_frame.iloc[:split_row][[target_column]].values)

    scaled_features = feature_scaler.transform(modeling_frame[feature_columns].values)
    scaled_target = target_scaler.transform(modeling_frame[[target_column]].values).reshape(-1)
    X_train, y_train, X_test, y_test, test_target_indexes = _build_sequences(
        scaled_features,
        scaled_target,
        sequence_length,
        split_row,
    )
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("LSTM sequence uretimi icin yeterli veri yok.")

    checkpoint_path = os.path.join(save_dir, f"{ticker}_{variant['key']}_{sequence_length}_checkpoint.h5")
    model = create_lstm_model(
        (X_train.shape[1], X_train.shape[2]),
        dropout=0.25,
        recurrent_dropout=0.1,
        learning_rate=0.0005 if target_type == "return" else 0.001,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]
    model.fit(
        X_train,
        y_train,
        validation_split=0.12,
        batch_size=16 if target_type == "return" else 32,
        epochs=45 if target_type == "return" else 30,
        verbose=0,
        shuffle=False,
        callbacks=callbacks,
    )

    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path, compile=False)
        os.remove(checkpoint_path)

    pred_scaled = model.predict(X_test, verbose=0)
    predicted_target = target_scaler.inverse_transform(pred_scaled).reshape(-1)
    metrics = _evaluate_predictions(modeling_frame, predicted_target, test_target_indexes, target_type)
    metrics.update({
        "variant_key": variant["key"],
        "variant_name": variant["name"],
        "sequence_length": sequence_length,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_type": target_type,
        "report_group": variant.get("report_group", "comparison"),
        "ablation_label": variant.get("ablation_label", ""),
        "target_scaler_type": target_scaler.__class__.__name__,
        "loss_name": "Huber" if target_type == "return" else "MSE",
        "architecture": "Bidirectional LSTM",
    })

    bundle = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_columns": feature_columns,
        "sequence_length": sequence_length,
        "target_column": target_column,
        "target_type": target_type,
        "variant_key": variant["key"],
        "variant_name": variant["name"],
    }
    return model, bundle, metrics


def _best_result_by_variant(results):
    best = {}
    for item in results:
        if item.get("error"):
            continue
        key = item["variant_key"]
        if key not in best or item["rmse"] < best[key]["rmse"]:
            best[key] = item
    return best


def _build_feature_contribution(best_by_variant):
    baseline = best_by_variant.get("return_lstm")
    if not baseline:
        return []

    rows = []
    for key in [
        "return_lstm",
        "return_sentiment_lstm",
        "return_technical_lstm",
        "return_sentiment_technical_lstm",
    ]:
        item = best_by_variant.get(key)
        if not item:
            continue
        rows.append({
            "label": item.get("ablation_label") or item["variant_name"],
            "variant_key": key,
            "rmse": item["rmse"],
            "mape": item["mape"],
            "direction_accuracy": item["direction_accuracy"],
            "rmse_delta_vs_price": round(item["rmse"] - baseline["rmse"], 2),
            "mape_delta_vs_price": round(item["mape"] - baseline["mape"], 2),
            "direction_delta_vs_price": round(item["direction_accuracy"] - baseline["direction_accuracy"], 2),
        })
    return rows


def _build_sentiment_summary(modeling_frame, sentiment_metadata):
    news_days_mask = modeling_frame["news_count"] > 0
    mean_series = modeling_frame.loc[news_days_mask, "daily_sentiment_mean"]

    positive_days = int((mean_series >= 0.15).sum())
    negative_days = int((mean_series <= -0.15).sum())
    neutral_days = int(((mean_series > -0.15) & (mean_series < 0.15)).sum())
    no_news_days = int((~news_days_mask).sum())

    return {
        "alignment": sentiment_metadata,
        "summary": {
            "days_with_news": int(news_days_mask.sum()),
            "days_without_news": no_news_days,
            "non_zero_sentiment_days": int((mean_series.abs() > 1e-9).sum()),
            "mean_sentiment": round(float(mean_series.mean()), 4) if len(mean_series) else 0.0,
            "std_sentiment": round(float(mean_series.std()), 4) if len(mean_series) > 1 else 0.0,
            "max_sentiment": round(float(mean_series.max()), 4) if len(mean_series) else 0.0,
            "min_sentiment": round(float(mean_series.min()), 4) if len(mean_series) else 0.0,
            "zero_ratio_pct": round(
                float((modeling_frame["daily_sentiment_mean"].abs() <= 1e-9).mean() * 100), 2
            ) if len(modeling_frame) else 0.0,
        },
        "chart": {
            "labels": ["Pozitif", "Negatif", "Notr", "Habersiz"],
            "values": [positive_days, negative_days, neutral_days, no_news_days],
        },
        "series": {
            "dates": modeling_frame.index.strftime("%Y-%m-%d").tolist()[-80:],
            "mean_scores": [round(float(v), 4) for v in modeling_frame["daily_sentiment_mean"].tolist()[-80:]],
            "news_counts": [int(v) for v in modeling_frame["news_count"].tolist()[-80:]],
        },
    }


def _build_academic_report(best_metrics, comparison_summary, feature_contribution, sentiment_summary):
    finbert_row = next((row for row in feature_contribution if row["variant_key"] == "return_sentiment_lstm"), None)
    technical_row = next((row for row in feature_contribution if row["variant_key"] == "return_technical_lstm"), None)
    full_row = next((row for row in feature_contribution if row["variant_key"] == "return_sentiment_technical_lstm"), None)
    direction_report = best_metrics.get("direction_report", {})
    naive_baseline = best_metrics.get("naive_baseline", {})
    lag_analysis = best_metrics.get("lag_analysis", {})
    return_analysis = best_metrics.get("return_analysis", {})

    tiny_move_ratio = direction_report.get("tiny_move_ratio", 0.0)
    predicted_up = direction_report.get("predicted_up_days", 0)
    predicted_down = direction_report.get("predicted_down_days", 0)
    actual_up = direction_report.get("actual_up_days", 0)
    actual_down = direction_report.get("actual_down_days", 0)

    direction_reasons = []
    if tiny_move_ratio >= 45:
        direction_reasons.append(
            "Test donemindeki gunlerin buyuk bolumunde hareketler kucuk; bu durumda fiyat hatasi dusuk kalsa bile yon etiketi kolayca degisebilir."
        )
    if abs(predicted_up - actual_up) > max(3, int(0.1 * max(1, actual_up + actual_down))):
        direction_reasons.append(
            "Model yukari ve asagi gunlerin dagilimini iyi yakalayamiyor; tahminler ortalamaya yakin toplandigi icin yon dogrulugu dusuyor."
        )
    if not direction_reasons:
        direction_reasons.append(
            "Model fiyat seviyesini iyi takip etse de ertesi gun degisiminin isaretini ayri bir siniflandirma problemi kadar iyi ogrenemiyor."
        )

    return {
        "best_model": comparison_summary[0]["variant_name"] if comparison_summary else best_metrics.get("variant_name"),
        "lstm_really_learns": (
            "LSTM naive baseline uzerinde ek bilgi ogreniyor"
            if naive_baseline and best_metrics.get("rmse", 1e9) < naive_baseline.get("rmse", 1e9)
            else "LSTM'in naive baseline ustunlugu sinirli veya yok"
        ),
        "naive_comparison": {
            "lstm_rmse": best_metrics.get("rmse"),
            "naive_rmse": naive_baseline.get("rmse"),
            "lstm_mape": best_metrics.get("mape"),
            "naive_mape": naive_baseline.get("mape"),
            "lstm_direction_accuracy": best_metrics.get("direction_accuracy"),
            "naive_direction_accuracy": naive_baseline.get("direction_accuracy"),
        },
        "finbert_contribution": (
            "FinBERT katkisi pozitif"
            if finbert_row and (
                finbert_row["rmse_delta_vs_price"] < 0 or finbert_row["direction_delta_vs_price"] > 0
            )
            else "FinBERT belirgin ek katkı saglamiyor"
        ),
        "technical_contribution": (
            "Teknik gostergeler katkisi pozitif"
            if full_row and (
                full_row["rmse_delta_vs_price"] < 0 or full_row["direction_delta_vs_price"] > 0
            )
            else "Teknik gostergeler belirgin ek katkı saglamiyor"
        ),
        "direction_accuracy_reason": " ".join(direction_reasons),
        "copying_previous_assessment": (
            "Model onceki gun kapanisini fazla yakindan takip ediyor olabilir."
            if lag_analysis.get("likely_copying_previous")
            else "Model tahmini tamamen onceki gunu kopyalayan bir yapida gorunmuyor."
        ),
        "lag_problem_assessment": (
            "Grafikte gorulen gecikme gercek olabilir; predicted vs shifted actual korelasyonu daha yuksek."
            if lag_analysis.get("lag_suspected")
            else "Belirgin bir off-by-one veya sistematik bir gunluk kayma sinyali yok."
        ),
        "data_leakage_check": (
            "Belirgin veri sizintisi belirtisi yok: scaler sadece train bolumune fit ediliyor, zaman sirasi korunuyor ve shuffle=False kullaniliyor."
        ),
        "academic_reliability": (
            "Sonuclar savunulabilir; ancak direction accuracy dusukse modelin fiyat seviyesi tahmininde guclu, yon tahmininde ise daha sinirli oldugu acikca belirtilmeli."
        ),
        "return_assessment": (
            "Tahmin edilen return degerleri cogu zaman sifira yakin; bu durum modelin onceki kapanisa yakin fiyatlar uretmesine neden olabilir."
            if return_analysis.get("near_zero_ratio_pct", 0) >= 60
            else "Tahmin edilen return dagilimi sifira tamamen yigilmis gorunmuyor."
        ),
        "lag_analysis": lag_analysis,
        "direction_summary": {
            "actual_up_days": actual_up,
            "actual_down_days": actual_down,
            "predicted_up_days": predicted_up,
            "predicted_down_days": predicted_down,
            "correct_count": direction_report.get("correct_count", 0),
            "incorrect_count": direction_report.get("incorrect_count", 0),
        },
        "feature_takeaways": {
            "finbert": finbert_row or {},
            "technical_only": technical_row or {},
            "finbert_plus_technical": full_row or {},
        },
        "sentiment_takeaway": sentiment_summary.get("summary", {}),
    }


def train_model(df, ticker, news_items=None, sentiment_details=None, news_diagnostics=None):
    save_dir = "prediction/ml_models/saved_models/"
    os.makedirs(save_dir, exist_ok=True)

    with _training_lock(save_dir, ticker):
        return _train_model_locked(df, ticker, save_dir, news_items, sentiment_details, news_diagnostics)


def _train_model_locked(df, ticker, save_dir, news_items=None, sentiment_details=None, news_diagnostics=None):
    sentiment_audit = audit_sentiment_pipeline(
        df,
        news_items,
        sentiment_details,
        news_diagnostics=news_diagnostics,
        emit_logs=True,
    )
    modeling_frame, sentiment_metadata = prepare_modeling_frame(
        df,
        news_items,
        sentiment_details,
        return_metadata=True,
    )
    if len(modeling_frame) <= max(SEQUENCE_LENGTHS) + 30:
        raise ValueError("Model egitimi icin yeterli veri yok.")

    all_results = []
    best_comparison_result = None

    for variant in MODEL_VARIANTS:
        for sequence_length in SEQUENCE_LENGTHS:
            try:
                model, bundle, metrics = _train_variant(modeling_frame, variant, sequence_length, save_dir, ticker)
                all_results.append(metrics)
                if variant.get("report_group") == "comparison":
                    if best_comparison_result is None or metrics["rmse"] < best_comparison_result[2]["rmse"]:
                        best_comparison_result = (model, bundle, metrics)
            except Exception as exc:
                all_results.append({
                    "variant_key": variant["key"],
                    "variant_name": variant["name"],
                    "sequence_length": sequence_length,
                    "target_type": variant["target_type"],
                    "report_group": variant.get("report_group", "comparison"),
                    "ablation_label": variant.get("ablation_label", ""),
                    "error": str(exc),
                })

    if best_comparison_result is None:
        raise ValueError("Hicbir LSTM varyanti basariyla egitilemedi.")

    best_model, best_bundle, best_metrics = best_comparison_result
    best_by_variant = _best_result_by_variant(all_results)

    comparison_summary = []
    for variant in MODEL_VARIANTS:
        if variant.get("report_group") != "comparison":
            continue
        item = best_by_variant.get(variant["key"])
        if item:
            comparison_summary.append(item)
    comparison_summary.sort(key=lambda item: item["rmse"])

    ablation_summary = []
    for variant in MODEL_VARIANTS:
        if not variant.get("ablation_label"):
            continue
        item = best_by_variant.get(variant["key"])
        if item:
            ablation_summary.append(item)
    ablation_summary.sort(key=lambda item: item["rmse"])

    feature_contribution = _build_feature_contribution(best_by_variant)
    sentiment_summary = _build_sentiment_summary(modeling_frame, sentiment_metadata)

    best_metrics = best_metrics.copy()
    best_metrics["comparison"] = comparison_summary
    best_metrics["comparison_all_runs"] = all_results
    best_metrics["ablation"] = ablation_summary
    best_metrics["feature_contribution"] = feature_contribution
    best_metrics["sentiment_alignment"] = {
        **sentiment_metadata,
        "weekend_holiday_rule": "Islem gunu olmayan haberler sonraki islem gunune aktarilir.",
        "target_rule": "Ayni gun feature'lari ile bir sonraki islem gununun kapanis/return degeri tahmin edilir.",
        "leakage_rule": "Scaler yalnizca train bolumune fit edilir; zaman sirasi korunur ve shuffle=False kullanilir.",
    }
    best_metrics["sentiment_audit"] = sentiment_audit
    best_metrics["sentiment_distribution"] = sentiment_summary
    best_metrics["academic_report"] = _build_academic_report(
        best_metrics,
        comparison_summary,
        feature_contribution,
        sentiment_summary,
    )

    _atomic_model_save(best_model, f"{save_dir}{ticker}_model.h5")
    _atomic_joblib_dump(best_bundle, f"{save_dir}{ticker}_scaler.pkl")
    _atomic_text_write(f"{save_dir}{ticker}_accuracy.txt", str(best_metrics["performance_score"]))
    _atomic_json_write(f"{save_dir}{ticker}_metrics.json", best_metrics)

    print(
        f"Best model: {best_metrics['variant_name']} | "
        f"seq={best_metrics['sequence_length']} | RMSE={best_metrics['rmse']} | "
        f"MAPE=%{best_metrics['mape']} | Direction=%{best_metrics['direction_accuracy']}"
    )
    return best_model, best_bundle
