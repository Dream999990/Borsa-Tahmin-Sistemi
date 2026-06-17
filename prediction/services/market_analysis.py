import math
import numpy as np


def _safe_float(value, default=0.0):
    try:
        if hasattr(value, "item"):
            value = value.item()
        if value is None or math.isnan(float(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _money(value):
    return round(_safe_float(value), 2)


def get_currency_symbol(ticker):
    ticker = ticker.upper() if ticker else ""
    if ticker.endswith(".IS") or ticker.endswith("-TRY") or ticker.endswith("TRY=X"):
        return "₺"
    return "$"


def _format_number(value, use_turkish_format=False):
    formatted = f"{_safe_float(value):,.2f}"
    if use_turkish_format:
        formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return formatted


def format_money(value, ticker=None):
    currency_symbol = get_currency_symbol(ticker)
    return f"{currency_symbol}{_format_number(value, currency_symbol == '₺')}"


def _quote_value(quote, key, fallback):
    if not quote:
        return fallback
    value = quote.get(key)
    return fallback if value is None else value


def _quote_volume_value(quote, fallback):
    value = _quote_value(quote, "volume", fallback)
    numeric_value = _safe_float(value, 0)
    return fallback if numeric_value <= 0 else value


def build_market_summary(df, prediction=None, ticker=None, quote=None):
    """Dashboard icin fiyat, risk ve teknik ozet metriklerini hazirlar."""
    clean_df = df.ffill().bfill()
    latest = clean_df.iloc[-1]
    previous = clean_df.iloc[-2] if len(clean_df) > 1 else latest

    last_price = _safe_float(_quote_value(quote, "last_price", latest.get("Close")))
    previous_close = _safe_float(_quote_value(quote, "previous_close", previous.get("Close")), last_price)
    change = last_price - previous_close
    change_percent = (change / previous_close * 100) if previous_close else 0.0

    last_30 = clean_df.tail(30)
    rsi = _safe_float(latest.get("RSI"), 50.0)
    sma_20 = _safe_float(latest.get("SMA_20"), last_price)
    sma_50 = _safe_float(latest.get("SMA_50"), last_price)
    volume = int(_safe_float(_quote_volume_value(quote, latest.get("Volume")), 0))
    avg_volume = int(_safe_float(clean_df["Volume"].tail(20).mean(), 0))
    volatility = _safe_float(clean_df["Daily_Return"].tail(60).std(), 0.0) * np.sqrt(252) * 100

    if rsi >= 70:
        rsi_label = "Aşırı Alım"
        rsi_class = "danger"
    elif rsi <= 30:
        rsi_label = "Aşırı Satım"
        rsi_class = "success"
    else:
        rsi_label = "Dengeli"
        rsi_class = "primary"

    trend_strength = "Pozitif" if last_price >= sma_20 >= sma_50 else "Zayıf"
    trend_class = "success" if trend_strength == "Pozitif" else "warning"

    risk_level = "Yüksek" if volatility >= 45 else "Orta" if volatility >= 25 else "Düşük"
    risk_class = "danger" if risk_level == "Yüksek" else "warning" if risk_level == "Orta" else "success"
    rsi_risk = abs(rsi - 50) * 1.2
    trend_risk = 0 if trend_strength == "Pozitif" else 18
    volatility_risk = min(volatility, 70)
    risk_score = round(min(100, (volatility_risk * 0.65) + rsi_risk + trend_risk), 1)

    prediction_change = None
    if isinstance(prediction, (int, float)):
        prediction_change = round(((prediction - last_price) / last_price) * 100, 2) if last_price else 0.0

    return {
        "currency_symbol": get_currency_symbol(ticker),
        "last_price": _money(last_price),
        "last_price_text": format_money(last_price, ticker),
        "previous_close": _money(previous_close),
        "previous_close_text": format_money(previous_close, ticker),
        "day_change": round(change, 2),
        "day_change_percent": round(change_percent, 2),
        "open_price": _money(_quote_value(quote, "open_price", latest.get("Open"))),
        "open_price_text": format_money(_quote_value(quote, "open_price", latest.get("Open")), ticker),
        "day_high": _money(_quote_value(quote, "day_high", latest.get("High"))),
        "day_high_text": format_money(_quote_value(quote, "day_high", latest.get("High")), ticker),
        "day_low": _money(_quote_value(quote, "day_low", latest.get("Low"))),
        "day_low_text": format_money(_quote_value(quote, "day_low", latest.get("Low")), ticker),
        "range_30_high": _money(last_30["High"].max()),
        "range_30_high_text": format_money(last_30["High"].max(), ticker),
        "range_30_low": _money(last_30["Low"].min()),
        "range_30_low_text": format_money(last_30["Low"].min(), ticker),
        "volume": f"{volume:,}".replace(",", "."),
        "avg_volume": f"{avg_volume:,}".replace(",", "."),
        "rsi": round(rsi, 2),
        "rsi_label": rsi_label,
        "rsi_class": rsi_class,
        "sma_20": _money(sma_20),
        "sma_20_text": format_money(sma_20, ticker),
        "sma_50": _money(sma_50),
        "sma_50_text": format_money(sma_50, ticker),
        "trend_strength": trend_strength,
        "trend_class": trend_class,
        "volatility": round(volatility, 2),
        "risk_level": risk_level,
        "risk_class": risk_class,
        "risk_score": risk_score,
        "prediction_change": prediction_change,
        "prediction_text": format_money(prediction, ticker) if isinstance(prediction, (int, float)) else prediction,
        "quote_source": quote.get("source", "Günlük kapanış verisi") if quote else "Günlük kapanış verisi",
        "is_realtime": bool(quote and quote.get("last_price") is not None),
    }


def build_decision_explanation(market_summary, prediction, sentiment_score, signal_text):
    """Al/sat kararinin hangi kanitlarla olustugunu kullaniciya anlatir."""
    reasons = []
    last_price = _safe_float(market_summary.get("last_price"))
    sma_20 = _safe_float(market_summary.get("sma_20"))
    rsi = _safe_float(market_summary.get("rsi"), 50.0)

    if isinstance(prediction, (int, float)):
        currency_symbol = market_summary.get("currency_symbol", "$")
        use_turkish_format = currency_symbol == "₺"
        prediction_text = f"{currency_symbol}{_format_number(prediction, use_turkish_format)}"
        last_price_text = f"{currency_symbol}{_format_number(last_price, use_turkish_format)}"
        if prediction > last_price:
            reasons.append(f"LSTM tahmini son fiyattan yüksek: {prediction_text} > {last_price_text}.")
        else:
            reasons.append(f"LSTM tahmini son fiyattan düşük: {prediction_text} < {last_price_text}.")
    else:
        reasons.append("LSTM tahmini bu sembol için henüz üretilemedi.")

    if rsi < 35:
        reasons.append(f"RSI {rsi} ile aşırı satım bölgesine yakın; tepki alımı ihtimali artar.")
    elif rsi > 65:
        reasons.append(f"RSI {rsi} ile aşırı alım bölgesine yakın; düzeltme riski artar.")
    else:
        reasons.append(f"RSI {rsi} ile dengeli bölgede.")

    if last_price >= sma_20:
        reasons.append(f"Fiyat SMA20 üzerinde; kısa vadeli trend destekleyici.")
    else:
        reasons.append(f"Fiyat SMA20 altında; kısa vadeli trend zayıf.")

    if sentiment_score > 0.15:
        reasons.append(f"Haber duygu skoru pozitif ({sentiment_score:.4f}); haber akışı kararı destekliyor.")
    elif sentiment_score < -0.15:
        reasons.append(f"Haber duygu skoru negatif ({sentiment_score:.4f}); haber akışı risk oluşturuyor.")
    else:
        reasons.append(f"Haber duygu skoru nötr ({sentiment_score:.4f}); karar daha çok teknik veriye dayanıyor.")

    return {
        "title": signal_text,
        "reasons": reasons,
    }
