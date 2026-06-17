from django.shortcuts import render
import os
import numpy as np
import glob
import time
import joblib
import json
from .services.data_fetcher import get_asset_display_name, get_market_news, get_realtime_quote, get_stock_data, get_stock_news, get_stock_profile
from .services.market_analysis import build_decision_explanation, build_market_summary, get_currency_symbol
from .services.sentiment import analyze_sentiment_details
from .ml_models.evaluate import evaluate_lstm_model
from .ml_models.predict import get_prediction
from .ml_models.train import train_model 

def _model_artifact_paths(model_dir, ticker):
    return [
        os.path.join(model_dir, f"{ticker}_model.h5"),
        os.path.join(model_dir, f"{ticker}_scaler.pkl"),
        os.path.join(model_dir, f"{ticker}_accuracy.txt"),
        os.path.join(model_dir, f"{ticker}_metrics.json"),
    ]


def _delete_model_artifacts(model_dir, ticker, reason):
    for path in _model_artifact_paths(model_dir, ticker):
        if os.path.exists(path):
            os.remove(path)
    print(f"♻️ Model cache: {ticker} temizlendi ({reason}).")


def _is_current_model_format(ticker):
    scaler_path = f"prediction/ml_models/saved_models/{ticker}_scaler.pkl"
    if not os.path.exists(scaler_path):
        return False
    try:
        bundle = joblib.load(scaler_path)
        return (
            isinstance(bundle, dict)
            and "target_scaler" in bundle
            and "feature_columns" in bundle
            and "target_type" in bundle
            and "target_column" in bundle
        )
    except Exception:
        return False


def _load_model_report(ticker):
    metrics_path = f"prediction/ml_models/saved_models/{ticker}_metrics.json"
    if not os.path.exists(metrics_path):
        return {}, [], {}
    try:
        with open(metrics_path, "r") as file:
            metrics = json.load(file)
        return (
            metrics,
            metrics.get("comparison", []),
            {
                "name": metrics.get("variant_name", "LSTM"),
                "sequence_length": metrics.get("sequence_length"),
                "feature_count": metrics.get("feature_count"),
            },
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return {}, [], {}


def limit_saved_models(max_models=3, max_age_days=7):
    """Model cache'ini sınırlar: en fazla N model ve en fazla belirli yaş."""
    model_dir = "prediction/ml_models/saved_models/"
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return

    models = glob.glob(os.path.join(model_dir, "*.h5"))
    model_sets = []
    now = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60

    for model_path in models:
        filename = os.path.basename(model_path)
        if not filename.endswith("_model.h5"):
            continue

        ticker = filename.replace("_model.h5", "")
        modified_at = os.path.getmtime(model_path)
        model_sets.append({
            "ticker": ticker,
            "model_path": model_path,
            "modified_at": modified_at,
        })

    for item in list(model_sets):
        if now - item["modified_at"] > max_age_seconds:
            try:
                _delete_model_artifacts(model_dir, item["ticker"], f"{max_age_days} günden eski")
                model_sets.remove(item)
            except Exception as e:
                print(f"⚠️ Model cache temizlik hatası ({item['ticker']}): {e}")

    if len(model_sets) >= max_models:
        model_sets.sort(key=lambda item: item["modified_at"])
        delete_count = len(model_sets) - max_models + 1
        for item in model_sets[:delete_count]:
            try:
                _delete_model_artifacts(model_dir, item["ticker"], f"en fazla {max_models} model sınırı")
            except Exception as e:
                print(f"⚠️ Model cache temizlik hatası ({item['ticker']}): {e}")

def index(request):
    context = {}
    if request.method == "POST":
        ticker = request.POST.get('ticker', '').strip().upper()
        display_name = get_asset_display_name(ticker)
        df = get_stock_data(ticker)
        
        if df is not None and not df.empty:
            realtime_quote = get_realtime_quote(ticker)

            # Haberler eğitimden önce hazırlanır; tarihli haber varsa LSTM + FinBERT feature'larına bağlanır.
            raw_news, news_diagnostics = get_stock_news(ticker, return_diagnostics=True)
            news_titles = [item["title"] for item in raw_news]
            sentiment_score, sentiment_details = analyze_sentiment_details(news_titles)
            detail_by_title = {item["title"]: item for item in sentiment_details}
            enriched_news = []
            for item in raw_news:
                detail = detail_by_title.get(item["title"], {})
                enriched_item = item.copy()
                enriched_item.update({
                    "sentiment_score": detail.get("score", 0.0),
                    "sentiment_label": detail.get("label", "N/A"),
                    "sentiment_class": detail.get("class", "secondary"),
                })
                enriched_news.append(enriched_item)

            # 1. Model Kontrolü ve Eğitimi
            model_path = f"prediction/ml_models/saved_models/{ticker}_model.h5"
            
            if not os.path.exists(model_path) or not _is_current_model_format(ticker):
                # Yeni model eğitimi öncesi hafıza temizliği
                limit_saved_models(max_models=3, max_age_days=7) 
                
                if len(df) > 60:
                    print(f"🧠 {ticker} için Multivariate LSTM eğitiliyor...")
                    train_model(df, ticker, raw_news, sentiment_details, news_diagnostics)
                else:
                    context['error'] = "Yeterli veri yok (En az 60 gün gerekli)."
            else:
                for path in _model_artifact_paths("prediction/ml_models/saved_models/", ticker):
                    if os.path.exists(path):
                        os.utime(path, None)

            # 2. Grafik İçin Veri Hazırlama (Son 90 İş Günü)
            recent_df = df.tail(90)
            chart_dates = recent_df.index.strftime('%Y-%m-%d').tolist()
            chart_prices = [round(float(p), 2) for p in recent_df['Close'].values.flatten()]
            chart_sma = [round(float(s), 2) if not np.isnan(s) else None for s in recent_df['SMA_20'].values.flatten()]
            chart_sma_50 = [round(float(s), 2) if not np.isnan(s) else None for s in recent_df['SMA_50'].values.flatten()]

            impact_news = sorted(
                enriched_news,
                key=lambda item: abs(item.get("sentiment_score", 0.0)),
                reverse=True,
            )[:6]

            # 4. Gelişmiş Tahmin ve Sinyal Mekanizması
            # prediction: Tahmin fiyatı
            # accuracy: legacy context adı; yeni yapıda performance score olarak kullanılır
            # signal_text: Al/Sat mesajı
            # signal_class: Bootstrap renk sınıfı (success, danger vb.)
            current_price = realtime_quote.get("last_price") if realtime_quote else None
            prediction, accuracy, signal_text, signal_class = get_prediction(
                ticker,
                df,
                sentiment_score,
                current_price,
                raw_news,
                sentiment_details,
            )
            numeric_prediction = prediction if isinstance(prediction, (int, float)) else None
            market_summary = build_market_summary(df, numeric_prediction, ticker, realtime_quote)
            decision_explanation = build_decision_explanation(market_summary, numeric_prediction, sentiment_score, signal_text)
            model_metrics, model_comparison, best_model = _load_model_report(ticker)
            sentiment_stats = {
                "positive": len([item for item in enriched_news if item.get("sentiment_class") == "success"]),
                "negative": len([item for item in enriched_news if item.get("sentiment_class") == "danger"]),
                "neutral": len([item for item in enriched_news if item.get("sentiment_class") == "secondary"]),
            }
            sentiment_total = max(1, sum(sentiment_stats.values()))
            sentiment_stats.update({
                "positive_pct": round(sentiment_stats["positive"] / sentiment_total * 100, 1),
                "negative_pct": round(sentiment_stats["negative"] / sentiment_total * 100, 1),
                "neutral_pct": round(sentiment_stats["neutral"] / sentiment_total * 100, 1),
            })
            
            context.update({
                'ticker': ticker,
                'display_name': display_name,
                'last_price': market_summary['last_price'],
                'market_summary': market_summary,
                'sentiment': sentiment_score,
                'prediction': prediction,
                'accuracy': accuracy,
                'signal_text': signal_text,
                'signal_class': signal_class,
                'chart_dates': json.dumps(chart_dates),
                'chart_prices': json.dumps(chart_prices),
                'chart_sma': json.dumps(chart_sma),
                'chart_sma_50': json.dumps(chart_sma_50),
                'news_list': enriched_news[:8],
                'impact_news': impact_news,
                'has_stock_news': bool(enriched_news),
                'decision_explanation': decision_explanation,
                'sentiment_stats': sentiment_stats,
                'model_metrics': model_metrics,
                'model_comparison': model_comparison,
                'best_model': best_model,
            })
        else:
            context['error'] = 'Hisse verisi çekilemedi. Sembolü kontrol edin.'

    return render(request, 'index.html', context)


def news_page(request):
    market_news = get_market_news()
    sectors = sorted({
        item.get('sector') or item.get('category')
        for item in market_news
        if item.get('sector') or item.get('category')
    })
    companies = sorted({
        item.get('company')
        for item in market_news
        if item.get('company') and item.get('company') != 'Genel'
    })

    return render(request, 'news.html', {
        'market_news': market_news,
        'sectors': sectors,
        'companies': companies,
    })


def stock_detail(request, ticker):
    ticker = ticker.strip().upper()
    display_name = get_asset_display_name(ticker)
    df = get_stock_data(ticker)
    if df is None or df.empty:
        return render(request, 'stock_detail.html', {
            'ticker': ticker,
            'error': 'Hisse verisi çekilemedi. Sembolü kontrol edin.',
        })

    realtime_quote = get_realtime_quote(ticker)
    current_price = realtime_quote.get("last_price") if realtime_quote else None
    raw_news, news_diagnostics = get_stock_news(ticker, return_diagnostics=True)
    sentiment_score, sentiment_details = analyze_sentiment_details([item["title"] for item in raw_news])
    prediction, accuracy, signal_text, signal_class = get_prediction(
        ticker,
        df,
        sentiment_score,
        current_price,
        raw_news,
        sentiment_details,
    )
    numeric_prediction = prediction if isinstance(prediction, (int, float)) else None
    market_summary = build_market_summary(df, numeric_prediction, ticker, realtime_quote)
    profile = get_stock_profile(ticker)
    decision_explanation = build_decision_explanation(market_summary, numeric_prediction, sentiment_score, signal_text)

    return render(request, 'stock_detail.html', {
        'ticker': ticker,
        'display_name': display_name,
        'profile': profile,
        'market_summary': market_summary,
        'prediction': prediction,
        'accuracy': accuracy,
        'signal_text': signal_text,
        'signal_class': signal_class,
        'sentiment': sentiment_score,
        'sentiment_details': sentiment_details[:6],
        'decision_explanation': decision_explanation,
    })


def lstm_performance_page(request):
    ticker = request.GET.get('ticker', 'AAPL').strip().upper()
    display_name = get_asset_display_name(ticker)
    df = get_stock_data(ticker)

    if df is None or df.empty:
        report = {
            "error": "Hisse verisi çekilemedi. Sembolü kontrol edin.",
            "metrics": {},
            "chart": {"dates": [], "actual": [], "predicted": []},
        }
    else:
        raw_news, news_diagnostics = get_stock_news(ticker, return_diagnostics=True)
        sentiment_score, sentiment_details = analyze_sentiment_details([item["title"] for item in raw_news])
        model_path = f"prediction/ml_models/saved_models/{ticker}_model.h5"
        if (not os.path.exists(model_path) or not _is_current_model_format(ticker)) and len(df) > 80:
            limit_saved_models(max_models=3, max_age_days=7)
            train_model(df, ticker, raw_news, sentiment_details, news_diagnostics)
        report = evaluate_lstm_model(ticker, df, raw_news, sentiment_details)

    return render(request, 'lstm_performance.html', {
        'ticker': ticker,
        'display_name': display_name,
        'report': report,
        'currency_symbol': get_currency_symbol(ticker),
    })
