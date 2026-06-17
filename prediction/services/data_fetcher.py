import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import re
import requests
import numpy as np

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()

COMMON_ASSET_NAMES = {
    "BTC-USD": "Bitcoin",
    "BTC-TRY": "Bitcoin",
    "ETH-USD": "Ethereum",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "GOOG": "Alphabet",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "NFLX": "Netflix",
    "AMD": "Advanced Micro Devices",
    "THYAO.IS": "Türk Hava Yolları",
    "GARAN.IS": "Garanti BBVA",
    "ASELS.IS": "Aselsan",
    "EREGL.IS": "Erdemir",
    "SISE.IS": "Şişecam",
}


def get_asset_display_name(ticker_symbol):
    ticker = ticker_symbol.upper() if ticker_symbol else ""
    return COMMON_ASSET_NAMES.get(ticker, ticker_symbol)


def _matches_keyword(text, keyword):
    keyword = keyword.lower()
    if len(keyword) <= 4 or keyword.isupper():
        return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None
    return keyword in text


def infer_news_sector(title="", summary="", source="", fallback="Genel Piyasa"):
    """Haber basligi ve ozetinden sektor tahmini yapar."""
    text = f"{title} {summary} {source}".lower()
    sector_keywords = [
        ("Kripto", ["bitcoin", "btc", "ethereum", "crypto", "kripto", "blockchain", "coinbase"]),
        ("Enerji ve Emtia", ["gold", "oil", "brent", "silver", "commodity", "petrol", "altın", "gümüş", "natural gas"]),
        ("Bankacılık ve Finans", ["bank", "banking", "credit", "loan", "jpmorgan", "goldman", "garanti", "akbank", "is bank", "yapı kredi"]),
        ("Teknoloji", ["ai", "artificial intelligence", "chip", "semiconductor", "software", "nvidia", "apple", "microsoft", "meta", "google"]),
        ("Otomotiv", ["tesla", "ford", "gm", "toyota", "vehicle", "ev", "electric vehicle", "otomotiv"]),
        ("Havacılık ve Ulaşım", ["airline", "aviation", "boeing", "airbus", "thy", "turkish airlines", "pegasus", "transport"]),
        ("Sağlık", ["healthcare", "pharma", "biotech", "drug", "medicine", "pfizer", "moderna"]),
        ("Perakende", ["retail", "consumer", "walmart", "amazon", "target", "sales"]),
        ("Makro Ekonomi", ["fed", "central bank", "inflation", "interest rate", "gdp", "economy", "enflasyon", "faiz", "merkez bankası"]),
    ]

    for sector, keywords in sector_keywords:
        if any(_matches_keyword(text, keyword) for keyword in keywords):
            return sector

    return fallback


def infer_news_company(title="", summary="", source=""):
    """Haber basligi ve ozetinden takip edilen sirket/sembol etiketini bulur."""
    text = f"{title} {summary} {source}".lower()
    company_keywords = [
        ("Apple", ["apple", "aapl"]),
        ("Microsoft", ["microsoft", "msft"]),
        ("Nvidia", ["nvidia", "nvda"]),
        ("Tesla", ["tesla", "tsla"]),
        ("Amazon", ["amazon", "amzn"]),
        ("Meta", ["meta", "facebook"]),
        ("Google", ["google", "alphabet", "googl", "goog"]),
        ("Netflix", ["netflix", "nflx"]),
        ("JPMorgan", ["jpmorgan", "jp morgan", "jpm"]),
        ("Goldman Sachs", ["goldman", "gs"]),
        ("Coinbase", ["coinbase", "coin"]),
        ("Turkish Airlines", ["turkish airlines", "thy", "thyao"]),
        ("Garanti BBVA", ["garanti", "garan"]),
        ("Akbank", ["akbank", "akbnk"]),
        ("İş Bankası", ["iş bankası", "is bank", "isctr"]),
        ("Yapı Kredi", ["yapı kredi", "yapi kredi", "ykbnk"]),
        ("Pegasus", ["pegasus", "pags"]),
    ]

    for company, keywords in company_keywords:
        if any(_matches_keyword(text, keyword) for keyword in keywords):
            return company

    return "Genel"


# Eski kullanım noktaları için ad korunur; yeni haber filtresinde sektör olarak kullanıyoruz.
def infer_news_category(title="", summary="", source="", fallback="Genel Piyasa"):
    return infer_news_sector(title, summary, source, fallback)


def get_stock_data(ticker_symbol):
    """Geçmiş 5 yıllık fiyat verisini ve teknik göstergeleri çeker."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1825)

        df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['return'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['volatility'] = df['return'].rolling(window=10).std()
        df['volume_change'] = df['Volume'].pct_change()
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Daily_Return'] = df['Close'].pct_change()

        return df[[
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'Daily_Return',
            'return', 'log_return', 'MA5', 'MA10', 'MA20',
            'volatility', 'MACD', 'MACD_signal', 'volume_change',
        ]]
        
    except Exception as e:
        print(f"Borsa verisi çekilemedi: {e}")
        return None


def _quote_number(value, default=None):
    try:
        if hasattr(value, "item"):
            value = value.item()
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_realtime_quote(ticker_symbol):
    """Yahoo Finance fast_info ile gun icine daha yakin quote verisini getirir."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        fast_info = ticker.fast_info or {}

        last_price = _quote_number(fast_info.get("last_price"))
        previous_close = _quote_number(fast_info.get("previous_close"))
        open_price = _quote_number(fast_info.get("open"))
        day_high = _quote_number(fast_info.get("day_high"))
        day_low = _quote_number(fast_info.get("day_low"))
        volume = _quote_number(fast_info.get("last_volume") or fast_info.get("volume"))

        if last_price is None:
            history = ticker.history(period="1d", interval="1m", prepost=True)
            if history is not None and not history.empty:
                last_price = _quote_number(history["Close"].dropna().iloc[-1])
                open_price = open_price if open_price is not None else _quote_number(history["Open"].dropna().iloc[0])
                day_high = day_high if day_high is not None else _quote_number(history["High"].max())
                day_low = day_low if day_low is not None else _quote_number(history["Low"].min())
                volume = volume if volume is not None else _quote_number(history["Volume"].sum())

        if last_price is None:
            return {}

        return {
            "last_price": last_price,
            "previous_close": previous_close,
            "open_price": open_price,
            "day_high": day_high,
            "day_low": day_low,
            "volume": volume,
            "source": "Yahoo Finance quote",
        }
    except Exception as e:
        print(f"Anlık quote verisi çekilemedi ({ticker_symbol}): {e}")
        return {}

def _normalize_news_item(title, published_at="", source="Yahoo Finance", url="", summary="", category="", image_url=""):
    sector = category or infer_news_sector(title, summary, source)
    company = infer_news_company(title, summary, source)
    return {
        "title": title or "Başlık yok",
        "date": str(published_at or ""),
        "source": source or "Piyasa Haberleri",
        "url": url or "",
        "summary": summary or "",
        "category": sector,
        "sector": sector,
        "company": company,
        "image_url": image_url or "",
    }


def _get_news_from_yfinance(ticker_name, limit=12, return_diagnostics=False):
    diagnostics = {
        "source": "Yahoo Finance",
        "requested": limit,
        "count": 0,
        "error": "",
    }
    try:
        news_items = []
        for item in yf.Ticker(ticker_name).news[:limit]:
            content = item.get("content", item)
            title = content.get("title") or item.get("title")
            published_at = content.get("pubDate") or item.get("providerPublishTime") or ""
            provider = content.get("provider", {})
            source = provider.get("displayName") if isinstance(provider, dict) else item.get("publisher", "Yahoo Finance")
            canonical_url = content.get("canonicalUrl", {})
            url = canonical_url.get("url") if isinstance(canonical_url, dict) else item.get("link", "")
            summary = content.get("summary", "")
            thumbnail = content.get("thumbnail") or item.get("thumbnail") or {}
            image_url = ""
            if isinstance(thumbnail, dict):
                resolutions = thumbnail.get("resolutions") or []
                if resolutions:
                    image_url = resolutions[-1].get("url", "")
                else:
                    image_url = thumbnail.get("url", "")

            if title:
                news_items.append(_normalize_news_item(title, published_at, source, url, summary, image_url=image_url))

        diagnostics["count"] = len(news_items)
        if return_diagnostics:
            return news_items, diagnostics
        return news_items
    except Exception as e:
        print(f"Yahoo Finance haberleri çekilemedi: {e}")
        diagnostics["error"] = str(e)
        if return_diagnostics:
            return [], diagnostics
        return []


def _get_news_from_newsapi(query, category="", limit=15, return_diagnostics=False):
    diagnostics = {
        "source": "NewsAPI",
        "enabled": bool(NEWS_API_KEY),
        "query": query,
        "requested": limit,
        "count": 0,
        "error": "",
    }
    try:
        if not NEWS_API_KEY:
            diagnostics["error"] = "NEWS_API_KEY tanımlı değil."
            if return_diagnostics:
                return [], diagnostics
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": NEWS_API_KEY,
        }
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        news = [
            _normalize_news_item(
                a.get("title"),
                a.get("publishedAt"),
                a.get("source", {}).get("name", "NewsAPI"),
                a.get("url", ""),
                a.get("description", ""),
                category,
                a.get("urlToImage", ""),
            )
            for a in articles[:limit]
            if a.get("title")
        ]
        diagnostics["count"] = len(news)
        if return_diagnostics:
            return news, diagnostics
        return news
    except Exception as e:
        print(f"NewsAPI haberleri çekilemedi: {e}")
        diagnostics["error"] = str(e)
        if return_diagnostics:
            return [], diagnostics
        return []


def get_stock_news(ticker_name, return_diagnostics=False):
    """Hisse hakkında son haberleri çeker. NewsAPI yoksa Yahoo Finance'a düşer."""
    newsapi_news, newsapi_diagnostics = _get_news_from_newsapi(
        f"{ticker_name} stock OR earnings OR shares",
        limit=15,
        return_diagnostics=True,
    )
    diagnostics = {
        "ticker": ticker_name,
        "selected_source": "",
        "newsapi": newsapi_diagnostics,
        "yahoo_finance": {
            "source": "Yahoo Finance",
            "requested": 12,
            "count": 0,
            "error": "",
        },
    }

    if newsapi_news:
        diagnostics["selected_source"] = "NewsAPI"
        if return_diagnostics:
            return newsapi_news, diagnostics
        return newsapi_news

    yahoo_news, yahoo_diagnostics = _get_news_from_yfinance(
        ticker_name,
        limit=12,
        return_diagnostics=True,
    )
    diagnostics["selected_source"] = "Yahoo Finance"
    diagnostics["yahoo_finance"] = yahoo_diagnostics
    if return_diagnostics:
        return yahoo_news, diagnostics

    return yahoo_news


def get_market_news():
    """Genel piyasa haberlerini haber sekmesi için çeker."""
    news = _get_news_from_newsapi("stock market OR finance OR economy OR Nasdaq OR BIST OR Bitcoin OR gold OR oil", limit=24)
    if news:
        return news

    fallback_symbols = ["^GSPC", "^IXIC", "BTC-USD", "THYAO.IS"]
    combined = []
    seen_titles = set()
    for symbol in fallback_symbols:
        for item in _get_news_from_yfinance(symbol, limit=5):
            if item["title"] in seen_titles:
                continue
            seen_titles.add(item["title"])
            item["sector"] = infer_news_sector(item.get("title", ""), item.get("summary", ""), item.get("source", ""))
            item["company"] = infer_news_company(item.get("title", ""), item.get("summary", ""), item.get("source", ""))
            item["category"] = item["sector"]
            combined.append(item)

    return combined[:20]


def get_stock_profile(ticker_name):
    """Hisse detay sayfasi icin sirket profil bilgilerini getirir."""
    try:
        info = yf.Ticker(ticker_name).info or {}
        return {
            "name": info.get("longName") or info.get("shortName") or ticker_name,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "market_cap": info.get("marketCap"),
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "website": info.get("website", ""),
            "summary": info.get("longBusinessSummary", ""),
        }
    except Exception as e:
        print(f"Profil bilgisi çekilemedi: {e}")
        return {
            "name": ticker_name,
            "sector": "N/A",
            "industry": "N/A",
            "country": "N/A",
            "market_cap": None,
            "beta": None,
            "fifty_two_week_high": None,
            "fifty_two_week_low": None,
            "website": "",
            "summary": "",
        }
