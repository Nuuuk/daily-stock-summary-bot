import datetime
from typing import List, Dict, Any
import yfinance as yf
import finnhub
import time

def assemble_full_market_payload(positions: List[Dict[str, Any]], finnhub_api_key: str, days_back: int = 1) -> Dict[str, Any]:
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
    today = datetime.date.today()
    
    macro_data = fetch_macro_indicators()
    unique_tickers = list(set(pos["ticker"].strip().upper() for pos in positions if "ticker" in pos))
    
    market_cache = {}
    for ticker in unique_tickers:
        market_cache[ticker] = {
            "tech": fetch_ticker_technical_data(ticker),
            "news": fetch_ticker_news(ticker, finnhub_client, days_back=days_back)
        }
        # 每次拉取间隔 0.3 秒，平滑请求节奏
        time.sleep(0.3)
        
    ...

MACRO_TICKERS = {
    "SP500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Russell2000": "^RUT",
    "US10Y_Yield": "^TNX",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB"
}

# 精选互补观察池（覆盖 AI 基础设施散热、清洁核电、网安、半导体代工与工业龙头）
CURATED_COMPLEMENTARY_POOL = [
    {"ticker": "VRT", "sector": "AI Data Center Cooling / Power", "name": "Vertiv Holdings"},
    {"ticker": "CEG", "sector": "Nuclear / Base-load Clean Energy", "name": "Constellation Energy"},
    {"ticker": "GEV", "sector": "Power Grid / Energy Infrastructure", "name": "GE Vernova"},
    {"ticker": "CRWD", "sector": "Enterprise Cybersecurity", "name": "CrowdStrike"},
    {"ticker": "TSM", "sector": "Leading Semiconductor Foundry", "name": "Taiwan Semiconductor"},
    {"ticker": "PLTR", "sector": "Enterprise AI & Defense Analytics", "name": "Palantir"},
    {"ticker": "ANET", "sector": "Cloud & AI Networking", "name": "Arista Networks"}
]

def fetch_complementary_candidates_data() -> List[Dict[str, Any]]:
    """拉取精选互补候选标的的最新真实技术面指标"""
    results = []
    for item in CURATED_COMPLEMENTARY_POOL:
        ticker = item["ticker"]
        tech = fetch_ticker_technical_data(ticker)
        if "error" not in tech:
            results.append({
                "ticker": ticker,
                "name": item["name"],
                "sector": item["sector"],
                "current_price": tech.get("current_price"),
                "change_pct": tech.get("change_pct"),
                "ma_50": tech.get("ma_50"),
                "ma_200": tech.get("ma_200"),
                "52w_low": tech.get("52w_low"),
                "52w_high": tech.get("52w_high"),
                "relative_volume": tech.get("relative_volume")
            })
    return results

def fetch_macro_indicators() -> Dict[str, Any]:
    macro_data = {}
    for name, symbol in MACRO_TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100
                macro_data[name] = {
                    "symbol": symbol,
                    "current": round(float(current_price), 2),
                    "change_pct": round(float(change_pct), 2)
                }
            else:
                macro_data[name] = {"symbol": symbol, "error": "No price history available"}
        except Exception as e:
            macro_data[name] = {"symbol": symbol, "error": str(e)}
    return macro_data


def fetch_ticker_technical_data(ticker_symbol: str) -> Dict[str, Any]:
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1y")
        if hist.empty:
            return {"error": "Failed to fetch market data"}
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100
        
        ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
        ma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else None
        
        latest_vol = hist['Volume'].iloc[-1]
        avg_vol_20 = hist['Volume'].rolling(window=20).mean().iloc[-1] if len(hist) >= 20 else latest_vol
        relative_volume = round(float(latest_vol / avg_vol_20), 2) if avg_vol_20 > 0 else 1.0
        
        low_52w = hist['Low'].min()
        high_52w = hist['High'].max()
        
        return {
            "current_price": round(float(current_price), 2),
            "change_pct": round(float(change_pct), 2),
            "ma_50": round(float(ma_50), 2) if ma_50 else None,
            "ma_200": round(float(ma_200), 2) if ma_200 else None,
            "relative_volume": relative_volume,
            "52w_low": round(float(low_52w), 2),
            "52w_high": round(float(high_52w), 2)
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_ticker_news(ticker_symbol: str, finnhub_client: finnhub.Client, days_back: int = 1) -> List[Dict[str, str]]:
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days_back)
    
    try:
        news_items = finnhub_client.company_news(
            ticker_symbol, 
            _from=start_date.strftime('%Y-%m-%d'), 
            to=today.strftime('%Y-%m-%d')
        )
        curated_news = []
        for item in news_items[:4]:
            curated_news.append({
                "headline": item.get("headline", ""),
                "summary": item.get("summary", "")[:200] + "..." if item.get("summary") else "",
                "source": item.get("source", ""),
                "datetime": datetime.datetime.fromtimestamp(item.get("datetime", 0)).strftime('%Y-%m-%d %H:%M')
            })
        return curated_news
    except Exception as e:
        return [{"error": f"Failed to fetch Finnhub news: {str(e)}"}]


def assemble_full_market_payload(positions: List[Dict[str, Any]], finnhub_api_key: str, days_back: int = 1) -> Dict[str, Any]:
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
    today = datetime.date.today()
    
    macro_data = fetch_macro_indicators()
    unique_tickers = list(set(pos["ticker"].strip().upper() for pos in positions if "ticker" in pos))
    
    market_cache = {}
    for ticker in unique_tickers:
        market_cache[ticker] = {
            "tech": fetch_ticker_technical_data(ticker),
            "news": fetch_ticker_news(ticker, finnhub_client, days_back=days_back)
        }
        
    enriched_tax_lots = []
    for pos in positions:
        ticker = pos.get("ticker", "").strip().upper()
        broker = pos.get("broker", "Default")
        buy_date_str = pos.get("buy_date")
        quantity = float(pos.get("quantity", 0))
        cost_basis = float(pos.get("average_buy_price", 0.0))
        strategy = pos.get("strategy", "Long-term")
        
        holding_days = 0
        if buy_date_str:
            try:
                buy_date = datetime.datetime.strptime(buy_date_str, "%Y-%m-%d").date()
                holding_days = (today - buy_date).days
            except ValueError:
                pass
        
        cached_data = market_cache.get(ticker, {})
        tech_data = cached_data.get("tech", {})
        current_price = tech_data.get("current_price", cost_basis)
        
        unrealized_pnl = round((current_price - cost_basis) * quantity, 2)
        unrealized_pnl_pct = round(((current_price - cost_basis) / cost_basis) * 100, 2) if cost_basis > 0 else 0.0
        
        if holding_days >= 365:
            tax_status_label = "长期税率 (已超1年，可优先操作)"
        elif holding_days >= 300:
            tax_status_label = f"冲刺长期税率中 (仅剩 {365 - holding_days} 天，严禁止盈)"
        else:
            tax_status_label = f"短期持仓 (已持有 {holding_days} 天)"
            
        enriched_tax_lots.append({
            "ticker": ticker,
            "broker": broker,
            "buy_date": buy_date_str,
            "quantity": quantity,
            "cost_basis": cost_basis,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "holding_days": holding_days,
            "tax_status_label": tax_status_label,
            "strategy": strategy,
            "technical_indicators": tech_data,
            "recent_news": cached_data.get("news", [])
        })
        
    return {
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S EST'),
        "macro_environment": macro_data,
        "positions_tax_lots": enriched_tax_lots,
        "complementary_candidates_market_data": fetch_complementary_candidates_data()  # 注入真实候选池行情
    }
