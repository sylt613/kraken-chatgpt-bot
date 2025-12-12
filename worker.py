"""
worker.py
Serverless trading worker designed to run once and exit.
- Uses OpenAI to pick top-10 Kraken symbols for swing trading
- Records simple history (timestamp, top10, equity placeholder) to data/history.json
- Optionally (DRY_RUN=False) will place simulated or real Kraken orders (simple market buys/sells)
- Intended to run in GitHub Actions on a schedule

IMPORTANT LIMITATION:
- OpenAI models don't have real-time web access or live market data
- Recommendations are based on training data (which has a cutoff date)
- For real web search, consider integrating: Perplexity API, Tavily, or SerpAPI
- This is a learning/experimental bot - NOT production-ready for live trading
"""

import os
import json
import time
from datetime import datetime
from openai import OpenAI
import requests

# Optional Kraken libs used only if DRY_RUN=False and KRAKEN credentials provided
try:
    from krakenex import API as KrakenAPIConnector
    from pykrakenapi import KrakenAPI
except Exception:
    KrakenAPIConnector = None
    KrakenAPI = None

# ---------- CONFIG ----------
DRY_RUN = os.getenv("DRY_RUN", "True").lower() in ("true", "1", "yes")  # Control via GitHub Secret
TRADE_ALLOCATION_PCT = float(os.getenv("TRADE_ALLOCATION_PCT", "10"))  # % of account per trade
TOP_N = 10
HISTORY_FILE = "data/history.json"
OPENAI_MODEL = "gpt-4o"  # gpt-4o for best reasoning (Note: GPT models don't have real-time web access)
# --------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

KRAKEN_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_SECRET = os.getenv("KRAKEN_API_SECRET")

def fetch_coingecko_trending():
    """Fetch trending coins from CoinGecko (free, no API key needed)"""
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        trending = []
        for item in data.get("coins", [])[:15]:
            coin = item.get("item", {})
            trending.append({
                "symbol": coin.get("symbol", "").upper(),
                "name": coin.get("name", ""),
                "market_cap_rank": coin.get("market_cap_rank", 0),
                "price_btc": coin.get("price_btc", 0)
            })
        return trending
    except Exception as e:
        print(f"CoinGecko trending error: {e}")
        return []

def fetch_kraken_ticker_data(pairs):
    """Fetch real-time price data from Kraken for given pairs"""
    try:
        # Kraken public API - no auth needed
        url = "https://api.kraken.com/0/public/Ticker"
        pairs_str = ",".join(pairs)
        resp = requests.get(url, params={"pair": pairs_str}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("error") and len(data["error"]) > 0:
            print(f"Kraken API error: {data['error']}")
            return {}
        
        result = {}
        for pair, info in data.get("result", {}).items():
            # Extract key metrics
            result[pair] = {
                "last_price": float(info["c"][0]),
                "24h_change_pct": float(info["p"][1]),
                "24h_volume": float(info["v"][1]),
                "24h_high": float(info["h"][1]),
                "24h_low": float(info["l"][1])
            }
        return result
    except Exception as e:
        print(f"Kraken ticker error: {e}")
        return {}

def ensure_history_file():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f, indent=2)

def read_history():
    ensure_history_file()
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def write_history(data):
    ensure_history_file()
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def ask_openai_for_top_symbols():
    # Fetch real market data
    trending = fetch_coingecko_trending()
    
    # Common Kraken pairs to check
    kraken_pairs = ["XBTUSD", "ETHUSD", "SOLUSD", "ADAUSD", "DOTUSD", 
                    "LINKUSD", "AVAXUSD", "MATICUSD", "UNIUSD", "ATOMUSD",
                    "LTCUSD", "DOGEUSD", "SHIBUSD", "APTUSD", "OPUSD"]
    ticker_data = fetch_kraken_ticker_data(kraken_pairs)
    
    # Build context with real market data
    market_context = "REAL-TIME MARKET DATA:\n\n"
    market_context += "CoinGecko Trending (Top 10):\n"
    for i, coin in enumerate(trending[:10], 1):
        market_context += f"{i}. {coin['symbol']} ({coin['name']}) - Rank: {coin['market_cap_rank']}\n"
    
    market_context += "\nKraken 24h Performance:\n"
    for pair, data in sorted(ticker_data.items(), key=lambda x: x[1]['24h_change_pct'], reverse=True)[:10]:
        market_context += f"{pair}: {data['24h_change_pct']:.2f}% | Vol: ${data['24h_volume']:,.0f}\n"
    
    prompt = (
        f"{market_context}\n\n"
        f"Based on the REAL market data above, return ONLY a JSON array of the top {TOP_N} "
        "spot trading pairs on Kraken for swing trading over the next 1-3 weeks. "
        "Prioritize coins with: strong 24h performance, high volume, trending status, and bullish momentum. "
        "Use Kraken pair format (example: \"XBT/USD\", \"ETH/USD\"). "
        "Output ONLY a JSON array of strings."
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"You are an expert crypto trader analyzing REAL market data. Base recommendations on the actual data provided."},
                {"role":"user","content":prompt}
            ],
            temperature=0.4,
            max_tokens=300
        )
        text = resp.choices[0].message.content.strip()
        # parse JSON array or fallback: split by commas
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr][:TOP_N]
        except Exception:
            # allow comma-separated fallback
            parts = [p.strip().strip('"').strip("'") for p in text.replace("\n",",").split(",") if p.strip()]
            return parts[:TOP_N]
    except Exception as e:
        print("OpenAI error:", e)
    return []

def ask_openai_bullish(symbol, market_data=None):
    # Add real market data context if available
    context = ""
    if market_data:
        context = f"REAL DATA for {symbol}: "
        context += f"24h Change: {market_data.get('24h_change_pct', 'N/A')}%, "
        context += f"Volume: ${market_data.get('24h_volume', 'N/A'):,.0f}, "
        context += f"Price: ${market_data.get('last_price', 'N/A')}\n\n"
    
    prompt = (
        f"{context}"
        f"Is {symbol} bullish for a 1-3 week swing trade horizon? "
        "Consider: the real data above, technical patterns, volume trends, and market momentum. "
        "Be specific about catalysts or risks. "
        "Return a JSON object EXACTLY like: {{\"bullish\": true/false, \"reason\": \"one-sentence reason\", \"confidence\": 0-100}}"
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"You are a crypto market analyst analyzing REAL market data. Base assessments on actual metrics provided."},
                {"role":"user","content":prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        text = resp.choices[0].message.content.strip()
        try:
            return json.loads(text)
        except Exception:
            # crude fallback
            lower = text.lower()
            bullish = "yes" in lower or "true" in lower or "bullish" in lower
            return {"bullish": bullish, "reason": text}
    except Exception as e:
        print("OpenAI bullish check error:", e)
        return {"bullish": False, "reason": "openai_error"}

def get_equity_placeholder():
    # Robust equity calculation would query Kraken balances. For safety & simplicity we return None if not available.
    if DRY_RUN or KrakenAPIConnector is None:
        return None
    try:
        conn = KrakenAPIConnector(KRAKEN_KEY, KRAKEN_SECRET)
        k = KrakenAPI(conn)
        bal = k.get_account_balance()
        # sum all balances in account — this is naive because conversion to USD isn't handled
        total = 0.0
        for v in bal["balance"].values():
            try:
                total += float(v)
            except:
                pass
        return total
    except Exception as e:
        print("Error fetching equity:", e)
        return None

# Optional simplified order functions (very naive; real trading requires careful step-size handling)
def place_market_buy(pair, equity):
    notional = equity * (TRADE_ALLOCATION_PCT / 100.0)
    if DRY_RUN:
        print(f"[DRY_RUN] would BUY {pair} for ${notional:.2f} ({TRADE_ALLOCATION_PCT}% of ${equity:.2f})")
        return {"status":"simulated","pair":pair,"notional":notional,"allocation_pct":TRADE_ALLOCATION_PCT}
    if KrakenAPIConnector is None:
        raise RuntimeError("Kraken libs not installed")
    # NOTE: implement carefully if enabling live trading
    api = KrakenAPIConnector(KRAKEN_KEY, KRAKEN_SECRET)
    k = KrakenAPI(api)
    # This is placeholder — user must implement precise sizing & pair handling
    return {"status":"not-implemented","notional":notional,"allocation_pct":TRADE_ALLOCATION_PCT}

def place_market_sell(pair, volume):
    if DRY_RUN:
        print(f"[DRY_RUN] would SELL {pair} volume {volume}")
        return {"status":"simulated","pair":pair,"volume":volume}
    return {"status":"not-implemented"}

def main():
    ensure_history_file()
    ts = datetime.utcnow().isoformat()
    print("=== worker run at", ts, "===\nDRY_RUN =", DRY_RUN)

    top = ask_openai_for_top_symbols()
    print("Top list from OpenAI:", top)

    # Fetch real market data for selected symbols
    ticker_data = fetch_kraken_ticker_data(top)
    
    # For each current position (we don't maintain a DB here in the serverless run),
    # we conservativey just record the top list and a bullish check for each symbol.
    results = []
    for s in top:
        # Get market data for this symbol if available
        market_info = ticker_data.get(s.replace("/", "").replace("-", ""), {})
        bull = ask_openai_bullish(s, market_info)
        results.append({
            "symbol": s, 
            "bullish": bull.get("bullish", False), 
            "reason": bull.get("reason",""),
            "confidence": bull.get("confidence", 0),
            "24h_change": market_info.get("24h_change_pct"),
            "volume_24h": market_info.get("24h_volume")
        })

    equity = get_equity_placeholder()

    # Append to history
    hist = read_history()
    hist.append({
        "time": ts,
        "top": top,
        "top_diagnostics": results,
        "equity": equity
    })
    # Keep history size reasonable (e.g., last 10k entries)
    if len(hist) > 20000:
        hist = hist[-20000:]
    write_history(hist)

    print("History length:", len(hist))
    print("Worker finished.")

if __name__ == "__main__":
    main()
