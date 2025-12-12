"""
worker.py
Serverless trading worker designed to run once and exit.
- Uses OpenAI to pick top-10 Kraken symbols for swing trading
- Records simple history (timestamp, top10, equity placeholder) to data/history.json
- Optionally (DRY_RUN=False) will place simulated or real Kraken orders (simple market buys/sells)
- Intended to run in GitHub Actions on a schedule
"""

import os
import json
import time
from datetime import datetime
import openai

# Optional Kraken libs used only if DRY_RUN=False and KRAKEN credentials provided
try:
    from krakenex import API as KrakenAPIConnector
    from pykrakenapi import KrakenAPI
except Exception:
    KrakenAPIConnector = None
    KrakenAPI = None

# ---------- CONFIG ----------
DRY_RUN = True   # KEEP True initially. Set to False only after careful local testing.
TOP_N = 10
HISTORY_FILE = "data/history.json"
OPENAI_MODEL = "gpt-4o"  # change if you prefer another model
# --------------------------------

openai.api_key = os.getenv("OPENAI_API_KEY")

KRAKEN_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_SECRET = os.getenv("KRAKEN_API_SECRET")

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
    prompt = (
        f"You are an expert crypto swing trader. Return ONLY a JSON array of the top {TOP_N} "
        "spot trading pairs on Kraken for swing trading over the next 1-3 weeks. "
        "Use Kraken pair format (example: \"XBT/USD\", \"ETH/USD\"). Output ONLY a JSON array."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.6,
            max_tokens=300
        )
        text = resp["choices"][0]["message"]["content"].strip()
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

def ask_openai_bullish(symbol):
    prompt = (
        f"Is {symbol} bullish for a 1-3 week swing trade horizon? Consider sentiment, recent news, and technicals. "
        "Return a JSON object EXACTLY like: {{\"bullish\": true/false, \"reason\": \"one-sentence reason\"}}"
    )
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=150
        )
        text = resp["choices"][0]["message"]["content"].strip()
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
def place_market_buy(pair, notional):
    if DRY_RUN:
        print(f"[DRY_RUN] would BUY {pair} for ${notional}")
        return {"status":"simulated","pair":pair,"notional":notional}
    if KrakenAPIConnector is None:
        raise RuntimeError("Kraken libs not installed")
    # NOTE: implement carefully if enabling live trading
    api = KrakenAPIConnector(KRAKEN_KEY, KRAKEN_SECRET)
    k = KrakenAPI(api)
    # This is placeholder — user must implement precise sizing & pair handling
    return {"status":"not-implemented"}

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

    # For each current position (we don't maintain a DB here in the serverless run),
    # we conservativey just record the top list and a bullish check for each symbol.
    results = []
    for s in top:
        bull = ask_openai_bullish(s)
        results.append({"symbol": s, "bullish": bull.get("bullish", False), "reason": bull.get("reason","")})

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
