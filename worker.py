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
import csv
from datetime import datetime, timedelta
from openai import OpenAI
import requests
import numpy as np

# Optional Kraken libs used only if DRY_RUN=False and KRAKEN credentials provided
try:
    from krakenex import API as KrakenAPIConnector
    from pykrakenapi import KrakenAPI
except Exception:
    KrakenAPIConnector = None
    KrakenAPI = None

# ---------- CONFIG ----------
DRY_RUN = os.getenv("DRY_RUN", "True").lower() in ("true", "1", "yes")  # Control via GitHub Secret
USE_GPT = os.getenv("USE_GPT", "True").lower() in ("true", "1", "yes")  # Optional GPT analysis
TRADE_ALLOCATION_PCT = float(os.getenv("TRADE_ALLOCATION_PCT", "10"))  # % of account per trade
TOP_N = 10
HISTORY_FILE = "data/history.json"
POSITIONS_FILE = "data/positions.json"
TRADES_CSV = "data/trades.csv"
PERFORMANCE_CSV = "data/performance.csv"
OPENAI_MODEL = "gpt-4o"  # gpt-4o for best reasoning
INITIAL_CAPITAL = 10000.0  # Starting capital for paper trading
STOP_LOSS_PCT = 9.0  # Stop loss percentage (optimized)
TAKE_PROFIT_PCT = 22.5  # Take profit percentage (optimized)
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

def fetch_kraken_ohlc(pair, interval=60):
    """Fetch OHLC data from Kraken for technical analysis (interval in minutes)"""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        resp = requests.get(url, params={"pair": pair, "interval": interval}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("error") and len(data["error"]) > 0:
            return []
        
        result = data.get("result", {})
        # Get the actual pair key (Kraken returns modified names)
        pair_key = [k for k in result.keys() if k != "last"][0] if result else None
        if not pair_key:
            return []
        
        ohlc = result[pair_key]
        # Return last 200 candles for MA calculation
        return ohlc[-200:] if len(ohlc) > 200 else ohlc
    except Exception as e:
        print(f"OHLC error for {pair}: {e}")
        return []

def calculate_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_moving_averages(prices):
    """Calculate 20, 50 moving averages"""
    mas = {}
    if len(prices) >= 20:
        mas['ma20'] = np.mean(prices[-20:])
    if len(prices) >= 50:
        mas['ma50'] = np.mean(prices[-50:])
    return mas

def analyze_technicals(pair):
    """Perform technical analysis on a trading pair"""
    ohlc = fetch_kraken_ohlc(pair, interval=60)  # 1-hour candles
    if not ohlc or len(ohlc) < 20:
        return {}
    
    # Extract closing prices
    closes = np.array([float(candle[4]) for candle in ohlc])
    volumes = np.array([float(candle[6]) for candle in ohlc])
    
    # Calculate indicators
    rsi = calculate_rsi(closes)
    mas = calculate_moving_averages(closes)
    
    current_price = closes[-1]
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes.mean()
    recent_volume = volumes[-1]
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
    
    # Momentum: price change over last 7 periods
    momentum = ((closes[-1] - closes[-7]) / closes[-7] * 100) if len(closes) >= 7 else 0
    
    # Trend: price vs MA50
    trend = "bullish" if mas.get('ma50') and current_price > mas['ma50'] else "bearish"
    
    return {
        "rsi": round(rsi, 2) if rsi else None,
        "ma20": round(mas.get('ma20', 0), 2),
        "ma50": round(mas.get('ma50', 0), 2),
        "volume_ratio": round(volume_ratio, 2),
        "momentum_7h": round(momentum, 2),
        "trend": trend,
        "current_price": round(current_price, 2)
    }

def ensure_history_file():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f, indent=2)

def read_positions():
    """Read current open positions"""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(POSITIONS_FILE):
        return {"cash": INITIAL_CAPITAL, "positions": []}
    with open(POSITIONS_FILE, "r") as f:
        return json.load(f)

def write_positions(data):
    """Write positions to file"""
    os.makedirs("data", exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def log_trade_to_csv(trade_data):
    """Append trade to CSV file"""
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(TRADES_CSV)
    
    with open(TRADES_CSV, "a", newline="") as f:
        fieldnames = ["timestamp", "symbol", "action", "entry_price", "exit_price",  
                     "quantity", "pnl", "pnl_pct", "hold_hours", "score", "rsi", 
                     "momentum", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(trade_data)

def calculate_portfolio_value(positions_data, ticker_data):
    """Calculate total portfolio value"""
    total = positions_data.get("cash", INITIAL_CAPITAL)
    
    for pos in positions_data.get("positions", []):
        pair = pos["symbol"].replace("/", "").replace("-", "")
        current_price = ticker_data.get(pair, {}).get("last_price", pos["entry_price"])
        position_value = pos["quantity"] * current_price
        total += position_value
    
    return total

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
    
    # Analyze technicals and filter by quality
    technical_analysis = {}
    scored_pairs = []
    
    print("Analyzing technicals...")
    for pair in kraken_pairs:
        if pair not in ticker_data:
            continue
            
        ta = analyze_technicals(pair)
        if not ta:
            continue
            
        technical_analysis[pair] = ta
        
        # Score the pair (0-100)
        score = 0
        
        # RSI scoring (45-65 is ideal, avoid extremes)
        if ta.get('rsi'):
            if 45 <= ta['rsi'] <= 65:
                score += 30
            elif 40 <= ta['rsi'] <= 70:
                score += 15
            elif ta['rsi'] > 75 or ta['rsi'] < 30:
                score -= 10  # Penalize extremes
        
        # Trend scoring (increased weight)
        if ta.get('trend') == 'bullish':
            score += 30
            # Bonus for strong trend (price above both MAs)
            if ta.get('ma20') and ta.get('ma50'):
                if ta['current_price'] > ta['ma20'] > ta['ma50']:
                    score += 10
        
        # Volume scoring (require above-average volume)
        if ta.get('volume_ratio', 0) > 2.0:
            score += 20
        elif ta.get('volume_ratio', 0) > 1.5:
            score += 10
        elif ta.get('volume_ratio', 0) < 0.8:
            score -= 5  # Penalize low volume
        
        # Momentum scoring (require positive momentum)
        momentum = ta.get('momentum_7h', 0)
        if momentum > 8:
            score += 25
        elif momentum > 3:
            score += 15
        elif momentum < -5:
            score -= 10  # Penalize negative momentum
        
        # 24h performance
        change_24h = ticker_data[pair]['24h_change_pct']
        if change_24h > 5:
            score += 10
        elif change_24h > 0:
            score += 5
        
        scored_pairs.append((pair, score, ta, ticker_data[pair]))
    
    # Sort by score and take top performers
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Option 1: Pure technical strategy (no GPT)
    if not USE_GPT or not client.api_key:
        print("Using pure technical scoring (GPT disabled)")
        top_picks = []
        for pair, score, ta, ticker in scored_pairs[:TOP_N]:
            # Convert to proper format
            formatted = pair.replace("USD", "/USD").replace("XBT/", "BTC/")
            if "/" not in formatted:
                formatted = formatted[:-3] + "/" + formatted[-3:]
            top_picks.append(formatted)
        return top_picks
    
    # Option 2: GPT-enhanced analysis
    # Build context with real market data + technical analysis
    market_context = "REAL-TIME MARKET DATA + TECHNICAL ANALYSIS:\n\n"
    market_context += "CoinGecko Trending (Top 5):\n"
    for i, coin in enumerate(trending[:5], 1):
        market_context += f"{i}. {coin['symbol']} ({coin['name']}) - Rank: {coin['market_cap_rank']}\n"
    
    market_context += "\nTop Kraken Pairs by Technical Score:\n"
    for pair, score, ta, ticker in scored_pairs[:12]:
        market_context += f"{pair}: Score={score}/100 | RSI={ta.get('rsi','N/A')} | "
        market_context += f"Trend={ta.get('trend','N/A')} | 24h={ticker['24h_change_pct']:.1f}% | "
        market_context += f"Vol Ratio={ta.get('volume_ratio','N/A')}x | Momentum={ta.get('momentum_7h','N/A')}%\n"
    
    prompt = (
        f"{market_context}\n\n"
        f"Based on the REAL technical analysis above, return ONLY a JSON array of the top {TOP_N} "
        "spot trading pairs on Kraken for swing trading over the next 1-3 weeks. "
        "Prioritize pairs with: high technical scores, bullish trends, healthy RSI (30-70), strong momentum, and high volume. "
        "AVOID: extreme RSI (>80 overbought, <20 oversold), low scores, bearish trends. "
        "Use Kraken pair format (example: \"XBT/USD\", \"ETH/USD\"). "
        "Output ONLY a JSON array of strings."
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"You are an expert crypto trader analyzing REAL technical data. Base recommendations ONLY on the technical scores and indicators provided."},
                {"role":"user","content":prompt}
            ],
            temperature=0.3,
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
    print("=== worker run at", ts, "===\nDRY_RUN =", DRY_RUN, "| USE_GPT =", USE_GPT)

    # Load current positions
    portfolio = read_positions()
    
    # Get top symbols based on technical analysis
    top = ask_openai_for_top_symbols()
    print("Top list:", top)

    # Fetch real market data for all pairs
    all_pairs = [s.replace("/", "").replace("-", "") for s in top]
    ticker_data = fetch_kraken_ticker_data(all_pairs)
    
    # Check existing positions for exit signals
    closed_positions = []
    for pos in portfolio.get("positions", [])[:]:  # Copy list to modify during iteration
        pair = pos["symbol"].replace("/", "").replace("-", "")
        current_price = ticker_data.get(pair, {}).get("last_price")
        
        if not current_price:
            continue
        
        entry_price = pos["entry_price"]
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        hold_hours = (datetime.utcnow() - datetime.fromisoformat(pos["entry_time"])).total_seconds() / 3600
        
        # Exit signals
        should_exit = False
        exit_reason = ""
        
        if pnl_pct <= -STOP_LOSS_PCT:
            should_exit = True
            exit_reason = f"Stop loss hit ({pnl_pct:.2f}%)"
        elif pnl_pct >= TAKE_PROFIT_PCT:
            should_exit = True
            exit_reason = f"Take profit hit ({pnl_pct:.2f}%)"
        elif pos["symbol"] not in top:
            should_exit = True
            exit_reason = "No longer in top picks"
        
        if should_exit:
            # Close position
            position_value = pos["quantity"] * current_price
            pnl = position_value - (pos["quantity"] * entry_price)
            
            portfolio["cash"] += position_value
            portfolio["positions"].remove(pos)
            closed_positions.append(pos)
            
            # Log to CSV
            log_trade_to_csv({
                "timestamp": ts,
                "symbol": pos["symbol"],
                "action": "CLOSE",
                "entry_price": round(entry_price, 2),
                "exit_price": round(current_price, 2),
                "quantity": round(pos["quantity"], 6),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "hold_hours": round(hold_hours, 1),
                "score": pos.get("score", 0),
                "rsi": pos.get("rsi", 0),
                "momentum": pos.get("momentum", 0),
                "reason": exit_reason
            })
            
            print(f"CLOSED {pos['symbol']}: Entry=${entry_price:.2f}, Exit=${current_price:.2f}, P&L={pnl:.2f} ({pnl_pct:.2f}%), Reason: {exit_reason}")
    
    # Open new positions from top picks
    current_symbols = [p["symbol"] for p in portfolio.get("positions", [])]
    portfolio_value = calculate_portfolio_value(portfolio, ticker_data)
    
    for symbol in top:
        if symbol in current_symbols:
            continue  # Already have position
        
        if len(portfolio["positions"]) >= TOP_N:
            break  # Max positions reached
        
        pair = symbol.replace("/", "").replace("-", "")
        price_data = ticker_data.get(pair)
        
        if not price_data:
            continue
        
        # Get technical data
        ta = analyze_technicals(pair)
        if not ta:
            continue
        
        # Calculate position size
        position_size_usd = portfolio_value * (TRADE_ALLOCATION_PCT / 100.0)
        quantity = position_size_usd / price_data["last_price"]
        
        if portfolio["cash"] < position_size_usd:
            continue  # Not enough cash
        
        # Open position
        portfolio["cash"] -= position_size_usd
        portfolio["positions"].append({
            "symbol": symbol,
            "entry_price": price_data["last_price"],
            "entry_time": ts,
            "quantity": quantity,
            "score": ta.get("score", 0),
            "rsi": ta.get("rsi", 0),
            "momentum": ta.get("momentum_7h", 0)
        })
        
        # Log to CSV
        log_trade_to_csv({
            "timestamp": ts,
            "symbol": symbol,
            "action": "OPEN",
            "entry_price": round(price_data["last_price"], 2),
            "exit_price": 0,
            "quantity": round(quantity, 6),
            "pnl": 0,
            "pnl_pct": 0,
            "hold_hours": 0,
            "score": ta.get("score", 0),
            "rsi": ta.get("rsi", 0),
            "momentum": ta.get("momentum_7h", 0),
            "reason": "New entry"
        })
        
        print(f"OPENED {symbol}: Entry=${price_data['last_price']:.2f}, Size=${position_size_usd:.2f}, Qty={quantity:.6f}")
    
    # Save updated portfolio
    write_positions(portfolio)
    
    # Calculate and save performance metrics
    final_portfolio_value = calculate_portfolio_value(portfolio, ticker_data)
    total_return = ((final_portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    performance = {
        "timestamp": ts,
        "portfolio_value": round(final_portfolio_value, 2),
        "cash": round(portfolio["cash"], 2),
        "total_return_pct": round(total_return, 2),
        "open_positions": len(portfolio.get("positions", [])),
        "closed_today": len(closed_positions)
    }
    
    # Save to performance CSV
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(PERFORMANCE_CSV)
    with open(PERFORMANCE_CSV, "a", newline="") as f:
        fieldnames = ["timestamp", "portfolio_value", "cash", "total_return_pct", 
                     "open_positions", "closed_today"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(performance)
    
    print(f"\n=== PORTFOLIO SUMMARY ===")
    print(f"Total Value: ${final_portfolio_value:.2f}")
    print(f"Cash: ${portfolio['cash']:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Open Positions: {len(portfolio.get('positions', []))}")
    print(f"Closed Today: {len(closed_positions)}")
    
    # Keep history for reference
    hist = read_history()
    hist.append({
        "time": ts,
        "top": top,
        "portfolio_value": final_portfolio_value,
        "total_return_pct": total_return
    })
    if len(hist) > 5000:
        hist = hist[-5000:]
    write_history(hist)
    
    print("Worker finished.")

if __name__ == "__main__":
    main()
