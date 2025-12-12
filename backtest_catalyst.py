"""
backtest_catalyst.py
NEWS/CATALYST DETECTION STRATEGY BACKTEST

This strategy aims to catch EXPLOSIVE moves by detecting catalysts:

LIVE TRADING SIGNALS:
1. GPT-analyzed news sentiment (bullish/bearish/neutral)
2. CoinGecko trending momentum 
3. Unusual volume detection
4. Cross-asset momentum correlation

BACKTEST METHODOLOGY:
Since we can't backtest actual news, we detect "catalyst fingerprints":
- Sudden volume spikes (3x+ average) = likely news event
- Price breakout from consolidation = likely catalyst
- Multi-day momentum acceleration = trend confirmation

TARGET: High-beta altcoins that respond to news:
SOL, LINK, MATIC, OP, APT, ATOM, AVAX, DOT, XRP, ADA
"""

import os
import sys
import requests
import numpy as np
from datetime import datetime, timedelta
import csv
import json
import time

# ============ CATALYST STRATEGY PARAMETERS ============
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 12.0      # Smaller positions for more diversification
MAX_POSITIONS = 7             # More trades, more opportunities
STOP_LOSS_PCT = 10.0          # Wider stops for volatile plays
TAKE_PROFIT_PCT = 35.0        # Let big winners run
TRAILING_STOP_PCT = 18.0      # Wide trailing stop

# Catalyst Detection Parameters
VOLUME_SPIKE_THRESHOLD = 3.0  # Higher volume threshold for conviction
BREAKOUT_THRESHOLD = 5.0      # Price must break out by 5%+
MOMENTUM_DAYS = 3             # Look at 3-day momentum
MIN_CATALYST_SCORE = 75       # Lower threshold for more trades
MIN_GPT_SENTIMENT = 25        # Lower sentiment threshold for more trades

# Focus on high-beta altcoins that respond to news
CATALYST_PAIRS = [
    "SOLUSD", "LINKUSD", "MATICUSD", "OPUSD", "APTUSD",
    "ATOMUSD", "AVAXUSD", "DOTUSD", "XRPUSD", "ADAUSD",
    "NEARUSD", "FILUSD", "ARBUSD", "RNDRUSD", "INJUSD",
    "SUIUSD", "SEIUSD", "TIAUSD", "WIFUSD", "BONKUSD"
]

RESULTS_DIR = "backtest_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def fetch_daily_ohlcv(pair, days=180):
    """Fetch daily OHLCV data from Kraken"""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        since = int((datetime.now() - timedelta(days=days+10)).timestamp())
        resp = requests.get(url, params={"pair": pair, "interval": 1440, "since": since}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("error") and len(data["error"]) > 0:
            print(f"  API error for {pair}: {data['error']}")
            return None
        
        result = data.get("result", {})
        pair_key = [k for k in result.keys() if k != "last"][0] if result else None
        if not pair_key:
            return None
        
        ohlcv = result[pair_key]
        
        # Parse OHLCV data
        parsed = []
        for candle in ohlcv:
            parsed.append({
                "timestamp": int(candle[0]),
                "date": datetime.fromtimestamp(int(candle[0])).strftime("%Y-%m-%d"),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[6])
            })
        
        return parsed[-days:] if len(parsed) > days else parsed
        
    except Exception as e:
        print(f"  Error fetching {pair}: {e}")
        return None


def calculate_catalyst_score(data, idx):
    """
    Calculate a CATALYST SCORE that detects news-like events.
    
    Components:
    1. Volume Spike (40 pts): Unusual volume = news event
    2. Price Breakout (30 pts): Breaking out of range = catalyst
    3. Momentum Acceleration (20 pts): Momentum picking up
    4. Range Expansion (10 pts): Volatility increasing
    
    Returns: score 0-100
    """
    if idx < 20:
        return 0, {}
    
    scores = {}
    
    # Current day data
    today = data[idx]
    yesterday = data[idx-1]
    
    # ===== 1. VOLUME SPIKE DETECTION (40 pts) =====
    # Compare today's volume to 20-day average
    volumes = [data[i]["volume"] for i in range(idx-20, idx)]
    avg_volume = np.mean(volumes) if volumes else 1
    volume_ratio = today["volume"] / avg_volume if avg_volume > 0 else 1
    
    if volume_ratio >= 4.0:
        scores["volume_spike"] = 40
    elif volume_ratio >= 3.0:
        scores["volume_spike"] = 32
    elif volume_ratio >= 2.5:
        scores["volume_spike"] = 25
    elif volume_ratio >= 2.0:
        scores["volume_spike"] = 18
    elif volume_ratio >= 1.5:
        scores["volume_spike"] = 10
    else:
        scores["volume_spike"] = 0
    
    # ===== 2. PRICE BREAKOUT DETECTION (30 pts) =====
    # Check if price breaking out of 10-day range
    highs = [data[i]["high"] for i in range(idx-10, idx)]
    lows = [data[i]["low"] for i in range(idx-10, idx)]
    range_high = max(highs)
    range_low = min(lows)
    
    # Calculate daily change
    daily_change_pct = ((today["close"] - yesterday["close"]) / yesterday["close"]) * 100
    
    # Breaking above range with momentum
    if today["close"] > range_high * 1.02 and daily_change_pct > 5:
        scores["breakout"] = 30
    elif today["close"] > range_high and daily_change_pct > 3:
        scores["breakout"] = 22
    elif daily_change_pct > 6:
        scores["breakout"] = 18
    elif daily_change_pct > 4:
        scores["breakout"] = 12
    elif daily_change_pct > 2:
        scores["breakout"] = 6
    else:
        scores["breakout"] = 0
    
    # ===== 3. MOMENTUM ACCELERATION (20 pts) =====
    # 3-day momentum accelerating
    if idx >= 3:
        mom_3d = ((today["close"] - data[idx-3]["close"]) / data[idx-3]["close"]) * 100
        mom_prev_3d = ((data[idx-1]["close"] - data[idx-4]["close"]) / data[idx-4]["close"]) * 100 if idx >= 4 else 0
        
        # Momentum accelerating
        if mom_3d > 10 and mom_3d > mom_prev_3d:
            scores["momentum"] = 20
        elif mom_3d > 7:
            scores["momentum"] = 15
        elif mom_3d > 5:
            scores["momentum"] = 10
        elif mom_3d > 3:
            scores["momentum"] = 5
        else:
            scores["momentum"] = 0
    else:
        mom_3d = 0
        scores["momentum"] = 0
    
    # ===== 4. RANGE EXPANSION (10 pts) =====
    # Today's range vs average range
    today_range = (today["high"] - today["low"]) / today["low"] * 100
    avg_ranges = [(data[i]["high"] - data[i]["low"]) / data[i]["low"] * 100 for i in range(idx-10, idx)]
    avg_range = np.mean(avg_ranges) if avg_ranges else 1
    
    if today_range > avg_range * 2.5:
        scores["range_expansion"] = 10
    elif today_range > avg_range * 2.0:
        scores["range_expansion"] = 7
    elif today_range > avg_range * 1.5:
        scores["range_expansion"] = 4
    else:
        scores["range_expansion"] = 0
    
    # ===== TOTAL SCORE =====
    total_score = sum(scores.values())
    
    details = {
        "volume_ratio": round(volume_ratio, 2),
        "daily_change_pct": round(daily_change_pct, 2),
        "momentum_3d": round(mom_3d, 2),
        "range_expansion": round(today_range / avg_range if avg_range > 0 else 1, 2),
        "scores": scores
    }
    
    return total_score, details


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return 50
    
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
    return 100 - (100 / (1 + rs))


def simulate_gpt_sentiment(data, idx):
    """
    Simulate what GPT would say about a coin based on price action.
    
    In LIVE trading, this calls OpenAI API to analyze news.
    For backtest, we simulate based on technicals.
    
    Returns: sentiment score -100 to +100
    """
    if idx < 14:
        return 0
    
    closes = [data[i]["close"] for i in range(max(0, idx-14), idx+1)]
    rsi = calculate_rsi(closes)
    
    # 7-day momentum
    mom_7d = ((data[idx]["close"] - data[idx-7]["close"]) / data[idx-7]["close"]) * 100 if idx >= 7 else 0
    
    # 3-day momentum
    mom_3d = ((data[idx]["close"] - data[idx-3]["close"]) / data[idx-3]["close"]) * 100 if idx >= 3 else 0
    
    # Volume trend
    vol_avg = np.mean([data[i]["volume"] for i in range(idx-5, idx)]) if idx >= 5 else data[idx]["volume"]
    vol_trend = data[idx]["volume"] / vol_avg if vol_avg > 0 else 1
    
    sentiment = 0
    
    # Strong bullish signals
    if mom_7d > 15 and vol_trend > 2:
        sentiment = 80
    elif mom_7d > 10 and rsi < 70:
        sentiment = 60
    elif mom_3d > 5 and vol_trend > 1.5:
        sentiment = 50
    elif mom_3d > 3:
        sentiment = 30
    elif mom_3d > 0:
        sentiment = 10
    # Bearish signals
    elif mom_7d < -15:
        sentiment = -80
    elif mom_7d < -10:
        sentiment = -50
    elif mom_3d < -5:
        sentiment = -30
    
    # RSI adjustments
    if rsi > 80:
        sentiment -= 20  # Overbought caution
    elif rsi < 30:
        sentiment += 15  # Oversold bounce potential
    
    return max(-100, min(100, sentiment))


def run_catalyst_backtest():
    """Run the NEWS/CATALYST detection backtest"""
    print("\n" + "="*70)
    print("🚀 NEWS/CATALYST DETECTION STRATEGY BACKTEST 🚀")
    print("="*70)
    print(f"\nStrategy: Detect catalyst-like events (volume spikes + breakouts)")
    print(f"Focus: High-beta altcoins that respond to news")
    print(f"Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Position Size: {POSITION_SIZE_PCT}% per trade")
    print(f"Max Positions: {MAX_POSITIONS}")
    print(f"Stop Loss: {STOP_LOSS_PCT}%")
    print(f"Take Profit: {TAKE_PROFIT_PCT}%")
    print(f"Trailing Stop: {TRAILING_STOP_PCT}%")
    print(f"Min Catalyst Score: {MIN_CATALYST_SCORE}")
    
    # Fetch data for all pairs
    print("\n📊 Fetching historical data for catalyst coins...")
    all_data = {}
    for pair in CATALYST_PAIRS:
        print(f"  Fetching {pair}...", end=" ")
        data = fetch_daily_ohlcv(pair, days=180)
        if data and len(data) >= 30:
            all_data[pair] = data
            print(f"✓ ({len(data)} days)")
        else:
            print("✗ (insufficient data)")
        time.sleep(0.5)
    
    if len(all_data) < 5:
        print("\n❌ Not enough pairs with data. Exiting.")
        return
    
    print(f"\n✅ Got data for {len(all_data)} pairs")
    
    # Find common date range
    min_len = min(len(d) for d in all_data.values())
    start_date = list(all_data.values())[0][0]["date"]
    end_date = list(all_data.values())[0][-1]["date"]
    
    print(f"📅 Backtest period: {start_date} to {end_date} ({min_len} days)")
    
    # Initialize portfolio
    cash = INITIAL_CAPITAL
    positions = []  # List of {pair, entry_price, quantity, entry_date, entry_idx, peak_price}
    trades = []
    daily_values = []
    
    # Run simulation
    print("\n🔄 Running catalyst detection simulation...")
    
    for day_idx in range(20, min_len):
        date = list(all_data.values())[0][day_idx]["date"]
        
        # Calculate portfolio value
        portfolio_value = cash
        for pos in positions:
            current_price = all_data[pos["pair"]][day_idx]["close"]
            portfolio_value += pos["quantity"] * current_price
        
        daily_values.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "num_positions": len(positions)
        })
        
        # ===== CHECK EXITS FIRST =====
        for pos in positions[:]:  # Copy list to allow modification
            current_price = all_data[pos["pair"]][day_idx]["close"]
            pnl_pct = ((current_price - pos["entry_price"]) / pos["entry_price"]) * 100
            
            # Update peak price for trailing stop
            pos["peak_price"] = max(pos.get("peak_price", pos["entry_price"]), current_price)
            
            # Calculate trailing stop level
            trailing_stop_price = pos["peak_price"] * (1 - TRAILING_STOP_PCT / 100)
            
            exit_reason = None
            
            # Check exit conditions
            if pnl_pct <= -STOP_LOSS_PCT:
                exit_reason = "STOP_LOSS"
            elif pnl_pct >= TAKE_PROFIT_PCT:
                exit_reason = "TAKE_PROFIT"
            elif current_price <= trailing_stop_price and pnl_pct > 0:
                exit_reason = "TRAILING_STOP"
            elif day_idx - pos["entry_idx"] >= 10:  # Max hold 10 days for catalyst plays
                exit_reason = "TIME_EXIT"
            
            if exit_reason:
                # Exit position
                exit_value = pos["quantity"] * current_price
                cash += exit_value
                pnl = exit_value - (pos["quantity"] * pos["entry_price"])
                
                trades.append({
                    "pair": pos["pair"],
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "quantity": pos["quantity"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "hold_days": day_idx - pos["entry_idx"],
                    "exit_reason": exit_reason,
                    "catalyst_score": pos.get("catalyst_score", 0)
                })
                
                positions.remove(pos)
        
        # ===== CHECK ENTRIES =====
        if len(positions) < MAX_POSITIONS and day_idx < min_len - 2:
            # Scan all pairs for catalyst signals
            candidates = []
            
            for pair, data in all_data.items():
                # Skip if already in position
                if any(p["pair"] == pair for p in positions):
                    continue
                
                # Calculate catalyst score
                catalyst_score, details = calculate_catalyst_score(data, day_idx)
                
                # Get simulated GPT sentiment
                gpt_sentiment = simulate_gpt_sentiment(data, day_idx)
                
                # Combined score: 70% catalyst + 30% sentiment
                combined_score = catalyst_score * 0.7 + (gpt_sentiment + 100) / 2 * 0.3
                
                if catalyst_score >= MIN_CATALYST_SCORE and gpt_sentiment >= MIN_GPT_SENTIMENT:
                    candidates.append({
                        "pair": pair,
                        "catalyst_score": catalyst_score,
                        "gpt_sentiment": gpt_sentiment,
                        "combined_score": combined_score,
                        "details": details,
                        "price": data[day_idx]["close"]
                    })
            
            # Sort by combined score and take top candidates
            candidates.sort(key=lambda x: x["combined_score"], reverse=True)
            
            for candidate in candidates[:MAX_POSITIONS - len(positions)]:
                # Enter position
                position_value = (INITIAL_CAPITAL * POSITION_SIZE_PCT / 100)
                if position_value > cash:
                    position_value = cash * 0.95
                
                if position_value < 100:
                    continue
                
                quantity = position_value / candidate["price"]
                cash -= position_value
                
                positions.append({
                    "pair": candidate["pair"],
                    "entry_price": candidate["price"],
                    "quantity": quantity,
                    "entry_date": date,
                    "entry_idx": day_idx,
                    "peak_price": candidate["price"],
                    "catalyst_score": candidate["catalyst_score"],
                    "gpt_sentiment": candidate["gpt_sentiment"]
                })
                
                if len(positions) >= MAX_POSITIONS:
                    break
    
    # Close any remaining positions at end
    for pos in positions:
        current_price = all_data[pos["pair"]][-1]["close"]
        pnl_pct = ((current_price - pos["entry_price"]) / pos["entry_price"]) * 100
        exit_value = pos["quantity"] * current_price
        pnl = exit_value - (pos["quantity"] * pos["entry_price"])
        
        trades.append({
            "pair": pos["pair"],
            "entry_date": pos["entry_date"],
            "exit_date": end_date,
            "entry_price": pos["entry_price"],
            "exit_price": current_price,
            "quantity": pos["quantity"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "hold_days": min_len - pos["entry_idx"],
            "exit_reason": "END_OF_BACKTEST",
            "catalyst_score": pos.get("catalyst_score", 0)
        })
        cash += exit_value
    
    # Calculate final metrics
    final_value = cash
    total_return = ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    # Trade statistics
    if trades:
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(trades) * 100
        
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
        
        # Max drawdown
        peak = INITIAL_CAPITAL
        max_dd = 0
        for dv in daily_values:
            if dv["portfolio_value"] > peak:
                peak = dv["portfolio_value"]
            dd = (peak - dv["portfolio_value"]) / peak * 100
            max_dd = max(max_dd, dd)
        
        # By exit reason
        exit_reasons = {}
        for t in trades:
            reason = t["exit_reason"]
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "total_pnl": 0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["total_pnl"] += t["pnl"]
    
    # Print results
    print("\n" + "="*70)
    print("📈 CATALYST STRATEGY RESULTS")
    print("="*70)
    
    print(f"\n💰 Portfolio Performance:")
    print(f"   Initial Capital:  ${INITIAL_CAPITAL:>12,.2f}")
    print(f"   Final Value:      ${final_value:>12,.2f}")
    print(f"   Total Return:     {total_return:>12.2f}%")
    print(f"   Max Drawdown:     {max_dd:>12.2f}%")
    
    if trades:
        print(f"\n📊 Trade Statistics:")
        print(f"   Total Trades:     {len(trades):>12}")
        print(f"   Winners:          {len(wins):>12}")
        print(f"   Losers:           {len(losses):>12}")
        print(f"   Win Rate:         {win_rate:>12.1f}%")
        print(f"   Avg Winner:       {avg_win:>12.2f}%")
        print(f"   Avg Loser:        {avg_loss:>12.2f}%")
        
        print(f"\n🎯 Exit Reasons:")
        for reason, stats in sorted(exit_reasons.items(), key=lambda x: -x[1]["count"]):
            print(f"   {reason}: {stats['count']} trades, ${stats['total_pnl']:+,.2f}")
        
        print(f"\n🏆 Top 5 Winning Trades:")
        top_wins = sorted(trades, key=lambda x: x["pnl"], reverse=True)[:5]
        for t in top_wins:
            print(f"   {t['pair']}: {t['pnl_pct']:+.1f}% (Score: {t['catalyst_score']}) - {t['entry_date']}")
        
        print(f"\n💔 Top 5 Losing Trades:")
        top_losses = sorted(trades, key=lambda x: x["pnl"])[:5]
        for t in top_losses:
            print(f"   {t['pair']}: {t['pnl_pct']:+.1f}% (Score: {t['catalyst_score']}) - {t['entry_date']}")
    
    # Save results
    trades_file = os.path.join(RESULTS_DIR, "catalyst_trades.csv")
    with open(trades_file, "w", newline="") as f:
        if trades:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
    
    daily_file = os.path.join(RESULTS_DIR, "catalyst_daily.csv")
    with open(daily_file, "w", newline="") as f:
        if daily_values:
            writer = csv.DictWriter(f, fieldnames=daily_values[0].keys())
            writer.writeheader()
            writer.writerows(daily_values)
    
    print(f"\n📁 Results saved to: {RESULTS_DIR}/")
    print(f"   - catalyst_trades.csv")
    print(f"   - catalyst_daily.csv")
    
    # Return summary for optimization
    return {
        "total_return": total_return,
        "win_rate": win_rate if trades else 0,
        "num_trades": len(trades),
        "max_drawdown": max_dd,
        "final_value": final_value
    }


if __name__ == "__main__":
    result = run_catalyst_backtest()
    
    print("\n" + "="*70)
    print("NEXT STEPS FOR LIVE TRADING:")
    print("="*70)
    print("""
1. This backtest simulates catalyst detection using price/volume patterns
2. For LIVE trading, the worker.py will use:
   - GPT-4o to analyze CoinGecko trending coins
   - GPT-4o to rate news sentiment (bullish/bearish/neutral)
   - Real-time volume spike detection
   - Cross-asset momentum correlation
   
3. The live strategy will:
   - Check CoinGecko trending every 15 minutes
   - Analyze any coin trending + showing volume spike
   - Use GPT to assess if the catalyst is sustainable
   - Enter only on HIGH-CONVICTION signals (score 80+)
""")
