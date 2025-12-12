"""
backtest_momentum.py
MOMENTUM BREAKOUT STRATEGY - Chase coins that are ALREADY surging

Key principles:
1. Don't predict - FOLLOW proven momentum
2. Only buy coins with strong 24h gains (5%+) AND volume surge (2x+)
3. Quick exits - take profit fast (15%), tight stop (6%)
4. Short hold periods (3-5 days max)
5. Prioritize coins breaking out with massive volume
"""

import os
import json
import csv
from datetime import datetime, timedelta
import requests
import numpy as np

# MOMENTUM BREAKOUT CONFIG
INITIAL_CAPITAL = 10000.0
TRADE_ALLOCATION_PCT = 12.0  # 12% per position
STOP_LOSS_PCT = 5.0  # Tight initial stop
TAKE_PROFIT_PCT = 18.0  # Let winners run a bit
MAX_HOLD_DAYS = 6  # Slightly longer hold
TOP_N = 8  # 8 positions max

# BREAKOUT REQUIREMENTS - BALANCED
MIN_24H_GAIN = 4.0  # Must be up 4%+ in 24h
MIN_3D_GAIN = 7.0  # Must be up 7%+ in 3 days
MIN_VOLUME_RATIO = 1.8  # Volume must be 1.8x+ normal
MAX_RSI = 80  # Avoid extremely overbought


def fetch_historical_ohlc(pair, since_days=180):
    """Fetch historical OHLC data from Kraken"""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        since = int((datetime.now() - timedelta(days=since_days)).timestamp())
        resp = requests.get(url, params={"pair": pair, "interval": 1440, "since": since}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("error") and len(data["error"]) > 0:
            return []
        
        result = data.get("result", {})
        pair_key = [k for k in result.keys() if k != "last"][0] if result else None
        if not pair_key:
            return []
        
        return result[pair_key]
    except Exception as e:
        print(f"Historical OHLC error for {pair}: {e}")
        return []


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return 50  # Default to neutral
    
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


def is_momentum_breakout(ohlc_window):
    """
    Check if this is a valid momentum breakout signal
    Returns: (is_breakout: bool, score: float, details: dict)
    """
    if len(ohlc_window) < 20:
        return False, 0, {}
    
    closes = np.array([float(c[4]) for c in ohlc_window])
    highs = np.array([float(c[2]) for c in ohlc_window])
    lows = np.array([float(c[3]) for c in ohlc_window])
    volumes = np.array([float(c[6]) for c in ohlc_window])
    
    current_price = closes[-1]
    
    # Calculate key metrics
    change_1d = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0
    change_3d = ((closes[-1] - closes[-4]) / closes[-4] * 100) if len(closes) >= 4 else 0
    change_7d = ((closes[-1] - closes[-8]) / closes[-8] * 100) if len(closes) >= 8 else 0
    
    # Volume analysis
    avg_volume_20d = np.mean(volumes[-21:-1]) if len(volumes) >= 21 else np.mean(volumes[:-1])
    current_volume = volumes[-1]
    volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1
    
    # RSI
    rsi = calculate_rsi(closes)
    
    # Moving averages
    ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else current_price
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
    
    # Check for new high (breakout confirmation)
    high_20d = np.max(highs[-21:-1]) if len(highs) >= 21 else np.max(highs[:-1])
    is_new_high = current_price > high_20d
    
    # BREAKOUT CRITERIA
    details = {
        'change_1d': round(change_1d, 2),
        'change_3d': round(change_3d, 2),
        'change_7d': round(change_7d, 2),
        'volume_ratio': round(volume_ratio, 2),
        'rsi': round(rsi, 1),
        'is_new_high': is_new_high,
        'above_ma10': current_price > ma10,
        'above_ma20': current_price > ma20
    }
    
    # SCORING - Pure momentum focus
    score = 0
    
    # 1. 24h momentum (CRITICAL)
    if change_1d >= 10:
        score += 40  # Massive daily move
    elif change_1d >= MIN_24H_GAIN:
        score += 25  # Strong daily move
    elif change_1d >= 3:
        score += 10  # Decent daily move
    elif change_1d < 0:
        return False, 0, details  # REJECT - not moving up today
    
    # 2. 3-day momentum
    if change_3d >= 15:
        score += 30
    elif change_3d >= MIN_3D_GAIN:
        score += 20
    elif change_3d >= 5:
        score += 10
    
    # 3. Volume surge (CRITICAL for breakout confirmation)
    if volume_ratio >= 3.0:
        score += 30  # Massive volume = institutions buying
    elif volume_ratio >= MIN_VOLUME_RATIO:
        score += 20
    elif volume_ratio >= 1.5:
        score += 10
    elif volume_ratio < 1.0:
        score -= 10  # Low volume = weak move
    
    # 4. New 20-day high (breakout confirmation)
    if is_new_high:
        score += 20
    
    # 5. Price above moving averages
    if current_price > ma10 > ma20:
        score += 15  # Strong uptrend structure
    elif current_price > ma20:
        score += 5
    
    # 6. RSI check (avoid extremes)
    if rsi > MAX_RSI:
        score -= 20  # Too overbought, likely to pull back
    elif rsi > 70:
        score -= 5  # Getting overbought
    elif 50 <= rsi <= 70:
        score += 10  # Sweet spot for momentum
    
    # MINIMUM REQUIREMENTS for breakout signal
    is_breakout = (
        change_1d >= 4 and  # Up at least 4% today
        change_3d >= 6 and  # Up at least 6% over 3 days
        volume_ratio >= 1.5 and  # Above average volume
        rsi <= MAX_RSI and  # Not extremely overbought
        score >= 55  # Decent score threshold
    )
    
    return is_breakout, score, details


def run_backtest():
    """Run momentum breakout backtest"""
    print("="*60)
    print("MOMENTUM BREAKOUT STRATEGY - 6 MONTH BACKTEST")
    print("="*60)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Position Size: {TRADE_ALLOCATION_PCT}% per trade")
    print(f"Stop Loss: {STOP_LOSS_PCT}% (tight - exit fast if momentum fails)")
    print(f"Take Profit: {TAKE_PROFIT_PCT}% (quick profit taking)")
    print(f"Max Hold: {MAX_HOLD_DAYS} days")
    print(f"Breakout Requirements: {MIN_24H_GAIN}%+ daily, {MIN_VOLUME_RATIO}x+ volume")
    print("="*60)
    
    # Extended pairs list - more opportunities
    pairs = ["XBTUSD", "ETHUSD", "SOLUSD", "ADAUSD", "DOTUSD", 
             "LINKUSD", "AVAXUSD", "UNIUSD", "ATOMUSD",
             "LTCUSD", "XRPUSD", "DOGEUSD", "SHIBUSD"]
    
    print("\nFetching 6 months of historical data...")
    historical_data = {}
    for pair in pairs:
        print(f"  Loading {pair}...")
        ohlc = fetch_historical_ohlc(pair, since_days=180)
        if ohlc and len(ohlc) > 50:
            historical_data[pair] = ohlc
    
    if not historical_data:
        print("ERROR: No historical data fetched!")
        return
    
    valid_pairs = list(historical_data.keys())
    min_length = min(len(historical_data[p]) for p in valid_pairs)
    
    print(f"\nBacktesting {len(valid_pairs)} pairs over {min_length} days...")
    print(f"Pairs: {', '.join(valid_pairs)}")
    
    # Initialize portfolio
    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    daily_values = []
    
    # Statistics
    breakout_signals = 0
    
    # Run simulation
    for day_idx in range(30, min_length):  # Start at day 30 for enough history
        current_date = datetime.fromtimestamp(float(historical_data[valid_pairs[0]][day_idx][0]))
        
        # Get current prices
        current_prices = {}
        for pair in valid_pairs:
            if day_idx < len(historical_data[pair]):
                current_prices[pair] = float(historical_data[pair][day_idx][4])
        
        # Check exits FIRST
        for pos in positions[:]:
            current_price = current_prices.get(pos['symbol'])
            if not current_price:
                continue
            
            pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
            days_held = day_idx - pos['entry_day']
            
            should_exit = False
            exit_reason = ""
            
            # Exit conditions
            if pnl_pct <= -STOP_LOSS_PCT:
                should_exit = True
                exit_reason = "Stop Loss"
            elif pnl_pct >= TAKE_PROFIT_PCT:
                should_exit = True
                exit_reason = "Take Profit"
            elif days_held >= MAX_HOLD_DAYS:
                should_exit = True
                exit_reason = "Time Exit"
            # TRAILING STOP: Lock in gains progressively
            elif pos.get('peak_pnl', 0) >= 12 and pnl_pct < pos['peak_pnl'] - 4:
                should_exit = True
                exit_reason = "Trailing Stop (locked profit)"
            elif pos.get('peak_pnl', 0) >= 8 and pnl_pct < pos['peak_pnl'] - 3:
                should_exit = True
                exit_reason = "Trailing Stop"
            elif pos.get('peak_pnl', 0) >= 5 and pnl_pct < 2:
                should_exit = True
                exit_reason = "Trailing Stop (protect gains)"
            
            # Update peak P&L for trailing stop
            if pnl_pct > pos.get('peak_pnl', 0):
                pos['peak_pnl'] = pnl_pct
            
            if should_exit:
                position_value = pos['quantity'] * current_price
                pnl = position_value - (pos['quantity'] * pos['entry_price'])
                cash += position_value
                
                trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': current_date.strftime('%Y-%m-%d'),
                    'symbol': pos['symbol'],
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'quantity': pos['quantity'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                    'entry_score': pos.get('score', 0)
                })
                
                positions.remove(pos)
        
        # Look for NEW breakout signals
        breakout_candidates = []
        for pair in valid_pairs:
            if day_idx >= len(historical_data[pair]):
                continue
            
            window = historical_data[pair][max(0, day_idx-30):day_idx+1]
            is_breakout, score, details = is_momentum_breakout(window)
            
            if is_breakout:
                breakout_signals += 1
                breakout_candidates.append((pair, score, current_prices.get(pair, 0), details))
        
        # Sort by score (highest first)
        breakout_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Open new positions
        current_symbols = [p['symbol'] for p in positions]
        portfolio_value = cash + sum(p['quantity'] * current_prices.get(p['symbol'], p['entry_price']) for p in positions)
        
        for pair, score, price, details in breakout_candidates[:TOP_N]:
            if pair in current_symbols:
                continue
            
            if len(positions) >= TOP_N:
                break
            
            position_size_usd = portfolio_value * (TRADE_ALLOCATION_PCT / 100.0)
            
            if cash < position_size_usd:
                continue
            
            quantity = position_size_usd / price
            cash -= position_size_usd
            
            positions.append({
                'symbol': pair,
                'entry_price': price,
                'entry_day': day_idx,
                'entry_date': current_date.strftime('%Y-%m-%d'),
                'quantity': quantity,
                'score': score,
                'details': details,
                'peak_pnl': 0
            })
        
        # Track daily portfolio value
        portfolio_value = cash + sum(p['quantity'] * current_prices.get(p['symbol'], p['entry_price']) for p in positions)
        daily_values.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'portfolio_value': portfolio_value,
            'cash': cash,
            'positions': len(positions)
        })
    
    # Close remaining positions at end
    for pos in positions:
        pair = pos['symbol']
        if pair in current_prices:
            current_price = current_prices[pair]
            pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
            position_value = pos['quantity'] * current_price
            pnl = position_value - (pos['quantity'] * pos['entry_price'])
            cash += position_value
            
            trades.append({
                'entry_date': pos['entry_date'],
                'exit_date': daily_values[-1]['date'] if daily_values else 'N/A',
                'symbol': pos['symbol'],
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'quantity': pos['quantity'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'days_held': min_length - pos['entry_day'],
                'exit_reason': 'Backtest End',
                'entry_score': pos.get('score', 0)
            })
    
    # Calculate final stats
    final_value = cash
    total_return = ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0
    avg_win_pct = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss_pct = np.mean([abs(t['pnl_pct']) for t in losses]) if losses else 0
    
    profit_factor = (sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses))) if losses and sum(t['pnl'] for t in losses) != 0 else float('inf')
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Starting Capital:      ${INITIAL_CAPITAL:,.2f}")
    print(f"Ending Capital:        ${final_value:,.2f}")
    print(f"Total Return:          {total_return:+.2f}%")
    print(f"Total P&L:             ${final_value - INITIAL_CAPITAL:+,.2f}")
    print()
    print(f"Total Breakout Signals: {breakout_signals}")
    print(f"Total Trades:          {len(trades)}")
    print(f"Winning Trades:        {len(wins)}")
    print(f"Losing Trades:         {len(losses)}")
    print(f"Win Rate:              {len(wins)/len(trades)*100:.1f}%" if trades else "N/A")
    print()
    print(f"Average Win:           ${avg_win:+,.2f} ({avg_win_pct:+.2f}%)")
    print(f"Average Loss:          ${avg_loss:,.2f} ({avg_loss_pct:.2f}%)")
    print(f"Profit Factor:         {profit_factor:.2f}")
    print("="*60)
    
    # Save results
    os.makedirs("backtest_results", exist_ok=True)
    
    with open("backtest_results/momentum_trades.csv", "w", newline="") as f:
        if trades:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
    
    with open("backtest_results/momentum_daily.csv", "w", newline="") as f:
        if daily_values:
            writer = csv.DictWriter(f, fieldnames=daily_values[0].keys())
            writer.writeheader()
            writer.writerows(daily_values)
    
    print("\n✅ Backtest results saved to backtest_results/")
    print("   - momentum_trades.csv")
    print("   - momentum_daily.csv")
    
    return {
        'total_return': total_return,
        'win_rate': len(wins)/len(trades)*100 if trades else 0,
        'profit_factor': profit_factor,
        'trades': len(trades)
    }


if __name__ == "__main__":
    run_backtest()
