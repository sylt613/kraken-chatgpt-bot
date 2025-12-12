"""
backtest_hybrid.py
HYBRID TRENDING + VOLUME SURGE STRATEGY

Combines multiple signals for HIGH PROBABILITY setups:
1. Social momentum (simulated trending score)
2. Volume surge (3x+ average = institutions buying)
3. Price momentum (strong recent gains)
4. Trend confirmation (above MA20)
5. RSI sweet spot (50-70 = momentum zone)

Only trade when MULTIPLE signals align!
"""

import os
import json
import csv
from datetime import datetime, timedelta
import requests
import numpy as np

# HYBRID STRATEGY CONFIG
INITIAL_CAPITAL = 10000.0
TRADE_ALLOCATION_PCT = 15.0  # Larger size for high-conviction
STOP_LOSS_PCT = 7.0  # Moderate stop
TAKE_PROFIT_PCT = 20.0  # Let winners run
MAX_HOLD_DAYS = 7
TOP_N = 5  # Only best setups

# SIGNAL REQUIREMENTS
MIN_VOLUME_SURGE = 2.5  # Volume must be 2.5x+ average
MIN_MOMENTUM_3D = 5.0  # Must be up 5%+ over 3 days
RSI_MIN = 45  # Not oversold (dead coins)
RSI_MAX = 72  # Not overbought (about to dump)


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
    rsi = 100 - (100 / (1 + rs))
    return rsi


def simulate_trending_score(ohlc_window):
    """
    Simulate a 'trending' score based on recent performance
    In live trading, this would use CoinGecko trending API
    Coins that are pumping get 'trending' organically
    """
    if len(ohlc_window) < 10:
        return 0
    
    closes = np.array([float(c[4]) for c in ohlc_window])
    volumes = np.array([float(c[6]) for c in ohlc_window])
    
    # Recent performance drives social buzz
    change_3d = ((closes[-1] - closes[-4]) / closes[-4] * 100) if len(closes) >= 4 else 0
    change_7d = ((closes[-1] - closes[-8]) / closes[-8] * 100) if len(closes) >= 8 else 0
    
    # Volume surge = more attention
    avg_volume = np.mean(volumes[-15:-1]) if len(volumes) >= 15 else np.mean(volumes[:-1])
    recent_volume = np.mean(volumes[-3:])
    volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1
    
    # Trending score (0-100)
    trending = 0
    
    if change_3d >= 15:
        trending += 40
    elif change_3d >= 8:
        trending += 25
    elif change_3d >= 4:
        trending += 15
    
    if change_7d >= 20:
        trending += 30
    elif change_7d >= 10:
        trending += 20
    elif change_7d >= 5:
        trending += 10
    
    if volume_surge >= 3:
        trending += 30
    elif volume_surge >= 2:
        trending += 20
    elif volume_surge >= 1.5:
        trending += 10
    
    return min(trending, 100)


def analyze_hybrid_signal(ohlc_window):
    """
    Analyze for HYBRID signal - multiple confirmations required
    Returns: (is_signal: bool, score: float, details: dict)
    """
    if len(ohlc_window) < 30:
        return False, 0, {}
    
    closes = np.array([float(c[4]) for c in ohlc_window])
    highs = np.array([float(c[2]) for c in ohlc_window])
    volumes = np.array([float(c[6]) for c in ohlc_window])
    
    current_price = closes[-1]
    
    # Calculate all metrics
    change_1d = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0
    change_3d = ((closes[-1] - closes[-4]) / closes[-4] * 100) if len(closes) >= 4 else 0
    change_7d = ((closes[-1] - closes[-8]) / closes[-8] * 100) if len(closes) >= 8 else 0
    
    # Volume surge
    avg_volume = np.mean(volumes[-21:-1]) if len(volumes) >= 21 else np.mean(volumes[:-1])
    current_volume = volumes[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    # 3-day volume trend (is volume increasing?)
    vol_3d_avg = np.mean(volumes[-3:])
    vol_prev_3d = np.mean(volumes[-6:-3]) if len(volumes) >= 6 else avg_volume
    volume_momentum = vol_3d_avg / vol_prev_3d if vol_prev_3d > 0 else 1
    
    # RSI
    rsi = calculate_rsi(closes)
    
    # Moving averages
    ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else current_price
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
    ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price
    
    # New high check
    high_20d = np.max(highs[-21:-1]) if len(highs) >= 21 else np.max(highs[:-1])
    is_new_high = current_price >= high_20d * 0.98  # Within 2% of high
    
    # Trending score (simulated)
    trending_score = simulate_trending_score(ohlc_window)
    
    details = {
        'change_1d': round(change_1d, 2),
        'change_3d': round(change_3d, 2),
        'change_7d': round(change_7d, 2),
        'volume_ratio': round(volume_ratio, 2),
        'volume_momentum': round(volume_momentum, 2),
        'rsi': round(rsi, 1),
        'trending': trending_score,
        'above_ma20': current_price > ma20,
        'is_new_high': is_new_high
    }
    
    # COUNT CONFIRMATIONS
    confirmations = 0
    score = 0
    
    # 1. VOLUME SURGE (critical for institutions)
    if volume_ratio >= 3.0:
        confirmations += 2
        score += 30
    elif volume_ratio >= MIN_VOLUME_SURGE:
        confirmations += 1
        score += 20
    
    # 2. MOMENTUM (price going up)
    if change_3d >= 10:
        confirmations += 2
        score += 25
    elif change_3d >= MIN_MOMENTUM_3D:
        confirmations += 1
        score += 15
    
    # 3. TRENDING (social momentum)
    if trending_score >= 70:
        confirmations += 2
        score += 25
    elif trending_score >= 50:
        confirmations += 1
        score += 15
    
    # 4. RSI SWEET SPOT
    if RSI_MIN <= rsi <= RSI_MAX:
        confirmations += 1
        score += 15
    elif rsi > RSI_MAX:
        score -= 15  # Overbought penalty
    
    # 5. TREND (above MA20)
    if current_price > ma20:
        confirmations += 1
        score += 10
        if current_price > ma10 > ma20:
            score += 5  # Strong trend structure
    
    # 6. NEW HIGH (breakout confirmation)
    if is_new_high:
        confirmations += 1
        score += 15
    
    # 7. VOLUME MOMENTUM (increasing volume)
    if volume_momentum >= 1.5:
        score += 10
    
    # 8. TODAY MUST BE GREEN
    if change_1d >= 2:
        score += 10
    elif change_1d < 0:
        score -= 20
        confirmations -= 1
    
    details['confirmations'] = confirmations
    
    # REQUIRE MULTIPLE CONFIRMATIONS
    is_signal = (
        confirmations >= 4 and  # At least 4 signals aligned
        change_1d >= 0 and  # Today is green
        change_3d >= MIN_MOMENTUM_3D and  # Strong 3-day momentum
        volume_ratio >= 1.5 and  # Above average volume
        RSI_MIN <= rsi <= RSI_MAX and  # RSI in sweet spot
        current_price > ma20 and  # Above trend
        score >= 60  # High total score
    )
    
    return is_signal, score, details


def run_backtest():
    """Run hybrid strategy backtest"""
    print("="*60)
    print("HYBRID TRENDING + VOLUME SURGE STRATEGY")
    print("="*60)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Position Size: {TRADE_ALLOCATION_PCT}%")
    print(f"Stop Loss: {STOP_LOSS_PCT}%")
    print(f"Take Profit: {TAKE_PROFIT_PCT}%")
    print(f"Max Hold: {MAX_HOLD_DAYS} days")
    print("Requires: 4+ confirmations (volume, momentum, trending, RSI, trend)")
    print("="*60)
    
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
    
    valid_pairs = list(historical_data.keys())
    min_length = min(len(historical_data[p]) for p in valid_pairs)
    
    print(f"\nBacktesting {len(valid_pairs)} pairs over {min_length} days...")
    
    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    daily_values = []
    signal_count = 0
    
    for day_idx in range(30, min_length):
        current_date = datetime.fromtimestamp(float(historical_data[valid_pairs[0]][day_idx][0]))
        
        current_prices = {}
        for pair in valid_pairs:
            if day_idx < len(historical_data[pair]):
                current_prices[pair] = float(historical_data[pair][day_idx][4])
        
        # Check exits
        for pos in positions[:]:
            current_price = current_prices.get(pos['symbol'])
            if not current_price:
                continue
            
            pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
            days_held = day_idx - pos['entry_day']
            
            should_exit = False
            exit_reason = ""
            
            if pnl_pct <= -STOP_LOSS_PCT:
                should_exit = True
                exit_reason = "Stop Loss"
            elif pnl_pct >= TAKE_PROFIT_PCT:
                should_exit = True
                exit_reason = "Take Profit"
            elif days_held >= MAX_HOLD_DAYS:
                should_exit = True
                exit_reason = "Time Exit"
            # Progressive trailing stop
            elif pos.get('peak_pnl', 0) >= 15 and pnl_pct < pos['peak_pnl'] - 5:
                should_exit = True
                exit_reason = "Trailing Stop"
            elif pos.get('peak_pnl', 0) >= 10 and pnl_pct < pos['peak_pnl'] - 4:
                should_exit = True
                exit_reason = "Trailing Stop"
            elif pos.get('peak_pnl', 0) >= 7 and pnl_pct < 3:
                should_exit = True
                exit_reason = "Protect Gains"
            
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
                    'confirmations': pos.get('confirmations', 0)
                })
                
                positions.remove(pos)
        
        # Look for hybrid signals
        candidates = []
        for pair in valid_pairs:
            if day_idx >= len(historical_data[pair]):
                continue
            
            window = historical_data[pair][max(0, day_idx-50):day_idx+1]
            is_signal, score, details = analyze_hybrid_signal(window)
            
            if is_signal:
                signal_count += 1
                candidates.append((pair, score, current_prices.get(pair, 0), details))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        current_symbols = [p['symbol'] for p in positions]
        portfolio_value = cash + sum(p['quantity'] * current_prices.get(p['symbol'], p['entry_price']) for p in positions)
        
        for pair, score, price, details in candidates[:TOP_N]:
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
                'confirmations': details.get('confirmations', 0),
                'peak_pnl': 0
            })
        
        portfolio_value = cash + sum(p['quantity'] * current_prices.get(p['symbol'], p['entry_price']) for p in positions)
        daily_values.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'portfolio_value': portfolio_value
        })
    
    # Close remaining
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
                'confirmations': pos.get('confirmations', 0)
            })
    
    # Stats
    final_value = cash
    total_return = ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0
    avg_win_pct = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss_pct = np.mean([abs(t['pnl_pct']) for t in losses]) if losses else 0
    
    profit_factor = (sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses))) if losses and sum(t['pnl'] for t in losses) != 0 else float('inf')
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Starting Capital:      ${INITIAL_CAPITAL:,.2f}")
    print(f"Ending Capital:        ${final_value:,.2f}")
    print(f"Total Return:          {total_return:+.2f}%")
    print(f"Total P&L:             ${final_value - INITIAL_CAPITAL:+,.2f}")
    print()
    print(f"Hybrid Signals:        {signal_count}")
    print(f"Total Trades:          {len(trades)}")
    print(f"Winning Trades:        {len(wins)}")
    print(f"Losing Trades:         {len(losses)}")
    print(f"Win Rate:              {len(wins)/len(trades)*100:.1f}%" if trades else "N/A")
    print()
    print(f"Average Win:           ${avg_win:+,.2f} ({avg_win_pct:+.2f}%)")
    print(f"Average Loss:          ${avg_loss:,.2f} ({avg_loss_pct:.2f}%)")
    print(f"Profit Factor:         {profit_factor:.2f}")
    print("="*60)
    
    # Save
    os.makedirs("backtest_results", exist_ok=True)
    
    with open("backtest_results/hybrid_trades.csv", "w", newline="") as f:
        if trades:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
    
    print(f"\n✅ Results saved to backtest_results/hybrid_trades.csv")
    
    return {'total_return': total_return, 'win_rate': len(wins)/len(trades)*100 if trades else 0, 'profit_factor': profit_factor}


if __name__ == "__main__":
    run_backtest()
