"""
backtest_dip_buying.py
BUY THE DIP STRATEGY - Buy strong coins after significant pullbacks

Key principles:
1. Wait for ESTABLISHED coins to dip 10-20%
2. Buy the bounce when momentum starts recovering
3. Quick profits (10-15%), tight stops (5%)
4. Only buy coins with long-term bullish trend (above MA50)
"""

import os
import json
import csv
from datetime import datetime, timedelta
import requests
import numpy as np

# DIP BUYING CONFIG
INITIAL_CAPITAL = 10000.0
TRADE_ALLOCATION_PCT = 12.0
STOP_LOSS_PCT = 5.0  # Tight stop below recent low
TAKE_PROFIT_PCT = 12.0  # Quick profit on bounce
MAX_HOLD_DAYS = 5  # Short hold
TOP_N = 8

# DIP REQUIREMENTS
MIN_DIP_FROM_HIGH = -10.0  # Must be down 10%+ from recent high
MAX_DIP_FROM_HIGH = -25.0  # Not too much damage (avoid falling knives)
MIN_BOUNCE = 2.0  # Must be bouncing (up 2%+ from low)


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


def is_dip_buy_signal(ohlc_window):
    """
    Check if this is a valid dip buying opportunity
    Returns: (is_signal: bool, score: float, details: dict)
    """
    if len(ohlc_window) < 30:
        return False, 0, {}
    
    closes = np.array([float(c[4]) for c in ohlc_window])
    highs = np.array([float(c[2]) for c in ohlc_window])
    lows = np.array([float(c[3]) for c in ohlc_window])
    volumes = np.array([float(c[6]) for c in ohlc_window])
    
    current_price = closes[-1]
    
    # Find recent high (last 20 days)
    recent_high = np.max(highs[-21:-1]) if len(highs) >= 21 else np.max(highs[:-1])
    recent_low = np.min(lows[-5:])  # Low in last 5 days
    
    # Calculate dip from high
    dip_from_high = ((current_price - recent_high) / recent_high) * 100
    
    # Calculate bounce from low
    bounce_from_low = ((current_price - recent_low) / recent_low) * 100 if recent_low > 0 else 0
    
    # Daily change
    change_1d = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0
    
    # Moving averages
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
    ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price
    
    # RSI
    rsi = calculate_rsi(closes)
    
    # Volume
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    current_volume = volumes[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    details = {
        'dip_from_high': round(dip_from_high, 2),
        'bounce_from_low': round(bounce_from_low, 2),
        'change_1d': round(change_1d, 2),
        'rsi': round(rsi, 1),
        'volume_ratio': round(volume_ratio, 2),
        'above_ma50': current_price > ma50
    }
    
    # SCORING for dip buying
    score = 0
    
    # 1. Dip size (sweet spot: -10% to -20%)
    if MIN_DIP_FROM_HIGH >= dip_from_high >= MAX_DIP_FROM_HIGH:
        score += 30  # Perfect dip range
    elif -8 >= dip_from_high >= -25:
        score += 15  # Acceptable dip
    else:
        return False, 0, details  # Not enough dip or too much
    
    # 2. Bounce confirmation (must be recovering)
    if bounce_from_low >= 4:
        score += 25  # Strong bounce
    elif bounce_from_low >= MIN_BOUNCE:
        score += 15  # Decent bounce
    elif bounce_from_low < 1:
        return False, 0, details  # Not bouncing yet
    
    # 3. Today must be green (buying on up day)
    if change_1d >= 3:
        score += 20
    elif change_1d >= 1:
        score += 10
    elif change_1d < 0:
        return False, 0, details  # Don't buy on red days
    
    # 4. RSI oversold = good entry
    if rsi <= 35:
        score += 25  # Oversold = great entry
    elif rsi <= 45:
        score += 15  # Getting oversold
    elif rsi >= 60:
        score -= 10  # Already recovered too much
    
    # 5. Long-term trend must be bullish (above MA50)
    if current_price > ma50:
        score += 15
    else:
        score -= 10  # Downtrend = risky
    
    # 6. Volume on bounce
    if volume_ratio >= 1.5:
        score += 15  # High volume bounce = conviction
    
    # MINIMUM REQUIREMENTS
    is_signal = (
        MIN_DIP_FROM_HIGH >= dip_from_high >= MAX_DIP_FROM_HIGH and
        bounce_from_low >= 1.5 and
        change_1d >= 0.5 and
        score >= 50
    )
    
    return is_signal, score, details


def run_backtest():
    """Run dip buying backtest"""
    print("="*60)
    print("BUY THE DIP STRATEGY - 6 MONTH BACKTEST")
    print("="*60)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Position Size: {TRADE_ALLOCATION_PCT}% per trade")
    print(f"Stop Loss: {STOP_LOSS_PCT}%")
    print(f"Take Profit: {TAKE_PROFIT_PCT}%")
    print(f"Max Hold: {MAX_HOLD_DAYS} days")
    print(f"Dip Requirements: {MIN_DIP_FROM_HIGH}% to {MAX_DIP_FROM_HIGH}% from high")
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
    
    if not historical_data:
        print("ERROR: No historical data fetched!")
        return
    
    valid_pairs = list(historical_data.keys())
    min_length = min(len(historical_data[p]) for p in valid_pairs)
    
    print(f"\nBacktesting {len(valid_pairs)} pairs over {min_length} days...")
    
    # Initialize
    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    daily_values = []
    dip_signals = 0
    
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
            # Trailing stop for dip buys
            elif pos.get('peak_pnl', 0) >= 8 and pnl_pct < 4:
                should_exit = True
                exit_reason = "Trailing Stop"
            
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
                    'exit_reason': exit_reason
                })
                
                positions.remove(pos)
        
        # Look for dip buy signals
        dip_candidates = []
        for pair in valid_pairs:
            if day_idx >= len(historical_data[pair]):
                continue
            
            window = historical_data[pair][max(0, day_idx-30):day_idx+1]
            is_signal, score, details = is_dip_buy_signal(window)
            
            if is_signal:
                dip_signals += 1
                dip_candidates.append((pair, score, current_prices.get(pair, 0), details))
        
        dip_candidates.sort(key=lambda x: x[1], reverse=True)
        
        current_symbols = [p['symbol'] for p in positions]
        portfolio_value = cash + sum(p['quantity'] * current_prices.get(p['symbol'], p['entry_price']) for p in positions)
        
        for pair, score, price, details in dip_candidates[:TOP_N]:
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
                'peak_pnl': 0
            })
        
        portfolio_value = cash + sum(p['quantity'] * current_prices.get(p['symbol'], p['entry_price']) for p in positions)
        daily_values.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'portfolio_value': portfolio_value,
            'cash': cash,
            'positions': len(positions)
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
                'exit_reason': 'Backtest End'
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
    print(f"Dip Buy Signals:       {dip_signals}")
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
    
    with open("backtest_results/dip_trades.csv", "w", newline="") as f:
        if trades:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
    
    print("\n✅ Results saved to backtest_results/dip_trades.csv")
    
    return {'total_return': total_return, 'win_rate': len(wins)/len(trades)*100 if trades else 0, 'profit_factor': profit_factor}


if __name__ == "__main__":
    run_backtest()
