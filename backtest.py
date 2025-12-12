"""
backtest.py
Run historical backtest of the trading strategy over the past 6 months
"""

import os
import json
import csv
from datetime import datetime, timedelta
import requests
import numpy as np

# Import config from worker
INITIAL_CAPITAL = 10000.0
TRADE_ALLOCATION_PCT = 10.0
STOP_LOSS_PCT = 15.0
TAKE_PROFIT_PCT = 25.0
TOP_N = 10

def fetch_historical_ohlc(pair, since_days=180):
    """Fetch historical OHLC data from Kraken"""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        # Use daily candles for backtest
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

def calculate_technical_score(ohlc_window):
    """Calculate technical score for a trading pair"""
    if len(ohlc_window) < 50:
        return 0
    
    closes = np.array([float(c[4]) for c in ohlc_window])
    volumes = np.array([float(c[6]) for c in ohlc_window])
    
    rsi = calculate_rsi(closes)
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes.mean()
    ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes.mean()
    
    current_price = closes[-1]
    avg_volume = np.mean(volumes[-20:])
    recent_volume = volumes[-1]
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
    
    momentum = ((closes[-1] - closes[-7]) / closes[-7] * 100) if len(closes) >= 7 else 0
    trend = "bullish" if current_price > ma50 else "bearish"
    
    # Score calculation
    score = 0
    
    if rsi:
        if 40 <= rsi <= 60:
            score += 25
        elif 30 <= rsi <= 70:
            score += 15
    
    if trend == 'bullish':
        score += 20
    
    if volume_ratio > 1.5:
        score += 15
    elif volume_ratio > 1.0:
        score += 10
    
    if momentum > 5:
        score += 20
    elif momentum > 0:
        score += 10
    
    # 24h change
    change_24h = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0
    if change_24h > 5:
        score += 10
    elif change_24h > 0:
        score += 5
    
    return score

def run_backtest():
    """Run backtest simulation"""
    print("="*60)
    print("KRAKEN SWING BOT - 6 MONTH BACKTEST")
    print("="*60)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Position Size: {TRADE_ALLOCATION_PCT}% per trade")
    print(f"Stop Loss: {STOP_LOSS_PCT}%")
    print(f"Take Profit: {TAKE_PROFIT_PCT}%")
    print("="*60)
    
    # Pairs to backtest
    pairs = ["XBTUSD", "ETHUSD", "SOLUSD", "ADAUSD", "DOTUSD", 
             "LINKUSD", "AVAXUSD", "MATICUSD", "UNIUSD", "ATOMUSD"]
    
    print("\nFetching 6 months of historical data...")
    historical_data = {}
    for pair in pairs:
        print(f"  Loading {pair}...")
        ohlc = fetch_historical_ohlc(pair, since_days=180)
        if ohlc:
            historical_data[pair] = ohlc
    
    if not historical_data:
        print("ERROR: No historical data fetched!")
        return
    
    # Find common date range - only use pairs that have data
    valid_pairs = [p for p, d in historical_data.items() if len(d) > 50]
    min_length = min(len(historical_data[p]) for p in valid_pairs)
    
    print(f"\nBacktesting {len(valid_pairs)} pairs over {min_length} days...")
    print(f"Pairs: {', '.join(valid_pairs)}")
    
    # Initialize portfolio
    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    daily_values = []
    
    # Run simulation day by day
    for day_idx in range(50, min_length):  # Start at day 50 for MA50
        current_date = datetime.fromtimestamp(float(historical_data[valid_pairs[0]][day_idx][0]))
        
        # Get current prices
        current_prices = {}
        for pair in valid_pairs:
            if day_idx < len(historical_data[pair]):
                current_prices[pair] = float(historical_data[pair][day_idx][4])  # Close price
        
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
            elif days_held >= 14:  # Max hold period for swing trade
                should_exit = True
                exit_reason = "Time Exit"
            
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
        
        # Score all pairs
        pair_scores = []
        for pair in valid_pairs:
            if day_idx >= len(historical_data[pair]):
                continue
            window = historical_data[pair][max(0, day_idx-50):day_idx+1]
            score = calculate_technical_score(window)
            if pair in current_prices:
                pair_scores.append((pair, score, current_prices[pair]))
        
        # Sort by score
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Open new positions
        current_symbols = [p['symbol'] for p in positions]
        portfolio_value = cash + sum(p['quantity'] * current_prices.get(p['symbol'], p['entry_price']) for p in positions)
        
        for pair, score, price in pair_scores[:TOP_N]:
            if pair in current_symbols:
                continue
            
            if len(positions) >= TOP_N:
                break
            
            if score < 40:  # Minimum score threshold
                continue
            
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
                'score': score
            })
        
        # Record daily portfolio value
        portfolio_value = cash + sum(p['quantity'] * current_prices.get(p['symbol'], p['entry_price']) for p in positions)
        daily_values.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'portfolio_value': portfolio_value,
            'cash': cash,
            'num_positions': len(positions)
        })
    
    # Close remaining positions at final prices
    for pos in positions:
        current_price = current_prices.get(pos['symbol'], pos['entry_price'])
        position_value = pos['quantity'] * current_price
        pnl = position_value - (pos['quantity'] * pos['entry_price'])
        pnl_pct = (pnl / (pos['quantity'] * pos['entry_price'])) * 100
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
            'days_held': min_length - 1 - pos['entry_day'],
            'exit_reason': 'Backtest End'
        })
    
    # Calculate performance metrics
    final_value = cash
    total_return = ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    
    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    avg_win_pct = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss_pct = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Starting Capital:      ${INITIAL_CAPITAL:,.2f}")
    print(f"Ending Capital:        ${final_value:,.2f}")
    print(f"Total Return:          {total_return:+.2f}%")
    print(f"Total P&L:             ${final_value - INITIAL_CAPITAL:+,.2f}")
    print(f"\nTotal Trades:          {len(trades)}")
    print(f"Winning Trades:        {len(winning_trades)}")
    print(f"Losing Trades:         {len(losing_trades)}")
    print(f"Win Rate:              {win_rate:.1f}%")
    print(f"\nAverage Win:           ${avg_win:+,.2f} ({avg_win_pct:+.2f}%)")
    print(f"Average Loss:          ${avg_loss:+,.2f} ({avg_loss_pct:+.2f}%)")
    if avg_loss != 0:
        print(f"Profit Factor:         {abs(avg_win/avg_loss):.2f}")
    print("="*60)
    
    # Save results to CSV
    os.makedirs("backtest_results", exist_ok=True)
    
    # Save trades
    with open("backtest_results/trades.csv", "w", newline="") as f:
        fieldnames = ['entry_date', 'exit_date', 'symbol', 'entry_price', 'exit_price', 
                     'quantity', 'pnl', 'pnl_pct', 'days_held', 'exit_reason']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)
    
    # Save daily values
    with open("backtest_results/daily_values.csv", "w", newline="") as f:
        fieldnames = ['date', 'portfolio_value', 'cash', 'num_positions']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(daily_values)
    
    print("\n✅ Backtest results saved to backtest_results/")
    print("   - trades.csv (all trades)")
    print("   - daily_values.csv (daily portfolio values)")
    print("\nOpen these CSV files in Excel or Google Sheets to analyze!")

if __name__ == "__main__":
    run_backtest()
