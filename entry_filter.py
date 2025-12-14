"""
Entry Filter for 1-Minute NQ Trading
Based on real backtest analysis showing +$4,224 improvement

Filters:
1. Time: Avoid 9am-1pm ET (market open chaos)
2. ATR: Skip when 1-hour ATR > 110.64 (extreme volatility)
3. Trend: Only trade WITH the 1-hour SMA(200) trend
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import pytz

# Configuration (optimized from backtest)
ATR_THRESHOLD = 110.64  # 1.5x mean of 73.76
ATR_PERIOD = 14
SMA_PERIOD = 200

def get_nq_data():
    """Download recent 1-hour NQ data for ATR and SMA calculation"""
    # Download more data for SMA(200) - need 200+ hours
    nq = yf.download('NQ=F', period='60d', interval='1h', progress=False)
    
    if nq.empty:
        print("⚠️ Warning: Could not download NQ data")
        return None
    
    # Calculate ATR
    high = nq['High']
    low = nq['Low']
    close = nq['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    nq['ATR'] = tr.rolling(ATR_PERIOD).mean()
    
    # Calculate SMA(200)
    nq['SMA_200'] = nq['Close'].rolling(SMA_PERIOD).mean()
    
    return nq

def should_take_trade(signal_direction, current_time=None):
    """
    Main filter function - returns True if trade should be taken, False if skip
    
    Args:
        signal_direction: 'long' or 'short'
        current_time: datetime object (if None, uses current time)
    
    Returns:
        bool: True to take trade, False to skip
    """
    
    if current_time is None:
        current_time = datetime.now(pytz.timezone('US/Eastern'))
    
    # FILTER 1: Time (avoid 9am-1pm ET)
    hour_et = current_time.hour
    if 9 <= hour_et < 13:
        print(f"❌ SKIP: Time filter - {hour_et}:00 ET is during market open (avoid 9am-1pm)")
        return False
    
    # Download NQ data
    print("📊 Checking market conditions...")
    nq = get_nq_data()
    
    if nq is None:
        print("⚠️ Warning: Cannot validate filters, defaulting to SKIP for safety")
        return False
    
    # Get current values
    current_atr = nq['ATR'].iloc[-1]
    current_price = nq['Close'].iloc[-1]
    current_sma200 = nq['SMA_200'].iloc[-1]
    
    # Convert to scalar if Series
    if isinstance(current_atr, pd.Series):
        current_atr = current_atr.iloc[0] if len(current_atr) > 0 else float('nan')
    if isinstance(current_price, pd.Series):
        current_price = current_price.iloc[0] if len(current_price) > 0 else float('nan')
    if isinstance(current_sma200, pd.Series):
        current_sma200 = current_sma200.iloc[0] if len(current_sma200) > 0 else float('nan')
    
    # FILTER 2: ATR (volatility check)
    if pd.isna(current_atr):
        print("⚠️ Warning: ATR not available (not enough data), skipping trade")
        return False
    
    if current_atr > ATR_THRESHOLD:
        print(f"❌ SKIP: ATR filter - ATR={current_atr:.2f} > {ATR_THRESHOLD:.2f} (market too volatile)")
        return False
    
    # FILTER 3: Trend alignment (SMA 200)
    if pd.isna(current_sma200):
        print("⚠️ Warning: SMA(200) not available (not enough data), skipping trade")
        return False
    
    signal_lower = signal_direction.lower()
    
    if 'long' in signal_lower:
        if current_price <= current_sma200:
            print(f"❌ SKIP: Trend filter - LONG but price {current_price:.2f} < SMA200 {current_sma200:.2f} (downtrend)")
            return False
    elif 'short' in signal_lower:
        if current_price >= current_sma200:
            print(f"❌ SKIP: Trend filter - SHORT but price {current_price:.2f} > SMA200 {current_sma200:.2f} (uptrend)")
            return False
    
    # All filters passed!
    print("✅ TAKE TRADE: All filters passed")
    print(f"   Time: {hour_et}:00 ET ✓")
    print(f"   ATR: {current_atr:.2f} < {ATR_THRESHOLD:.2f} ✓")
    print(f"   Trend: {'Uptrend' if current_price > current_sma200 else 'Downtrend'} aligned with {signal_direction} ✓")
    
    return True


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("NQ ENTRY FILTER TEST")
    print("=" * 60)
    
    # Example 1: Test long signal
    print("\n📊 Testing LONG signal...")
    result = should_take_trade('long')
    
    print("\n" + "=" * 60)
    
    # Example 2: Test short signal
    print("\n📊 Testing SHORT signal...")
    result = should_take_trade('short')
    
    print("\n" + "=" * 60)
    print("\nHow to use in your trading bot:")
    print("=" * 60)
    print("""
# When you get a trading signal from LuxAlgo:

if should_take_trade('long'):  # or 'short'
    # Execute the trade
    place_order(...)
else:
    # Skip this signal
    pass
    
# That's it! The filter handles everything.
    """)
