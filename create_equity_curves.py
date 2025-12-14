import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

print("=" * 80)
print("EQUITY CURVE VISUALIZATION - WITH vs WITHOUT FILTERS")
print("=" * 80)

# Load the data
df = pd.read_csv('backtest_data.csv')
entries = df[df['Type'].str.contains('Entry', case=False, na=False)].copy()
entries['Datetime'] = pd.to_datetime(entries['Date/Time'])
entries = entries.sort_values('Datetime').reset_index(drop=True)

# Load filter results
all_trades = pd.read_csv('all_trades_with_filters.csv')
all_trades['Datetime'] = pd.to_datetime(all_trades['Datetime'])

print(f"\nTotal trades: {len(entries)}")
print(f"Original P&L: ${entries['Net P&L USD'].sum():.2f}")

# Calculate equity curves
# Original (no filters)
entries['Cumulative_PnL'] = entries['Net P&L USD'].cumsum()

# With filters (only trades that passed)
filtered_trades = all_trades[all_trades['Filter_ALL'] == True].copy()
filtered_trades = filtered_trades.sort_values('Datetime')
filtered_trades['Cumulative_PnL'] = filtered_trades['Net P&L USD'].cumsum()

print(f"Filtered trades: {len(filtered_trades)}")
print(f"Filtered P&L: ${filtered_trades['Net P&L USD'].sum():.2f}")

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Top plot: Equity curves comparison
ax1.plot(entries['Datetime'], entries['Cumulative_PnL'], 
         label='Original (No Filters)', color='#e74c3c', linewidth=2, alpha=0.7)
ax1.plot(filtered_trades['Datetime'], filtered_trades['Cumulative_PnL'], 
         label='With Filters (Time + ATR + Trend)', color='#27ae60', linewidth=2.5)

ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.fill_between(entries['Datetime'], 0, entries['Cumulative_PnL'], 
                  alpha=0.1, color='red')
ax1.fill_between(filtered_trades['Datetime'], 0, filtered_trades['Cumulative_PnL'], 
                  alpha=0.2, color='green')

ax1.set_title('Equity Curve Comparison: Filtered vs Unfiltered Strategy', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Cumulative P&L (USD)', fontsize=12)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Add performance metrics to plot
orig_final = entries['Cumulative_PnL'].iloc[-1]
filt_final = filtered_trades['Cumulative_PnL'].iloc[-1]
improvement = filt_final - orig_final
improvement_pct = (improvement / abs(orig_final)) * 100 if orig_final != 0 else 0

# Calculate max drawdowns
orig_running_max = entries['Cumulative_PnL'].cummax()
orig_drawdown = entries['Cumulative_PnL'] - orig_running_max
orig_max_dd = orig_drawdown.min()

filt_running_max = filtered_trades['Cumulative_PnL'].cummax()
filt_drawdown = filtered_trades['Cumulative_PnL'] - filt_running_max
filt_max_dd = filt_drawdown.min()

# Text box with stats
stats_text = f"""
ORIGINAL STRATEGY
• Trades: {len(entries)}
• Final P&L: ${orig_final:,.2f}
• Max DD: ${orig_max_dd:,.2f}
• P/DD Ratio: {abs(orig_final/orig_max_dd):.2f}

WITH FILTERS
• Trades: {len(filtered_trades)} ({len(filtered_trades)/len(entries)*100:.1f}%)
• Final P&L: ${filt_final:,.2f}
• Max DD: ${filt_max_dd:,.2f}
• P/DD Ratio: {abs(filt_final/filt_max_dd):.2f}

IMPROVEMENT
• P&L: +${improvement:,.2f} (+{improvement_pct:.1f}%)
• DD Reduction: ${filt_max_dd - orig_max_dd:,.2f}
• Filtered Out: {len(entries) - len(filtered_trades)} trades
"""

ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         family='monospace')

# Bottom plot: Drawdown comparison
ax2.fill_between(entries['Datetime'], 0, orig_drawdown, 
                  color='#e74c3c', alpha=0.3, label='Original DD')
ax2.fill_between(filtered_trades['Datetime'], 0, filt_drawdown, 
                  color='#27ae60', alpha=0.5, label='Filtered DD')
ax2.plot(entries['Datetime'], orig_drawdown, color='#e74c3c', linewidth=1, alpha=0.7)
ax2.plot(filtered_trades['Datetime'], filt_drawdown, color='#27ae60', linewidth=1.5)

ax2.axhline(y=orig_max_dd, color='red', linestyle='--', alpha=0.5, 
            label=f'Original Max DD: ${orig_max_dd:,.0f}')
ax2.axhline(y=filt_max_dd, color='green', linestyle='--', alpha=0.5,
            label=f'Filtered Max DD: ${filt_max_dd:,.0f}')

ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Drawdown (USD)', fontsize=12)
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()

# Save the figure
output_file = 'equity_curve_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Chart saved: {output_file}")

# Also create a monthly performance comparison
fig2, ax = plt.subplots(figsize=(14, 8))

# Group by month
entries['Month'] = entries['Datetime'].dt.to_period('M')
filtered_trades['Month'] = filtered_trades['Datetime'].dt.to_period('M')

monthly_orig = entries.groupby('Month')['Net P&L USD'].sum()
monthly_filt = filtered_trades.groupby('Month')['Net P&L USD'].sum()

# Align months
all_months = monthly_orig.index.union(monthly_filt.index)
monthly_orig = monthly_orig.reindex(all_months, fill_value=0)
monthly_filt = monthly_filt.reindex(all_months, fill_value=0)

x = np.arange(len(all_months))
width = 0.35

bars1 = ax.bar(x - width/2, monthly_orig.values, width, 
               label='Original', color='#e74c3c', alpha=0.7)
bars2 = ax.bar(x + width/2, monthly_filt.values, width,
               label='With Filters', color='#27ae60', alpha=0.8)

ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Monthly P&L (USD)', fontsize=12)
ax.set_title('Monthly Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([str(m) for m in all_months], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 100:  # Only label significant values
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.0f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8)

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()

monthly_file = 'monthly_performance_comparison.png'
plt.savefig(monthly_file, dpi=300, bbox_inches='tight')
print(f"✅ Chart saved: {monthly_file}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nOriginal Strategy:")
print(f"  • {len(entries)} trades over {(entries['Datetime'].max() - entries['Datetime'].min()).days} days")
print(f"  • Final P&L: ${orig_final:,.2f}")
print(f"  • Max Drawdown: ${orig_max_dd:,.2f}")
print(f"  • Profit/DD Ratio: {abs(orig_final/orig_max_dd):.2f}")

print(f"\nWith Filters (Time + ATR + Trend):")
print(f"  • {len(filtered_trades)} trades ({len(filtered_trades)/len(entries)*100:.1f}% of signals)")
print(f"  • Final P&L: ${filt_final:,.2f}")
print(f"  • Max Drawdown: ${filt_max_dd:,.2f}")
print(f"  • Profit/DD Ratio: {abs(filt_final/filt_max_dd):.2f}")

print(f"\nImprovement:")
print(f"  • P&L: +${improvement:,.2f} (+{improvement_pct:.1f}%)")
print(f"  • DD Reduction: ${filt_max_dd - orig_max_dd:,.2f} ({(1 - filt_max_dd/orig_max_dd)*100:.1f}% better)")
print(f"  • Profit/DD Ratio: {abs(filt_final/filt_max_dd) - abs(orig_final/orig_max_dd):+.2f}x")
print(f"  • Trade Reduction: {len(entries) - len(filtered_trades)} trades filtered (avg P&L: ${(orig_final - filt_final)/(len(entries) - len(filtered_trades)):,.2f})")

print("\n✅ Charts created successfully!")
print(f"   • equity_curve_comparison.png")
print(f"   • monthly_performance_comparison.png")
