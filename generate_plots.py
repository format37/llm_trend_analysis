#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
import os

def generate_plots(symbol, lookback_days=60, step_days=60, start_date=None):
    # Load data
    df = pd.read_csv(os.path.join('data', 'source', f'{symbol}.csv'))
    df['date'] = pd.to_datetime(df['date'])
    if start_date:
        df = df[df['date'] >= start_date]
    df.set_index('date', inplace=True)
    df = df.sort_index()
    
    # Create data directory if it doesn't exist
    os.makedirs(f'data/plots/{symbol}', exist_ok=True)
    
    # Configure chart style
    style = mpf.make_mpf_style(
        base_mpl_style='seaborn-v0_8-whitegrid',
        gridstyle='',
        y_on_right=False,
        marketcolors=mpf.make_marketcolors(
            up='green',
            down='red',
            edge='inherit',
            wick={'up':'green', 'down':'red'},
            volume='blue'
        )
    )
    
    # Get the most recent date and work backwards
    end_date = df.index[-1]
    current_end = end_date
    
    plot_count = 0
    
    while True:
        # Calculate start date for current window
        start_date = current_end - timedelta(days=lookback_days)
        
        # Check if we have enough data
        if start_date < df.index[0]:
            break
            
        # Filter data for current window
        window_data = df[(df.index >= start_date) & (df.index <= current_end)]
        
        if len(window_data) == 0:
            break
            
        # Create timestamp for filename
        timestamp = current_end.strftime('%Y%m%d')
        filename = f'data/plots/{symbol}/{timestamp}_{lookback_days}.png'
        
        # Plot candlestick chart
        mpf.plot(
            window_data,
            type='candle',
            style=style,
            volume=False,
            figsize=(5.12, 5.12),
            tight_layout=True,
            show_nontrading=False,
            datetime_format='',
            xrotation=0,
            ylabel='',
            ylabel_lower='',
            savefig=dict(fname=filename, dpi=100, bbox_inches='tight'),
            axisoff=True
        )
        
        print(f"Generated: {filename} (Date range: {window_data.index[0].date()} to {window_data.index[-1].date()})")
        plot_count += 1
        
        # Move to next window (step backwards)
        current_end = current_end - timedelta(days=step_days)
    
    print(f"\nTotal plots generated: {plot_count}")

if __name__ == "__main__":
    lookback_days = 60
    step_days = 2
    start_date = '2022-01-01'
    # List files and extract symbols list
    files = os.listdir(os.path.join('data', 'source'))
    for file in files:
        symbol = file.split('.')[0]
        generate_plots(symbol, lookback_days, step_days, start_date)