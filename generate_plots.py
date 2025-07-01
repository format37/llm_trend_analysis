#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor
import argparse
import tqdm
import threading

def generate_plots_for_symbol(args):
    """
    Generate plots for a single symbol with progress tracking
    Args:
        args: Tuple containing (symbol, lookback_days, step_days, start_date, pbar, existing_counter)
    Returns:
        tuple: (symbol, plots_generated, plots_skipped)
    """
    symbol, lookback_days, step_days, start_date, pbar, existing_counter = args
    
    try:
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
        end_date = df.index.max()
        current_end = end_date
        first_date = df.index.min()
        
        plots_generated = 0
        plots_skipped = 0
        
        while True:
            # Calculate start date for current window
            start_date_window = current_end - pd.Timedelta(days=lookback_days)
            
            # Check if we have enough data
            if start_date_window < first_date:
                break
                
            # Filter data for current window
            window_data = df[(df.index >= start_date_window) & (df.index <= current_end)]
            
            if len(window_data) == 0:
                break
                
            # Create timestamp for filename
            timestamp = current_end.strftime('%Y%m%d')
            filename = f'data/plots/{symbol}/{timestamp}_{lookback_days}.png'
            
            # Check if plot already exists
            if os.path.exists(filename):
                with existing_counter['lock']:
                    existing_counter['count'] += 1
                plots_skipped += 1
                current_end = current_end - pd.Timedelta(days=step_days)
                continue
            
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
            
            print(f"Generated: {filename} (Date range: {window_data.index.min().date()} to {window_data.index.max().date()})")
            plots_generated += 1
            
            # Move to next window (step backwards)
            current_end = current_end - pd.Timedelta(days=step_days)
        
        pbar.update(1)
        return symbol, plots_generated, plots_skipped
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        pbar.update(1)
        return symbol, 0, 0

def generate_plots(lookback_days=60, step_days=60, start_date=None, threads=1):
    """
    Generate plots for all symbols with optional parallel processing
    """
    # List files and extract symbols list
    files = os.listdir(os.path.join('data', 'source'))
    symbols = [file.split('.')[0] for file in files if file.endswith('.csv')]
    
    print(f"Found {len(symbols)} symbols to process")
    
    # Count existing plots to show initial progress
    existing_count = 0
    for symbol in symbols:
        symbol_dir = f'data/plots/{symbol}'
        if os.path.exists(symbol_dir):
            existing_files = [f for f in os.listdir(symbol_dir) if f.endswith('.png')]
            existing_count += len(existing_files)
    
    print(f"Found {existing_count} existing plots")
    
    # Counter for tracking existing files found during processing
    existing_counter = {'count': 0, 'lock': threading.Lock()}
    
    # Create progress bar
    pbar = tqdm.tqdm(total=len(symbols), desc="Processing symbols", position=0, leave=True)
    
    # Prepare arguments for parallel processing
    args_list = [(symbol, lookback_days, step_days, start_date, pbar, existing_counter) 
                 for symbol in symbols]
    
    results = []
    total_plots_generated = 0
    total_plots_skipped = 0
    
    if threads <= 1:
        # Sequential processing
        for args in args_list:
            result = generate_plots_for_symbol(args)
            results.append(result)
            total_plots_generated += result[1]
            total_plots_skipped += result[2]
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=threads) as executor:
            results = list(executor.map(generate_plots_for_symbol, args_list))
            for result in results:
                total_plots_generated += result[1]
                total_plots_skipped += result[2]
    
    pbar.close()
    
    # Print final summary
    print(f"\n=== Processing Complete ===")
    print(f"Symbols processed: {len(symbols)}")
    print(f"New plots generated: {total_plots_generated}")
    print(f"Plots skipped (already existed): {total_plots_skipped + existing_counter['count']}")
    print(f"Total plots generated: {total_plots_generated}")

def main():
    parser = argparse.ArgumentParser(description='Generate candlestick plots for financial data')
    parser.add_argument('--lookback_days', type=int, default=60, help='Number of days to include in each plot')
    parser.add_argument('--step_days', type=int, default=2, help='Number of days to step back for each new plot')
    parser.add_argument('--start_date', type=str, default=None, help='Start date for data filtering (YYYY-MM-DD)')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads for parallel processing')
    
    args = parser.parse_args()
    
    start_date = args.start_date
    if start_date:
        start_date = pd.to_datetime(start_date)
    
    generate_plots(args.lookback_days, args.step_days, start_date, args.threads)

if __name__ == "__main__":
    main()