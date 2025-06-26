#!/usr/bin/env python3
"""
Convert hourly OHLC data to daily OHLC data.
Reads all CSV files from data/hourly/ and converts them to daily format,
saving to data/source/ with the same filename.
"""

import pandas as pd
import os
from pathlib import Path


def convert_hourly_to_daily(input_file: str, output_file: str) -> None:
    """
    Convert a single hourly CSV file to daily format.
    
    Args:
        input_file: Path to input hourly CSV file
        output_file: Path to output daily CSV file
    """
    print(f"Processing {input_file}...")
    
    # Read the hourly data
    df = pd.read_csv(input_file)
    
    # Convert date column to datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Resample to daily frequency with OHLC aggregations
    daily_ohlc = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Remove NaN values (gaps like weekends, holidays)
    daily_ohlc = daily_ohlc.dropna()
    
    # Add volume column set to 0.0
    daily_ohlc['volume'] = 0.0
    
    # Reset index to make date a column again
    daily_ohlc.reset_index(inplace=True)
    
    # Format date as YYYY-MM-DD (remove time component)
    daily_ohlc['date'] = daily_ohlc['date'].dt.strftime('%Y-%m-%d')
    
    # Reorder columns to match target format
    daily_ohlc = daily_ohlc[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    # Save to output file
    daily_ohlc.to_csv(output_file, index=False)
    print(f"  → Saved {len(daily_ohlc)} daily records to {output_file}")


def main():
    """Main function to process all hourly files."""
    
    # Define directories
    hourly_dir = Path("data/hourly")
    source_dir = Path("data/source")
    
    # Create source directory if it doesn't exist
    source_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files from hourly directory
    csv_files = list(hourly_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in data/hourly/")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    
    # Process each file
    for csv_file in csv_files:
        input_file = csv_file
        output_file = source_dir / csv_file.name
        
        try:
            convert_hourly_to_daily(str(input_file), str(output_file))
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
    
    print("\nConversion completed!")


if __name__ == "__main__":
    main()
