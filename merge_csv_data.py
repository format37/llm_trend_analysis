#!/usr/bin/env python3
import pandas as pd
import os
import argparse
from pathlib import Path

def merge_csv_files(output_file='data/merged_data.csv', limit=None):
    """
    Merge all CSV files from data/source/ into a single CSV file
    
    Args:
        output_file (str): Name of the output CSV file
        limit (int): Optional limit for number of records to include
    """
    source_dir = Path('data/source')
    
    if not source_dir.exists():
        print(f"Error: Directory {source_dir} does not exist")
        return
    
    # Get all CSV files
    csv_files = list(source_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {source_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to merge")
    
    # List to store all dataframes
    all_dataframes = []
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Extract symbol from filename (remove .csv extension)
            symbol = csv_file.stem
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Reorder columns to put symbol first
            columns = ['symbol'] + [col for col in df.columns if col != 'symbol']
            df = df[columns]
            
            all_dataframes.append(df)
            print(f"Processed {symbol}: {len(df)} records")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    if not all_dataframes:
        print("No valid CSV files were processed")
        return
    
    # Concatenate all dataframes
    print("Merging all data...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort by symbol and date for consistent ordering
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Apply limit if specified
    if limit and limit > 0:
        merged_df = merged_df.head(limit)
        print(f"Limited output to first {limit} records")
    
    # Save to output file
    merged_df.to_csv(output_file, index=False)
    
    print(f"\n=== Merge Complete ===")
    print(f"Total records: {len(merged_df)}")
    print(f"Unique symbols: {merged_df['symbol'].nunique()}")
    print(f"Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")
    print(f"Output saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Merge all CSV files from data/source/ into a single CSV')
    parser.add_argument('--output', '-o', type=str, default='data/merged_data.csv', 
                        help='Output CSV filename (default: merged_data.csv)')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit output to first N records (default: no limit)')
    
    args = parser.parse_args()
    
    merge_csv_files(args.output, args.limit)

if __name__ == "__main__":
    main() 