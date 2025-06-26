import pandas as pd
import json
import glob
import os

def collect_results():
    """Collect all JSON results into a single DataFrame and save as CSV"""
    
    # Find all JSON files in results directories
    json_files = glob.glob("data/results/**/*.json", recursive=True)
    
    print(f"Found {len(json_files)} JSON result files")
    
    if len(json_files) == 0:
        print("No JSON files found in data/results/")
        return
    
    # Collect all results
    all_results = []
    
    for json_file in json_files:
        try:
            # Extract symbol from path (data/results/{symbol}/{filename}.json)
            path_parts = json_file.split(os.sep)
            symbol = path_parts[-2]  # Get the symbol (second to last part)
            
            # Load JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Add symbol to the data
            data['symbol'] = symbol
            
            all_results.append(data)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    if len(all_results) == 0:
        print("No valid results collected")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns to put symbol first
    columns = ['symbol'] + [col for col in df.columns if col != 'symbol']
    df = df[columns]
    
    # Sort by symbol and date
    df = df.sort_values(['symbol', 'date'])
    
    # Save to CSV
    output_file = "data/trend_analysis_results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n=== Results Summary ===")
    print(f"Total results collected: {len(df)}")
    print(f"Unique symbols: {df['symbol'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Results saved to: {output_file}")
    
    # Show sample of data
    print(f"\nSample data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    collect_results() 