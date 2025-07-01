import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

def load_large_csv():
    """Load the large CSV file efficiently"""
    print("Loading large CSV file...")
    
    # Read CSV in chunks to handle large file
    chunk_size = 10000
    chunks = []
    
    try:
        for chunk in pd.read_csv("data/trend_analysis_results.csv", chunksize=chunk_size):
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(df)} records from CSV file")
        
        # Convert date to datetime for better handling
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        
        return df
        
    except FileNotFoundError:
        print("❌ File data/trend_analysis_results.csv not found")
        return None
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return None

def create_overall_consistency_plots(df):
    """Create consistency plots for all data"""
    
    print("Creating overall consistency plots...")
    
    # Create figure with 5 subplots for all metrics
    fig, axes = plt.subplots(5, 1, figsize=(16, 24))
    fig.suptitle('Overall Trend Analysis Consistency (All Symbols)', fontsize=16, fontweight='bold')
    
    # Plot 1: Direction over time
    if 'direction' in df.columns:
        # Sample data for plotting if too large
        if len(df) > 50000:
            sample_df = df.sample(n=50000, random_state=42)
            axes[0].scatter(sample_df['date'], sample_df['direction'], alpha=0.1, s=1)
            axes[0].set_title('Direction Over Time (50k sample)')
        else:
            axes[0].scatter(df['date'], df['direction'], alpha=0.3, s=1)
            axes[0].set_title('Direction Over Time')
        
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Direction (-1 to 1)')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: Trend Duration
    if 'trend_duration' in df.columns:
        if len(df) > 50000:
            sample_df = df.sample(n=50000, random_state=42)
            axes[1].scatter(sample_df['date'], sample_df['trend_duration'], alpha=0.1, s=1)
            axes[1].set_title('Trend Duration Over Time (50k sample)')
        else:
            axes[1].scatter(df['date'], df['trend_duration'], alpha=0.3, s=1)
            axes[1].set_title('Trend Duration Over Time')
        
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Duration (days)')
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Trend Vulnerability Score
    if 'trend_vulnerability_score' in df.columns:
        if len(df) > 50000:
            sample_df = df.sample(n=50000, random_state=42)
            axes[2].scatter(sample_df['date'], sample_df['trend_vulnerability_score'], alpha=0.1, s=1)
            axes[2].set_title('Trend Vulnerability Score Over Time (50k sample)')
        else:
            axes[2].scatter(df['date'], df['trend_vulnerability_score'], alpha=0.3, s=1)
            axes[2].set_title('Trend Vulnerability Score Over Time')
        
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('TVS (0 to 1)')
        axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Price Positioning
    if 'price_positioning' in df.columns:
        if len(df) > 50000:
            sample_df = df.sample(n=50000, random_state=42)
            axes[3].scatter(sample_df['date'], sample_df['price_positioning'], alpha=0.1, s=1)
            axes[3].set_title('Price Positioning Over Time (50k sample)')
        else:
            axes[3].scatter(df['date'], df['price_positioning'], alpha=0.3, s=1)
            axes[3].set_title('Price Positioning Over Time')
        
        axes[3].set_xlabel('Date')
        axes[3].set_ylabel('Position (0=trendline, 1=boundary)')
        axes[3].grid(True, alpha=0.3)
        axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[3].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 5: Continuation Likelihood
    if 'continuation_likelihood' in df.columns:
        if len(df) > 50000:
            sample_df = df.sample(n=50000, random_state=42)
            axes[4].scatter(sample_df['date'], sample_df['continuation_likelihood'], alpha=0.1, s=1)
            axes[4].set_title('Continuation Likelihood Over Time (50k sample)')
        else:
            axes[4].scatter(df['date'], df['continuation_likelihood'], alpha=0.3, s=1)
            axes[4].set_title('Continuation Likelihood Over Time')
        
        axes[4].set_xlabel('Date')
        axes[4].set_ylabel('Likelihood (0 to 1)')
        axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"data/reports/overall_consistency_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Overall consistency plots saved to {plot_filename}")
    
    plt.close(fig)

def create_distribution_plots(df):
    """Create distribution plots for key metrics"""
    
    print("Creating distribution plots...")
    
    # Create figure with 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Trend Metrics Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Direction distribution
    if 'direction' in df.columns:
        axes[0, 0].hist(df['direction'].dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Direction Distribution')
        axes[0, 0].set_xlabel('Direction')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].grid(True, alpha=0.3)
    
    # Trend Duration distribution
    if 'trend_duration' in df.columns:
        axes[0, 1].hist(df['trend_duration'].dropna(), bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Trend Duration Distribution')
        axes[0, 1].set_xlabel('Duration (days)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
    
    # TVS distribution
    if 'trend_vulnerability_score' in df.columns:
        axes[0, 2].hist(df['trend_vulnerability_score'].dropna(), bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_title('Trend Vulnerability Score Distribution')
        axes[0, 2].set_xlabel('TVS')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Price Positioning distribution
    if 'price_positioning' in df.columns:
        axes[1, 0].hist(df['price_positioning'].dropna(), bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('Price Positioning Distribution')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        axes[1, 0].axvline(x=1, color='gray', linestyle='--', alpha=0.7)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Continuation Likelihood distribution
    if 'continuation_likelihood' in df.columns:
        axes[1, 1].hist(df['continuation_likelihood'].dropna(), bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('Continuation Likelihood Distribution')
        axes[1, 1].set_xlabel('Likelihood')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Symbol count (top 20 symbols)
    if 'symbol' in df.columns:
        symbol_counts = df['symbol'].value_counts().head(20)
        axes[1, 2].bar(range(len(symbol_counts)), symbol_counts.values, alpha=0.7, color='brown')
        axes[1, 2].set_title('Top 20 Symbols by Record Count')
        axes[1, 2].set_xlabel('Symbol Rank')
        axes[1, 2].set_ylabel('Record Count')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"data/reports/distributions_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Distribution plots saved to {plot_filename}")
    
    plt.close(fig)

def create_correlation_plots(df):
    """Create correlation and scatter plots"""
    
    print("Creating correlation plots...")
    
    # Select numeric columns for correlation
    numeric_cols = ['direction', 'trend_duration', 'trend_vulnerability_score', 
                   'price_positioning', 'continuation_likelihood']
    
    # Filter to only existing columns
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print("Not enough numeric columns for correlation analysis")
        return
    
    # Create correlation matrix
    corr_data = df[available_cols].dropna()
    
    # Sample if too large
    if len(corr_data) > 100000:
        corr_data = corr_data.sample(n=100000, random_state=42)
        print(f"Using sample of 100k records for correlation analysis")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Trend Metrics Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Correlation heatmap
    correlation_matrix = corr_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0, 0], fmt='.3f')
    axes[0, 0].set_title('Correlation Matrix')
    
    # Direction vs Continuation Likelihood
    if 'direction' in corr_data.columns and 'continuation_likelihood' in corr_data.columns:
        axes[0, 1].scatter(corr_data['direction'], corr_data['continuation_likelihood'], 
                          alpha=0.1, s=1)
        axes[0, 1].set_title('Direction vs Continuation Likelihood')
        axes[0, 1].set_xlabel('Direction')
        axes[0, 1].set_ylabel('Continuation Likelihood')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Trend Duration vs TVS
    if 'trend_duration' in corr_data.columns and 'trend_vulnerability_score' in corr_data.columns:
        axes[1, 0].scatter(corr_data['trend_duration'], corr_data['trend_vulnerability_score'], 
                          alpha=0.1, s=1)
        axes[1, 0].set_title('Trend Duration vs Vulnerability Score')
        axes[1, 0].set_xlabel('Trend Duration')
        axes[1, 0].set_ylabel('TVS')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Price Positioning vs Continuation Likelihood
    if 'price_positioning' in corr_data.columns and 'continuation_likelihood' in corr_data.columns:
        axes[1, 1].scatter(corr_data['price_positioning'], corr_data['continuation_likelihood'], 
                          alpha=0.1, s=1)
        axes[1, 1].set_title('Price Positioning vs Continuation Likelihood')
        axes[1, 1].set_xlabel('Price Positioning')
        axes[1, 1].set_ylabel('Continuation Likelihood')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"data/reports/correlations_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Correlation plots saved to {plot_filename}")
    
    plt.close(fig)

def create_symbol_comparison_plots(df, top_n=10):
    """Create plots comparing top symbols"""
    
    print(f"Creating comparison plots for top {top_n} symbols...")
    
    # Get top symbols by count
    top_symbols = df['symbol'].value_counts().head(top_n).index.tolist()
    
    # Filter data for top symbols
    symbol_data = df[df['symbol'].isin(top_symbols)]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Top {top_n} Symbols Comparison', fontsize=16, fontweight='bold')
    
    # Direction comparison
    if 'direction' in symbol_data.columns:
        for symbol in top_symbols:
            symbol_subset = symbol_data[symbol_data['symbol'] == symbol]
            axes[0, 0].plot(symbol_subset['date'].dt.strftime('%Y-%m'), 
                           symbol_subset['direction'].rolling(window=30).mean(), 
                           label=symbol, alpha=0.7, linewidth=1)
        
        axes[0, 0].set_title('Direction Trends (30-day Moving Average)')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Direction')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # TVS comparison
    if 'trend_vulnerability_score' in symbol_data.columns:
        symbol_tvs = symbol_data.groupby('symbol')['trend_vulnerability_score'].mean()
        axes[0, 1].bar(symbol_tvs.index, symbol_tvs.values, alpha=0.7)
        axes[0, 1].set_title('Average Trend Vulnerability Score by Symbol')
        axes[0, 1].set_xlabel('Symbol')
        axes[0, 1].set_ylabel('Average TVS')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
    
    # Continuation Likelihood comparison
    if 'continuation_likelihood' in symbol_data.columns:
        symbol_cl = symbol_data.groupby('symbol')['continuation_likelihood'].mean()
        axes[1, 0].bar(symbol_cl.index, symbol_cl.values, alpha=0.7, color='green')
        axes[1, 0].set_title('Average Continuation Likelihood by Symbol')
        axes[1, 0].set_xlabel('Symbol')
        axes[1, 0].set_ylabel('Average Continuation Likelihood')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Record count by symbol
    symbol_counts = symbol_data['symbol'].value_counts()
    axes[1, 1].bar(symbol_counts.index, symbol_counts.values, alpha=0.7, color='orange')
    axes[1, 1].set_title('Record Count by Symbol')
    axes[1, 1].set_xlabel('Symbol')
    axes[1, 1].set_ylabel('Record Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"data/reports/symbol_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Symbol comparison plots saved to {plot_filename}")
    
    plt.close(fig)

def print_summary_stats(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"Total records: {len(df):,}")
    print(f"Unique symbols: {df['symbol'].nunique():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    print(f"\nTop 10 symbols by record count:")
    symbol_counts = df['symbol'].value_counts().head(10)
    for symbol, count in symbol_counts.items():
        print(f"  {symbol}: {count:,} records")
    
    print(f"\nMetric Statistics:")
    numeric_cols = ['direction', 'trend_duration', 'trend_vulnerability_score', 
                   'price_positioning', 'continuation_likelihood']
    
    for col in numeric_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.4f}")
            print(f"  Std:  {df[col].std():.4f}")
            print(f"  Min:  {df[col].min():.4f}")
            print(f"  Max:  {df[col].max():.4f}")

def main():
    """Main function"""
    print("📊 Large Scale Trend Analysis Consistency Plotter")
    print("=" * 60)
    
    # Ensure reports directory exists
    import os
    os.makedirs("data/reports", exist_ok=True)
    
    # Load the large CSV file
    df = load_large_csv()
    
    if df is None:
        return
    
    # Print summary statistics
    print_summary_stats(df)
    
    # Create all plots
    create_overall_consistency_plots(df)
    create_distribution_plots(df)
    create_correlation_plots(df)
    create_symbol_comparison_plots(df)
    
    print("\n✅ All consistency plots generated successfully!")
    print("📁 Check the data/reports/ directory for output files")

if __name__ == "__main__":
    main()