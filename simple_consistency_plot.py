import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

def load_csv_files():
    """Load all CSV files from data/reports/"""
    csv_files = glob.glob("data/reports/*.csv")
    csv_files.sort()
    
    data = {}
    for csv_file in csv_files:
        filename = os.path.basename(csv_file).replace('.csv', '')
        df = pd.read_csv(csv_file)
        data[filename] = df
    
    print(f"Loaded {len(data)} CSV files")
    return data

def create_simple_plots(data):
    """Create line plots for all trend characteristics"""
    
    # Create figure with 5 subplots for all metrics
    fig, axes = plt.subplots(5, 1, figsize=(14, 20))
    fig.suptitle('Trend Analysis Consistency Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Direction (replaces trend_strength)
    for filename, df in data.items():
        if 'direction' in df.columns:
            axes[0].plot(df['direction'], label=filename, marker='o', markersize=2, alpha=0.8)
    
    axes[0].set_title('Direction (Trend Strength/Steepness)')
    axes[0].set_xlabel('Data Point Index')
    axes[0].set_ylabel('Direction (-1 to 1)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Trend Duration
    for filename, df in data.items():
        if 'trend_duration' in df.columns:
            axes[1].plot(df['trend_duration'], label=filename, marker='o', markersize=2, alpha=0.8)
    
    axes[1].set_title('Trend Duration')
    axes[1].set_xlabel('Data Point Index')
    axes[1].set_ylabel('Duration (days)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Trend Vulnerability Score
    for filename, df in data.items():
        if 'trend_vulnerability_score' in df.columns:
            axes[2].plot(df['trend_vulnerability_score'], label=filename, marker='o', markersize=2, alpha=0.8)
    
    axes[2].set_title('Trend Vulnerability Score (TVS)')
    axes[2].set_xlabel('Data Point Index')
    axes[2].set_ylabel('TVS (0 to 1)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Price Positioning
    for filename, df in data.items():
        if 'price_positioning' in df.columns:
            axes[3].plot(df['price_positioning'], label=filename, marker='o', markersize=2, alpha=0.8)
    
    axes[3].set_title('Price Positioning in Trend Corridor')
    axes[3].set_xlabel('Data Point Index')
    axes[3].set_ylabel('Position (0=trendline, 1=boundary)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[3].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 5: Continuation Likelihood
    for filename, df in data.items():
        if 'continuation_likelihood' in df.columns:
            axes[4].plot(df['continuation_likelihood'], label=filename, marker='o', markersize=2, alpha=0.8)
    
    axes[4].set_title('Continuation Likelihood')
    axes[4].set_xlabel('Data Point Index')
    axes[4].set_ylabel('Likelihood (0 to 1)')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"data/reports/trend_analysis_consistency_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Trend analysis consistency plots saved to {plot_filename}")
    
    plt.close(fig)

def create_summary_plot(data):
    """Create a summary plot with key metrics"""
    
    # Create figure with 2x2 subplot layout for key metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Key Trend Metrics Summary', fontsize=16, fontweight='bold')
    
    # Direction vs Continuation Likelihood scatter
    for filename, df in data.items():
        if 'direction' in df.columns and 'continuation_likelihood' in df.columns:
            ax1.scatter(df['direction'], df['continuation_likelihood'], 
                       label=filename, alpha=0.6, s=30)
    
    ax1.set_title('Direction vs Continuation Likelihood')
    ax1.set_xlabel('Direction (-1 to 1)')
    ax1.set_ylabel('Continuation Likelihood (0 to 1)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Trend Duration vs TVS scatter
    for filename, df in data.items():
        if 'trend_duration' in df.columns and 'trend_vulnerability_score' in df.columns:
            ax2.scatter(df['trend_duration'], df['trend_vulnerability_score'], 
                       label=filename, alpha=0.6, s=30)
    
    ax2.set_title('Trend Duration vs Vulnerability Score')
    ax2.set_xlabel('Trend Duration (days)')
    ax2.set_ylabel('TVS (0 to 1)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Price Positioning distribution
    for filename, df in data.items():
        if 'price_positioning' in df.columns:
            ax3.hist(df['price_positioning'], bins=20, alpha=0.6, label=filename)
    
    ax3.set_title('Price Positioning Distribution')
    ax3.set_xlabel('Price Position')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Direction distribution
    for filename, df in data.items():
        if 'direction' in df.columns:
            ax4.hist(df['direction'], bins=20, alpha=0.6, label=filename)
    
    ax4.set_title('Direction Distribution')
    ax4.set_xlabel('Direction')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"data/reports/trend_summary_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Trend summary plots saved to {plot_filename}")
    
    plt.close(fig)

def main():
    """Main function"""
    print("📊 Enhanced Trend Analysis Consistency Plot Generator")
    print("=" * 50)
    
    # Load CSV files
    data = load_csv_files()
    
    if len(data) < 1:
        print("❌ Need at least 1 CSV file")
        return
    
    # Create plots
    create_simple_plots(data)
    
    if len(data) >= 2:
        create_summary_plot(data)
    
    print("✅ Done!")

if __name__ == "__main__":
    main() 