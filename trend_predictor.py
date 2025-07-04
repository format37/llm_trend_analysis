import base64
from openai import OpenAI
import json
from pydantic import BaseModel, Field
from typing import Literal, List
import pandas as pd
import glob
import os
from datetime import datetime
import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import logging
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx INFO messages (HTTP request logs)
logging.getLogger("httpx").setLevel(logging.WARNING)

class TokenCounter:
    """Track token usage over time with rate limiting awareness"""
    def __init__(self, tpm_limit=150000000):  # 150M tokens per minute default
        self.tpm_limit = tpm_limit
        self.tokens_history = []  # List of (timestamp, token_count) tuples
        self.lock = threading.Lock()
    
    def add_tokens(self, tokens):
        """Add tokens used in a request"""
        with self.lock:
            current_time = time.time()
            self.tokens_history.append((current_time, tokens))
            # Clean up old entries (older than 1 minute)
            cutoff_time = current_time - 60
            self.tokens_history = [(t, c) for t, c in self.tokens_history if t > cutoff_time]
    
    def get_last_minute_usage(self):
        """Get token usage in the last minute"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - 60
            recent_tokens = [count for timestamp, count in self.tokens_history if timestamp > cutoff_time]
            total_tokens = sum(recent_tokens)
            percentage = (total_tokens / self.tpm_limit) * 100 if self.tpm_limit > 0 else 0
            return total_tokens, percentage

def append_message(messages, role, text, image_url):
    messages.append(
        {
            "role": role,
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
            ],
        }
    )

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class TrendAnalysis(BaseModel):
    direction: float = Field(
        ..., 
        ge=-1, 
        le=1, 
        description="Direction strength/steepness from -1 (clear steep downtrend) to 1 (clear steep uptrend), ~0 for horizontal/directionless movement or poor trend pattern"
    )
    trend_duration: int = Field(
        ...,
        ge=0,
        description="Trend duration in days - how long the current trend has persisted before now"
    )
    trend_vulnerability_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="TVS: Proportion of trend duration that would be undone by a typical 2-standard-deviation adverse price move based on realized volatility inside of the currently recognized trend. Higher values (closer to 1) indicate more vulnerable trends."
    )
    price_positioning: float = Field(
        ...,
        ge=-1,
        le=2,
        description="Price position within trend corridor: 0=touching the trendline (lower for up trend and upper for down trend), <0=breakout against trend, 1=touching resistance trendline (upper for up trend and lower for down trend), >1=broke out of the resistance - support corridor into the direction of the trend"
    )
    continuation_likelihood: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Estimated short term likelihood (0 to 1) of trend continuation with same characteristics vs stagnation or reversal"
    )

def predict(system_prompt, user_prompt, image_path, token_counter=None, api_key=None, client=None):
    """
    Predict using OpenAI API with optional pre-created client
    """
    if client is None:
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano-2025-04-14")
        client = OpenAI(api_key=api_key, timeout=600)
    else:
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano-2025-04-14")
    
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    base64_image = encode_image(image_path)
    image_url = f"data:image/jpeg;base64,{base64_image}"    
    append_message(
        messages, 
        "user",
        user_prompt,
        image_url
    )

    # Build kwargs properly with correct types
    kwargs = {
        "model": model,
        "messages": messages,
        "response_format": TrendAnalysis
    }
    
    # Only add temperature for non-o3 models
    if model != 'o3':
        kwargs['temperature'] = 0.0
    
    completion = client.beta.chat.completions.parse(**kwargs)    
    response = completion.choices[0].message
    response_object = response.parsed
    
    # Safe JSON parsing
    response_json = None
    if response.content:
        response_json = json.loads(response.content)
    
    # Add token counting
    token_info = None
    if token_counter is not None and hasattr(completion, 'usage') and completion.usage:
        token_counter.add_tokens(completion.usage.total_tokens)
        total, percent = token_counter.get_last_minute_usage()
        token_info = (total, percent)
    
    return response_object, response_json, completion, token_info

def get_result_path(image_path):
    """Get the path where individual result should be saved"""
    # Extract symbol from path like "data/plots/SNPS/20250513_60.png"
    path_parts = image_path.split(os.sep)
    symbol = path_parts[-2]  # Get the symbol (second to last part)
    
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    return f"data/results/{symbol}/{name_without_ext}.json"

# Thread-local storage for OpenAI clients
_thread_local = threading.local()

def get_thread_client(api_key):
    """Get or create thread-local OpenAI client"""
    if not hasattr(_thread_local, 'openai_client'):
        _thread_local.openai_client = OpenAI(api_key=api_key, timeout=600)
    return _thread_local.openai_client

def process_image_with_progress_threaded(args):
    """
    Process a single image with thread-local client creation
    Args:
        args: Tuple containing (image_path, system_prompt, user_prompt, pbar, token_counter, api_key, existing_counter)
    Returns:
        bool: True if processed or skipped (existing), False if error
    """
    image_path, system_prompt, user_prompt, pbar, token_counter, api_key, existing_counter = args
    
    # Get thread-local client (one per worker thread, not per image)
    client = get_thread_client(api_key)
    
    try:
        # Check if result already exists
        result_path = get_result_path(image_path)
        if os.path.exists(result_path):
            with existing_counter['lock']:
                existing_counter['count'] += 1
                if existing_counter['count'] % 50 == 0:  # Print every 50 existing files
                    logger.info(f"Found {existing_counter['count']} existing results so far...")
            pbar.update(1)
            return True
        
        # Extract date from filename (assuming format like 20250118_60.png)
        filename = os.path.basename(image_path)
        date_str = filename.split('_')[0]  # Extract date part
        
        # Run trend analysis with thread-local client
        trend_analysis, response_json, completion, token_info = predict(system_prompt, user_prompt, image_path, token_counter, client=client)
        
        # Check if trend analysis was successful
        if trend_analysis is None:
            logger.error(f"✗ Failed to get trend analysis for {image_path}")
            pbar.update(1)
            return False
        
        # Store results
        result = {
            'filename': filename,
            'filepath': image_path,
            'date': date_str,
            'direction': trend_analysis.direction,
            'trend_duration': trend_analysis.trend_duration,
            'trend_vulnerability_score': trend_analysis.trend_vulnerability_score,
            'price_positioning': trend_analysis.price_positioning,
            'continuation_likelihood': trend_analysis.continuation_likelihood,
            'processed_at': datetime.now().isoformat()
        }
        
        # Save individual result to JSON file
        os.makedirs(os.path.dirname(result_path), exist_ok=True)  # Create symbol directory
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Update progress bar with last saved file and token usage
        postfix_parts = [f"✓ {os.path.basename(result_path)}"]
        if token_info:
            total, percent = token_info
            postfix_parts.append(f"Tokens: {total} ({percent:.1f}%)")
        pbar.set_postfix_str(" | ".join(postfix_parts))
        pbar.update(1)
        return True
        
    except Exception as e:
        logger.error(f"✗ Error processing {image_path}: {str(e)}")
        pbar.update(1)
        return False

def batch_process_images(threads=1):
    """Process all PNG files from data and data_ directories and save individual results to JSON files"""
    
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")

    # Get all PNG files from both directories
    png_files = []
    # png_files.extend(glob.glob("data/*.png"))
    png_files.extend(glob.glob("data/plots/**/*.png", recursive=True))
    # png_files.extend(glob.glob("data_/*.png"))
    
    # Remove duplicates (in case same files exist in multiple directories)
    png_files = list(set(png_files))
    png_files.sort()
    
    print(f"Found {len(png_files)} PNG files to process")
    
    # Create directory for results
    os.makedirs("data/results", exist_ok=True)
    
    # Count existing and pending files
    existing_count = 0
    pending_files = []
    
    for image_path in png_files:
        result_path = get_result_path(image_path)
        if os.path.exists(result_path):
            existing_count += 1
        else:
            pending_files.append(image_path)
    
    print(f"Found {existing_count} existing results, {len(pending_files)} files to process")
    print(f"Overall progress: {existing_count}/{len(png_files)} ({existing_count/len(png_files)*100:.1f}%) already complete")
    
    if len(pending_files) == 0:
        print("No files to process!")
        return 0, 0

    system_prompt = """
    You are a quantitative finance analyst with deep expertise in technical analysis, pattern recognition, and market structure. Your primary task is to analyze candlestick charts using visual features such as price action, volatility, and support/resistance behavior.

    You operate on raw OHLCV data and extract meaningful trend structures. Your analysis assumes a fixed-length price chart is provided, and from that window, you must identify the most prominent recent trend—either still ongoing or just recently broken. You focus on trends that are at least 15 candles in length (minimum), with a preference for longer and more statistically stable formations.
    """

    user_prompt = """
    You are given a candlestick chart showing OHLCV data with a fixed recent lookback period. Analyze the chart and identify the most coherent and prominent trend segment (minimum 15 candles long), whether it is still active or just recently violated.

    For the trend you identify, output a comprehensive analysis including the following characteristics:

    1. **DIRECTION** — Estimate the overall trend direction on a scale from -1 to 1:
       - -1 = strong downtrend (steep and consistent decline)
       -  0 = sideways or directionless price action (no sustained movement)
       - +1 = strong uptrend (steep and consistent rise)

    2. **TREND DURATION** — Count how many candles the identified trend lasted up to the final candle in the chart.

    3. **TREND VULNERABILITY SCORE (TVS)** — Estimate the fraction of the trend's duration since its start that would be negated by a single 2-standard-deviation move against the trend, based on realized volatility inside the recognized trend channel. Values close to and above 1 indicate fragile trends; values near 0 indicate robust trends. Score saturates at 1.

    4. **PRICE POSITIONING** — Assess the current price's location within the identified trend corridor:
       -  0 = on the trendline (support in an uptrend, resistance in a downtrend)
       - <0 = price has broken against the trend (e.g., below support in an uptrend)
       -  1 = at the upper (uptrend) or lower (downtrend) boundary of the channel
       - >1 = price has broken out beyond the channel in the direction of the trend

    5. **CONTINUATION LIKELIHOOD** — Estimate the probability (0 to 1) that the trend will continue in the same direction versus reversing or consolidating in the short term.
    """
    
    # Initialize token counter
    token_counter = TokenCounter()
    
    # Create progress bar for ALL files (not just pending), starting from existing count
    pbar = tqdm.tqdm(total=len(png_files), desc="Processing images", position=0, leave=True, initial=existing_count)
    
    # Counter for tracking existing files found during processing
    existing_counter = {'count': 0, 'lock': threading.Lock()}
    
    # Process images
    new_processed_count = 0  # Only newly processed files
    failed_count = 0
    
    if threads <= 1:
        # Sequential processing - use thread-local client approach for consistency
        for image_path in pending_files:
            args = (image_path, system_prompt, user_prompt, pbar, token_counter, api_key, existing_counter)
            success = process_image_with_progress_threaded(args)
            if success:
                new_processed_count += 1
            else:
                failed_count += 1
    else:
        # Parallel processing - thread-local clients will be created automatically
        args_list = [(image_path, system_prompt, user_prompt, pbar, token_counter, api_key, existing_counter) for image_path in pending_files]
        with ThreadPoolExecutor(max_workers=threads) as executor:
            results = list(executor.map(process_image_with_progress_threaded, args_list))
            new_processed_count = sum(1 for r in results if r)
            failed_count = sum(1 for r in results if not r)
    
    pbar.close()
    
    # Print final summary
    total_files = len(png_files)
    total_existing = existing_count + existing_counter['count']  # Pre-existing + found during processing
    total_processed = new_processed_count
    total_failed = failed_count
    
    print(f"\n=== Processing Complete ===")
    print(f"Total files found: {total_files}")
    print(f"Previously completed: {total_existing}")
    print(f"Newly processed: {total_processed}")
    print(f"Failed: {total_failed}")
    print(f"Total completed: {total_existing + total_processed}")
    
    if total_processed > 0:
        print(f"\nNewly processed results saved to symbol-specific JSON files in: data/results/{{symbol}}/")
        print(f"Use a separate script to collect all results into a single CSV file.")
        print(f"\n✨ Performance improvement: Each worker thread now uses its own OpenAI client instance")
        print(f"   to avoid bottlenecks and improve throughput with {threads} threads.")
    
    return total_processed, total_failed

def main(threads=1):
    # Process all images in batch
    processed, failed = batch_process_images(threads=threads)
    return processed, failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trend Predictor - Batch Image Processing')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads for parallel processing')
    args = parser.parse_args()
    main(threads=args.threads)
