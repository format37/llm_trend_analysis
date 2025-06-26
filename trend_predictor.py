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

def predict(system_prompt, user_prompt, image_path):
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano-2025-04-14")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    messages=[
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

    kwargs = dict(
            model=model,
            messages=messages,
        )
    if model != 'o3':
        kwargs['temperature'] = 0.0
    kwargs["response_format"] = TrendAnalysis
    completion = client.beta.chat.completions.parse(**kwargs)    
    response = completion.choices[0].message
    response_object = response.parsed
    response_json = json.loads(response.content)
    return response_object, response_json, completion

def batch_process_images():
    """Process all PNG files from data and data_ directories and save results to CSV"""
    
    # Get all PNG files from both directories
    png_files = []
    # png_files.extend(glob.glob("data/*.png"))
    png_files.extend(glob.glob("data/plots/*.png"))
    # png_files.extend(glob.glob("data_/*.png"))
    
    # Remove duplicates (in case same files exist in multiple directories)
    png_files = list(set(png_files))
    png_files.sort()
    
    print(f"Found {len(png_files)} PNG files to process")
    
    # Initialize results list
    results = []
    
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

    3. **TREND VULNERABILITY SCORE (TVS)** — Estimate the fraction of the trend’s duration since its start that would be negated by a single 2-standard-deviation move against the trend, based on realized volatility inside the recognized trend channel. Values close to and above 1 indicate fragile trends; values near 0 indicate robust trends. Score saturates at 1.

    4. **PRICE POSITIONING** — Assess the current price’s location within the identified trend corridor:
       -  0 = on the trendline (support in an uptrend, resistance in a downtrend)
       - <0 = price has broken against the trend (e.g., below support in an uptrend)
       -  1 = at the upper (uptrend) or lower (downtrend) boundary of the channel
       - >1 = price has broken out beyond the channel in the direction of the trend

    5. **CONTINUATION LIKELIHOOD** — Estimate the probability (0 to 1) that the trend will continue in the same direction versus reversing or consolidating in the short term.
    """
    
    # Process each image
    for i, image_path in tqdm.tqdm(enumerate(png_files), total=len(png_files)):
        try:
            # Extract date from filename (assuming format like 20250118_60.png)
            filename = os.path.basename(image_path)
            date_str = filename.split('_')[0]  # Extract date part
            
            # Run trend analysis
            trend_analysis, response_json, completion = predict(system_prompt, user_prompt, image_path)
            
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
            results.append(result)
            
            # print(f"  - Direction: {trend_analysis.direction:.3f}")
            # print(f"  - Trend duration: {trend_analysis.trend_duration} days")
            # print(f"  - TVS: {trend_analysis.trend_vulnerability_score:.3f}")
            # print(f"  - Price positioning: {trend_analysis.price_positioning:.3f}")
            # print(f"  - Continuation likelihood: {trend_analysis.continuation_likelihood:.3f}")
            
        except Exception as e:
            print(f"  - Error processing {image_path}: {str(e)}")
            # Store error result
            result = {
                'filename': filename,
                'filepath': image_path,
                'date': date_str if 'date_str' in locals() else 'unknown',
                'direction': None,
                'trend_duration': None,
                'trend_vulnerability_score': None,
                'price_positioning': None,
                'continuation_likelihood': None,
                'processed_at': datetime.now().isoformat(),
                'error': str(e)
            }
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/reports/{timestamp}.csv"
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"Processed {len(results)} files")
    
    # Display summary statistics
    if len(df) > 0:
        print("\nSummary Statistics:")
        print(f"Average direction: {df['direction'].mean():.3f}")
        print(f"Average trend duration: {df['trend_duration'].mean():.1f} days")
        print(f"Average TVS: {df['trend_vulnerability_score'].mean():.3f}")
        print(f"Average price positioning: {df['price_positioning'].mean():.3f}")
        print(f"Average continuation likelihood: {df['continuation_likelihood'].mean():.3f}")
        print(f"Successful predictions: {df['direction'].notna().sum()}")
        print(f"Failed predictions: {df['direction'].isna().sum()}")
    
    return df, output_path

def main():
    # Process all images in batch
    df, output_path = batch_process_images()
    return df, output_path

if __name__ == "__main__":
    main()
