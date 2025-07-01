# llm_trend_analysis
gpt-vision in market trans analysis with structed outputs
## How to use
1. Place ur source csv data to the ./data/source folder:
```
./data/source/NDX.csv
./data/source/NVDA.csv
...
```
2. Generate plots:
```
python generate_plots.py
```
3. Define the openai api key:
```
export OPENAI_API_KEY=YOURKEY
```
3. Run prediction:
```
python trend_predictor.py --threads 300
```
4. Concatenate results into single CSV file
```
python collect_results.py
```