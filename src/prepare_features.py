import os
import glob
import pandas as pd
import numpy as np

def process_stock_file(file_path):
    stock_name = os.path.basename(file_path).split('__')[0]
    df = pd.read_csv(file_path)
    
    # Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['stock_name'] = stock_name
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    
    # Forward fill missing values
    df.ffill(inplace=True)
    
    # Feature 1: 10-minute rolling average of close price
    df['rolling_avg_10'] = df['close'].rolling(window='10min', min_periods=1).mean()
    
    # Feature 2: Total volume traded over last 10 minutes
    df['volume_sum_10'] = df['volume'].rolling(window='10min', min_periods=1).sum()
    
    # Drop NaNs in features
    df.dropna(subset=['rolling_avg_10', 'volume_sum_10'], inplace=True)
    
    # Target: 1 if close price 5 minutes later is strictly greater
    df['close_5min_future'] = df['close'].shift(-5)
    df['target'] = (df['close_5min_future'] > df['close']).astype(int)
    
    # Drop rows where target is NaN (last 5 rows of time series)
    df.drop(['close_5min_future'], axis=1, inplace=True)
    df_clean = df.dropna(subset=['target']).copy()
    
    # Reset index to make timestamp a column again for Feast
    df_clean.reset_index(inplace=True)
    return df_clean

def prepare_all_data(version_dir, output_parquet):
    csv_files = glob.glob(os.path.join(version_dir, "*.csv"))
    all_dfs = [process_stock_file(f) for f in csv_files]
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure Feast event timestamp formatting
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    
    # Save as Parquet file for Feast FileSource
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    merged_df.to_parquet(output_parquet, index=False)
    print(f"Saved processed features to {output_parquet} (Shape: {merged_df.shape})")

if __name__ == "__main__":
    prepare_all_data("data/v0", "feature_repo/data/stock_features_v0.parquet")
    
    # Create merged v0 + v1 dataset for Iteration 2
    import shutil
    os.makedirs("data/merged", exist_ok=True)
    for folder in ["data/v0", "data/v1"]:
        for file in glob.glob(f"{folder}/*.csv"):
            shutil.copy(file, "data/merged/")
            
    prepare_all_data("data/merged", "feature_repo/data/stock_features_v1.parquet")
