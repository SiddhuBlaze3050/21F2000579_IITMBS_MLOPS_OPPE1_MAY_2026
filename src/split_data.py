import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(input_parquet):
    print(f"Loading data from {input_parquet}...")
    df = pd.read_parquet(input_parquet)
    
    # Ensure data is sorted chronologically to avoid look-ahead bias
    df = df.sort_values('timestamp')
    
    # 80/10/10 Chronological Split (shuffle=False is critical here)
    train_df, temp_df = train_test_split(df, test_size=0.2, shuffle=False)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, shuffle=False)
    
    os.makedirs("data/splits", exist_ok=True)
    
    train_df.to_parquet("data/splits/train.parquet", index=False)
    val_df.to_parquet("data/splits/val.parquet", index=False)
    test_df.to_parquet("data/splits/test.parquet", index=False)
    
    print(f"Splits saved successfully!")
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_data.py <path_to_parquet>")
        sys.exit(1)
    create_splits(sys.argv[1])
