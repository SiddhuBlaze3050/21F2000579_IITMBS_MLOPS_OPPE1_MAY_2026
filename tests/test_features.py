import pandas as pd
import glob
import pytest
import os

@pytest.fixture
def data_samples():
    # Load a raw CSV
    raw_files = glob.glob("data/v0/*.csv")
    if not raw_files:
        pytest.skip("Raw data not found in data/v0/")
        
    raw_path = raw_files[0]
    stock_name = os.path.basename(raw_path).split('__')[0]
    raw_df = pd.read_csv(raw_path)
    
    # Load the processed test split
    test_df = pd.read_parquet("data/splits/test.parquet")
    
    return raw_df, test_df, stock_name

def test_volume_sum_logic_against_raw(data_samples):
    raw_df, test_df, stock_name = data_samples
    
    # Filter test data for the specific stock
    stock_test_df = test_df[test_df['stock_name'] == stock_name]
    
    if not stock_test_df.empty:
        # The processed rolling volume sum should never exceed the total raw volume for that stock
        total_raw_volume = raw_df['volume'].sum()
        max_rolling_volume = stock_test_df['volume_sum_10'].max()
        
        assert max_rolling_volume <= total_raw_volume, \
            "Sanity check failed: 10-min rolling volume exceeds total historical volume."

def test_rolling_avg_logic_against_raw(data_samples):
    raw_df, test_df, stock_name = data_samples
    
    stock_test_df = test_df[test_df['stock_name'] == stock_name]
    
    if not stock_test_df.empty:
        # The rolling average close price should stay within the absolute min/max of the raw close prices
        raw_min_close = raw_df['close'].min()
        raw_max_close = raw_df['close'].max()
        
        feature_min_avg = stock_test_df['rolling_avg_10'].min()
        feature_max_avg = stock_test_df['rolling_avg_10'].max()
        
        assert feature_min_avg >= raw_min_close, "Sanity check failed: Rolling average dipped below raw minimum."
        assert feature_max_avg <= raw_max_close, "Sanity check failed: Rolling average exceeded raw maximum."
