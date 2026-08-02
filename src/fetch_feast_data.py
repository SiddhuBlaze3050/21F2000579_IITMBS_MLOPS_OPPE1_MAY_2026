import pandas as pd
from feast import FeatureStore
import warnings
warnings.filterwarnings("ignore")

def fetch_training_data():
    store = FeatureStore(repo_path="feature_repo")
    
    # Load raw parquet data to get entity dataframe (timestamps and stock_names)
    entity_df = pd.read_parquet("feature_repo/data/stock_features_v0.parquet")[['timestamp', 'stock_name', 'target']].head(100)
    
    # EXPLICITLY rename to event_timestamp to speed up Feast's internal join
    entity_df = entity_df.rename(columns={"timestamp": "event_timestamp"})
    
    print("Fetching historical features from Feast (this may take 30-60 seconds)...")
    
    # Retrieve historical point-in-time features from Feast
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "stock_rolling_features:rolling_avg_10",
            "stock_rolling_features:volume_sum_10",
        ],
    ).to_df()
    
    print("\nSuccessfully retrieved point-in-time features from Feast!")
    print(training_df.head())
    return training_df

if __name__ == "__main__":
    fetch_training_data()
