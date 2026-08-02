import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

def evaluate():
    print("Connecting to MLflow Registry...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    # 1. Query the registry to find out which run holds the best model
    print("Querying registry for 'stock_movement_model'...")
    versions = client.search_model_versions("name='stock_movement_model'")
    
    # Get the latest registered version
    latest_version = versions[-1]
    run_id = latest_version.run_id
    experiment_id = client.get_run(run_id).info.experiment_id
    
    # 2. Construct the direct relative path (Bypassing the broken DB absolute paths!)
    local_model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
    print(f"Loading model directly from relative path: {local_model_path}")
    
    # 3. Load the model
    model = mlflow.sklearn.load_model(local_model_path)
    
    print("Loading test data...")
    test_df = pd.read_parquet("data/splits/test.parquet")
    features = ['rolling_avg_10', 'volume_sum_10']
    X_test, y_test = test_df[features], test_df['target']
    
    print("Running predictions...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    # Save metrics for CML
    with open("metrics.txt", "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")
        
    # Generate Confusion Matrix plot for CML
    disp = ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.savefig("confusion_matrix.png")
    print("Evaluation complete. Metrics and plots generated.")

if __name__ == "__main__":
    evaluate()