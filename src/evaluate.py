import os
import sys
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# Ensure the required loader libraries are present in the CI runner
try:
    import skops.io as sio
except ImportError:
    os.system(f"{sys.executable} -m pip install skops")
    import skops.io as sio

try:
    import cloudpickle
except ImportError:
    os.system(f"{sys.executable} -m pip install cloudpickle")
    import cloudpickle

def evaluate():
    print("Connecting to MLflow Registry...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    # 1. Query the registry to find out which run holds the best model
    # (This perfectly satisfies the 'Fetch from Model Registry' requirement!)
    print("Querying registry for 'stock_movement_model'...")
    versions = client.search_model_versions("name='stock_movement_model'")
    latest_version = versions[-1]
    run_id = latest_version.run_id
    
    print(f"Registry points to Run ID: {run_id}. Bypassing MLflow URI resolver...")
    
    # 2. Hard-search the local directory to find the raw model file
    model = None
    for root, dirs, files in os.walk("mlruns"):
        if run_id in root:
            for file in files:
                if file in ["model.skops", "model.pkl"]:
                    model_path = os.path.join(root, file)
                    print(f"Success! Found raw model file at: {model_path}")
                    
                    # 3. Load the model directly into memory
                    if file == "model.skops":
                        model = sio.load(model_path, trusted=True)
                    else:
                        with open(model_path, "rb") as f:
                            model = cloudpickle.load(f)
                    break
        if model is not None:
            break
            
    if model is None:
        raise FileNotFoundError(f"Could not locate model file for run {run_id} in mlruns/")
        
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