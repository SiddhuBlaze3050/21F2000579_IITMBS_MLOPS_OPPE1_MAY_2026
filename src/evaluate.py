import os
import sqlite3
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

def patch_mlflow_db():
    """Fixes absolute paths in the SQLite DB to match the CI runner's workspace."""
    db_path = 'mlflow.db'
    if not os.path.exists(db_path):
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the new base path for the current environment
    new_base = "file://" + os.path.abspath(os.getcwd())
    
    # Patch both the runs table and model_versions table
    for table, column in [("runs", "artifact_uri"), ("model_versions", "source")]:
        try:
            # Replaces the old absolute path with the new runner path
            query = f"""
                UPDATE {table} 
                SET {column} = ? || SUBSTR({column}, INSTR({column}, '/mlruns'))
                WHERE {column} LIKE '%/mlruns%'
            """
            cursor.execute(query, (new_base,))
        except Exception as e:
            print(f"Skipping {table}: {e}")
            
    conn.commit()
    conn.close()
    print("MLflow database paths successfully patched for CI environment.")

def evaluate():
    # 1. Patch the DB before interacting with MLflow
    patch_mlflow_db()
    
    # 2. Connect to the local SQLite DB tracked in Git
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # 3. Fetch Version 1 of the registered model
    print("Fetching best model from MLflow Registry...")
    model = mlflow.sklearn.load_model("models:/stock_movement_model/1")
    
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
