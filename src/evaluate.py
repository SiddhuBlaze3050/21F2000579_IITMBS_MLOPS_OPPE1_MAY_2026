import os
import sqlite3
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

def patch_mlflow_db():
    """Converts absolute DB paths to universal relative paths (./mlruns/...)."""
    db_path = 'mlflow.db'
    if not os.path.exists(db_path):
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    tables_and_columns = [
        ("experiments", "artifact_location"),
        ("runs", "artifact_uri"),
        ("model_versions", "source")
    ]
    
    for table, column in tables_and_columns:
        try:
            cursor.execute(f"SELECT rowid, {column} FROM {table} WHERE {column} IS NOT NULL")
            rows = cursor.fetchall()
            for rowid, old_uri in rows:
                if 'mlruns' in old_uri:
                    # Extract everything from 'mlruns' onward and make it relative
                    idx = old_uri.find('mlruns')
                    new_uri = './' + old_uri[idx:]
                    cursor.execute(f"UPDATE {table} SET {column} = ? WHERE rowid = ?", (new_uri, rowid))
        except Exception:
            pass
            
    conn.commit()
    conn.close()

def evaluate():
    patch_mlflow_db()
    
    # Connect to the patched DB
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Fetch Version 1 of the registered model by name
    model = mlflow.sklearn.load_model("models:/stock_movement_model/1")
    
    test_df = pd.read_parquet("data/splits/test.parquet")
    features = ['rolling_avg_10', 'volume_sum_10']
    X_test, y_test = test_df[features], test_df['target']
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    with open("metrics.txt", "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")
        
    disp = ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    evaluate()