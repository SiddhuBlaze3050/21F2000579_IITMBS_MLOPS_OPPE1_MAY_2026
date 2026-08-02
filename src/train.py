import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import warnings
warnings.filterwarnings("ignore")

def train_and_tune():
    # 1. Connect to local MLflow server
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("stock_movement_tuning")

    print("Loading datasets...")
    train_df = pd.read_parquet("data/splits/train.parquet")
    val_df = pd.read_parquet("data/splits/val.parquet")
    
    features = ['rolling_avg_10', 'volume_sum_10']
    X_train, y_train = train_df[features], train_df['target']
    X_val, y_val = val_df[features], val_df['target']
    
    # 2. Define Hyperparameter Grid for the sweep
    param_grid = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 150, "max_depth": 15}
    ]
    
    best_f1 = -1.0
    best_run_id = None
    
    print("\nStarting Hyperparameter Sweep...")
    
    # 3. Execute the sweep
    for params in param_grid:
        with mlflow.start_run():
            print(f"Training with params: {params}")
            
            clf = RandomForestClassifier(**params, random_state=42)
            clf.fit(X_train, y_train)
            
            preds = clf.predict(X_val)
            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds)
            
            print(f"  -> Validation Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
            
            # Log params and metrics to MLflow
            mlflow.log_params(params)
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
            
            # Log the model artifact
            signature = infer_signature(X_train, preds)
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                signature=signature
            )
            
            # Keep track of the best run based on F1 Score
            if f1 > best_f1:
                best_f1 = f1
                best_run_id = mlflow.active_run().info.run_id

    print("-" * 40)
    print(f"Best F1 Score: {best_f1:.4f} (Run ID: {best_run_id})")
    print("-" * 40)
    
    # 4. Register the Best Model
    print("\nRegistering the best model to MLflow Model Registry...")
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="stock_movement_model")
    print("Success! Model registered as 'stock_movement_model'")

if __name__ == "__main__":
    train_and_tune()
