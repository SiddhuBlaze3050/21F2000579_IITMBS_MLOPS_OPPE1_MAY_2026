import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_and_evaluate():
    print("Loading datasets...")
    train_df = pd.read_parquet("data/splits/train.parquet")
    val_df = pd.read_parquet("data/splits/val.parquet")
    
    features = ['rolling_avg_10', 'volume_sum_10']
    X_train, y_train = train_df[features], train_df['target']
    X_val, y_val = val_df[features], val_df['target']
    
    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating on Validation Set...")
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    
    print("-" * 30)
    print(f"Validation Accuracy : {acc:.4f}")
    print(f"Validation F1 Score : {f1:.4f}")
    print("-" * 30)
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.joblib"
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
