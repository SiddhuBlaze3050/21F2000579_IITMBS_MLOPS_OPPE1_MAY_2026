import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

def evaluate():
    # Connect to the local SQLite DB tracked in Git
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Fetch Version 1 of the registered model
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
