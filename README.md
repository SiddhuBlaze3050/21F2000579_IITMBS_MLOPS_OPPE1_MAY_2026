# Stock Movement Predictor - MLOps Pipeline

This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline for predicting stock movements using historical, minute-level financial data. The project was built iteratively to demonstrate production-grade MLOps practices, including data versioning, feature store management, experiment tracking, and automated CI/CD workflows.

---

## 🏗️ Project Architecture & Deliverables

### Deliverable 1: Version Control & Environment Setup
* **Repository:** Managed via GitHub with strict access controls.
* **Environment:** Python 3.12 managed via Micromamba on a Google Cloud Platform (GCP) Workbench instance.
* **Structure:** Code is modularized into `src/` (scripts), `tests/` (testing), `data/` (data storage), and `feature_repo/` (feature store configs).

### Deliverable 2: Data Versioning with DVC
To manage large datasets without bloating the Git repository, Data Version Control (DVC) was implemented.
* **Remote Storage:** Google Cloud Storage (GCS) is configured as the remote DVC backend.
* **Tracking:** Raw datasets (`v0` and `v1` parquets) are tracked via `.dvc` files, ensuring deterministic reproducibility of the data state for any given commit.

### Deliverable 3: Feature Store Implementation (Feast)
To guarantee consistency between training and serving, and to prevent time-series data leakage, we integrated Feast.
* **Offline/Online Store:** Configured a local SQLite offline/online store.
* **Feature Views:** Defined `stock_rolling_features` bound to a `stock_name` entity.
* **Point-in-Time Joins:** Utilized Feast's historical retrieval to safely join rolling averages and volume sums to specific event timestamps, automatically resolving timezones to UTC to prevent Dask engine hangs.

### Deliverable 4: Chronological Splitting & Incremental Training
Because this is time-series data, a strict chronological split (80% Train, 10% Validation, 10% Test) was implemented to prevent look-ahead bias.
* **Iteration 1:** Trained a baseline Random Forest Classifier exclusively on `v0` data.
* **Iteration 2:** Merged `v0` and `v1` data, re-split chronologically, and retrained the model to observe metric shifts as new data becomes available.
* All data splits (`train.parquet`, `val.parquet`, `test.parquet`) are tracked by DVC.

### Deliverable 5: Experiment Tracking & Model Registry (MLflow)
A local MLflow tracking server backed by SQLite (`mlflow.db`) was implemented to track hyperparameter tuning.
* **Hyperparameter Sweep:** Evaluated multiple configurations of `n_estimators` and `max_depth`.
* **Artifact Logging:** Logged parameters, Accuracy, and F1 Scores for every run.
* **Model Registry:** The best-performing model (based on validation F1 score) was programmatically promoted and registered under the name `stock_movement_model`.
* *Note: The local `mlruns/` directory and `mlflow.db` are tracked via DVC to bypass GitHub file size limits while allowing CI access.*

### Deliverable 6: Continuous Integration & Automated Reporting
A robust GitHub Actions pipeline (`ci.yml`) is triggered on every push and pull request to the `main` branch.
* **Keyless Authentication:** Utilizes GCP Workload Identity Federation (WIF) for secure, keyless access to the GCS bucket.
* **Artifact Retrieval:** Executes `dvc pull` to download the versioned test data and the MLflow SQLite registry.
* **Sanity Testing (Pytest):** Runs automated tests validating that engineered features in the test set logically align with the boundaries of the raw CSV datasets (e.g., ensuring rolling volumes do not exceed total raw volumes).
* **Model Evaluation:** Fetches the registered `stock_movement_model` from MLflow, generates predictions on the held-out test set, and calculates final metrics.
* **CML Reporting:** Uses Continuous Machine Learning (CML) to automatically comment on the PR with the performance metrics and a generated Confusion Matrix plot.

---

## 🚀 How to Run Locally

**1. Pull Data and Models:**
```bash
dvc pull

```

**2. Setup MLflow Tracking Server:**

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 \
    --cors-allowed-origins "*"

```

**3. Run Feature Store Materialization:**

```bash
cd feature_repo
feast apply
feast materialize 2017-01-01T00:00:00 2026-01-01T00:00:00
cd ..

```

**4. Execute Pipeline:**

```bash
# Generate Splits
python src/split_data.py feature_repo/data/stock_features_v1.parquet

# Run Training & MLflow Sweep
python src/train.py

# Run Evaluation
python src/evaluate.py

```

**5. Run Tests:**

```bash
PYTHONPATH=. pytest tests/ -v

```

Final CML Report: https://github.com/SiddhuBlaze3050/21F2000579_IITMBS_MLOPS_OPPE1_MAY_2026/commit/1f8751820cd0e74962e7020c464d6bb88b17a3a7#commitcomment-194677712