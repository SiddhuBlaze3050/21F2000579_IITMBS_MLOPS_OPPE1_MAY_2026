# OPPE AI Usage Documentation 

**Student**: Siddhartha Devulapalli \
**Roll Number**: 21F2000579\
**Exam Date**: August 2, 2026

---

## AI Tools Utilized and Conversation History
> List all GenAI / LLM tools used during the exam  
> Provide **public share links** to AI chats or attach conversation files if links are not available

- **Tool Name:** Gemini
    - **Purpose:** End-to-end assistance for OPPE-1 MLOps pipeline construction, including GCP infrastructure debugging, DVC versioning, Feast feature store setup, GitHub Actions CI testing, and MLflow tracking.
    - **Shared Chat Link:** https://share.gemini.google/pk0rmnnGvh6G
    - **Notes:** This single Gemini thread was the only AI tool utilized during the entire OPPE examination.

---

## Key Areas of AI Assistance

### Pipeline Architecture & Debugging
- **Feast Feature Store:** Resolved Dask hanging issues by converting timestamps to UTC to facilitate point-in-time joins.
- **MLflow Tracking:** Configured local SQLite backend and bypassed GCP Workbench middleware CORS restrictions for UI port-forwarding.
- **GitHub Actions (CI/CD):** Resolved Node 20 deprecation warnings, migrated from JSON keys to Workload Identity Federation (WIF) authentication, and dynamically patched MLflow SQLite absolute paths to function inside the CI runner environment.
- **Data Versioning:** Transitioned large MLflow artifacts (`mlruns/`) from Git to DVC tracking to bypass GitHub's 100 MB file limit.

---

## Prompts and Responses Used
> Include **all prompts** that contributed to solving the exam tasks
> Include **all responses** in case of public share links are not available to share

### Tool Name #1: Gemini

- **Prompt 1:** `My code is still stuck even after 2 minutes, I know that my data has timestamps from 2017 to 2021, should we give some point-in-time timestamp for easier fetching during this process?`
    - **Response Log:** [The AI identified that Feast's Dask engine hangs on timezone-aware datetimes and provided a script update to cast `event_timestamp` to `utc=True`.]

- **Prompt 2:** `I am getting the following error. I think the middleware issue I told where cors has to be allowed while server initiation has to be done here: Blocked cross-origin request from https://5000-cs...`
    - **Response Log:** [The AI provided the updated server startup command appending `--cors-allowed-origins "*"` to bypass strict security middleware.]

- **Prompt 3:** `remote: error: File mlruns/1/models/m-346d34f9f3dd45fb9ab387544897015e/artifacts/model.skops is 127.81 MB; this exceeds GitHub's file size limit of 100.00 MB... How do we approach this problem now?`
    - **Response Log:** [The AI provided git commands to soft reset the commit, unstage the MLflow files, restore `.gitignore`, and use `dvc add mlflow.db mlruns/` to track the artifacts via Google Cloud Storage instead.]

- **Prompt 4:** `Let's use WIF in github actions as I have already set it up in github secrets. Give me an updated ci.yaml for the above.`
    - **Response Log:** [The AI updated the GitHub Actions workflow to use `google-github-actions/auth@v2` with `workload_identity_provider` and added the critical `id-token: write` permission.]

- **Prompt 5:** `This is the error in github actions: mlflow.exceptions.MlflowException: No such artifact: ''`
    - **Response Log:** [The AI identified an absolute path discrepancy between the local SQLite database and the GitHub runner environment, providing a custom `patch_mlflow_db()` Python function to dynamically rewrite the URI paths before evaluation.]

---

## Files Generated with AI Assistance

### Fully AI-Generated (>90%)
1. `src/evaluate.py` - Model evaluation script with SQLite path-patching logic.
2. `tests/test_features.py` - Pytest suite explicitly validating engineered features against raw data bounds.

### Heavily AI-Assisted (50-90%)
1. `.github/workflows/ci.yml` - CI/CD pipeline integrated with DVC, WIF, and CML reporting[cite: 1].
2. `src/train.py` - Incremental training loop, chronological splitting, and MLflow hyperparameter sweep integration.
3. `src/split_data.py` - Chronological train/val/test data splitting logic.
4. `src/fetch_feast_data.py` - Point-in-time historical feature retrieval.

---

## Critical AI-Assisted Decisions

### ✅ Successful AI Recommendations

1. **Bypassing GitHub File Limits:** 
   - *Problem:* Random Forest model binaries exceeded the 100MB GitHub limit.
   - *AI Solution:* Stripped artifacts from Git tracking and pushed the local `mlruns/` and `mlflow.db` to GCS using DVC.
   - *Impact:* CI runner successfully retrieved the model registry via `dvc pull` without bloating the repository.

2. **Workload Identity Federation for CI/CD:**
   - *Problem:* Outdated GitHub action versions and secret evaluation errors during GCP authentication.
   - *AI Solution:* Upgraded to auth action v2, implemented `id-token: write` permissions, and formatted the WIF variables cleanly.
   - *Impact:* Secure, keyless access from GitHub Actions to Google Cloud Storage.

3. **Dynamic MLflow Path Patching:**
   - *Problem:* `evaluate.py` failed in GitHub actions because MLflow saved absolute paths from the local Workbench environment.
   - *AI Solution:* Created a SQL execution block to dynamically update the absolute paths to the CI runner's local workspace at runtime.
   - *Impact:* Allowed offline, SQLite-backed MLflow evaluation inside an isolated Ubuntu runner.

---