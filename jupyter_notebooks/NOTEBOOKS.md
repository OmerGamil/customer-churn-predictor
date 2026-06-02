# Jupyter Notebooks — Run Order

Run the notebooks **in sequence**. Each notebook depends on outputs from the previous one.

## Before Running

Set the working directory inside each notebook. Change the `os.chdir()` call at the top of each notebook to match your local project root path, or use the provided Gitpod/CI workspace where the path is already correct.

---

## Execution Order

### 01_DataCollection.ipynb
**Input**: `outputs/datasets/collection/WA_Fn-UseC_-Telco-Customer-Churn.csv`
**Output**: `outputs/datasets/collection/telco_churn_raw.csv`

Loads the original Telco dataset, inspects its structure, and saves a raw baseline copy.

---

### 02_DataCleaning.ipynb
**Input**: `outputs/datasets/collection/telco_churn_raw.csv`
**Output**: `outputs/datasets/collection/telco_churn_cleaned.csv`

Fixes the `TotalCharges` blank-string issue, drops `customerID`, and label-encodes the Churn target.

---

### 03_EDA.ipynb
**Input**: `outputs/datasets/collection/telco_churn_cleaned.csv`
**Output**: PNG plots in `outputs/eda/`

Produces correlation analysis and feature-vs-churn visualisations. Run this before launching the Streamlit app so the EDA page has images to display.

---

### 04_ModellingEvaluation.ipynb
**Input**: `outputs/datasets/collection/telco_churn_cleaned.csv`
**Outputs**:
- `outputs/ml_pipeline/predict_churn/v1/clf_pipeline.pkl`
- `outputs/ml_pipeline/predict_churn/v1/confusion_matrix.png`
- `outputs/ml_pipeline/predict_churn/v1/classification_report.csv`

Trains and tunes the ML pipeline, evaluates on the test set, and saves the fitted pipeline. Run this before using the Churn Predictor page in the dashboard.
