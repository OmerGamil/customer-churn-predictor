# Customer Churn Predictor

A machine learning web application that predicts whether a telecom customer will churn, built as part of the Code Institute Diploma in Full Stack Software Development (Predictive Analytics).

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and Validation](#hypothesis-and-validation)
4. [Rationale to Map Business Requirements to Data Visualisations and ML Tasks](#rationale-to-map-business-requirements-to-data-visualisations-and-ml-tasks)
5. [ML Business Case](#ml-business-case)
6. [Dashboard Design](#dashboard-design)
7. [How to Run the Project](#how-to-run-the-project)
8. [Deployment](#deployment)
9. [Known Issues and Fixes](#known-issues-and-fixes)
10. [Main Libraries Used](#main-libraries-used)
11. [Credits](#credits)

---

## Dataset Content

The dataset is sourced from IBM's **Telco Customer Churn** dataset, available on Kaggle. It contains **7,043 records** and **20 features** describing:

- **Customer demographics**: gender, SeniorCitizen, Partner, Dependents
- **Service subscriptions**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account information**: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Target variable**: Churn (Yes = customer left in the last month, No = still active)

---

## Business Requirements

**Business Requirement 1**
> The client wants to understand which customer attributes most strongly correlate with churn, so that the retention team can identify and focus on at-risk customer segments.

**Business Requirement 2**
> The client wants to predict whether a given customer will churn, enabling proactive intervention before a customer leaves. The model must achieve **≥75% recall on the Churn class** to be considered successful.

---

## Hypothesis and Validation

| # | Hypothesis | Validation |
|---|-----------|-----------|
| 1 | Customers with **month-to-month contracts** are more likely to churn than those on annual contracts | ✅ Confirmed via EDA — month-to-month customers churn at ~43% vs ~3% for two-year contracts |
| 2 | Customers with **higher monthly charges** are more likely to churn | ✅ Confirmed via correlation analysis — MonthlyCharges shows positive correlation (+0.19) with Churn |
| 3 | Customers with **lower tenure** are more likely to churn | ✅ Confirmed via EDA — churned customers have significantly lower average tenure |

---

## Rationale to Map Business Requirements to Data Visualisations and ML Tasks

### Business Requirement 1 — Correlation Study

**User Story**: As a data analyst at the telecom company, I want to see which customer attributes are most correlated with churn, so that I can brief the retention team on which customer segments to prioritise.

- An EDA was conducted examining the relationship between each feature and the Churn target variable.
- A correlation heatmap was produced for numeric features.
- Bar charts and distribution plots were created for the top correlated features.
- All plots are displayed on the **EDA & Correlation Study** dashboard page with written interpretations.

### Business Requirement 2 — ML Prediction

**User Story**: As a customer success manager, I want to enter a customer's details and receive a churn prediction, so that I can proactively reach out to at-risk customers before they cancel.

- A binary classification ML pipeline was built using RandomForestClassifier.
- The pipeline was evaluated against the 75% recall target.
- The trained pipeline is deployed in the **Churn Predictor** dashboard page, where staff can input customer attributes and receive an instant prediction.

---

## ML Business Case

**Aim**: Develop a binary classification model that predicts customer churn (Yes/No) from customer account and service attributes.

**Learning Method**: Supervised learning — binary classification using RandomForestClassifier.

**Ideal Outcome**: A model that correctly identifies at least 75% of customers who will churn (Recall ≥ 0.75 on the Churn class), allowing the retention team to act before customers leave.

**Success Metrics**:
- Primary: Recall ≥ 75% on the Churn class (label = 1)
- Secondary: Overall accuracy and F1-score on the held-out test set

**Model Output**: A binary prediction — "likely to churn" (1) or "not likely to churn" (0) — for a single customer record entered via the Streamlit interface.

**Heuristics**: Without an ML model, the retention team would need to manually review all customer accounts or rely on anecdotal signals. A rule-of-thumb approach based on contract type alone would miss many at-risk customers in other segments.

**Training Data**: `telco_churn_cleaned.csv` — 7,043 customer records with 19 features and one binary target. Class imbalance (~26.5% churn) was addressed using SMOTE oversampling applied within the training pipeline.

---

## Dashboard Design

The Streamlit dashboard consists of five pages, accessible via the left sidebar navigation.

### Page 1 — Project Summary

Displays a high-level overview of the project for any visitor who opens the dashboard.

- Description of the dataset (source, size, key features)
- Summary of both business requirements
- Link to this README

![Project Summary Page](docs/screenshots/page_summary.png)

### Page 2 — EDA & Correlation Study

Addresses **Business Requirement 1** by presenting the full correlation analysis visually.

- Churn distribution bar chart with class imbalance note
- Numeric correlation heatmap showing feature relationships with Churn
- Distribution plots for top numeric features: tenure and MonthlyCharges
- Churn rate bar charts for Contract type, TechSupport, and InternetService
- Written interpretation beneath each plot
- Conclusion statement directly answering Business Requirement 1

![EDA Page](docs/screenshots/page_eda.png)

### Page 3 — Project Hypothesis

Documents the analytical assumptions made before the EDA and whether the data confirmed them.

- Hypothesis 1: Contract type and churn — **CONFIRMED**
- Hypothesis 2: Monthly charges and churn — **CONFIRMED**
- Hypothesis 3: Tenure and churn — **CONFIRMED**
- Each hypothesis includes a rationale and a validation statement linked to EDA findings

![Hypothesis Page](docs/screenshots/page_hypothesis.png)

### Page 4 — Model Performance

Addresses **Business Requirement 2** by showing full model evaluation results.

- ML pipeline architecture overview (preprocessing steps + classifier)
- Confusion matrix image from the test set
- Full classification report displayed as a table
- Dynamically reads the actual Churn recall from `classification_report.csv` and displays whether the 75% target was met

![Model Performance Page](docs/screenshots/page_model_performance.png)

### Page 5 — Churn Predictor

Addresses **Business Requirement 2** by providing a live prediction interface.

- Input widgets (dropdowns and sliders) for all 19 customer features
- A **Predict** button that triggers the pipeline
- The submitted customer profile is shown as a table before the result
- Clear prediction output: "This customer is **likely to churn**" or "**not likely to churn**"
- A retention recommendation is shown for at-risk customers

![Churn Predictor Page](docs/screenshots/page_predictor.png)

---

## How to Run the Project

### Prerequisites

- Python 3.12
- The raw dataset file: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (not included in the repo — place it manually in `outputs/datasets/collection/`)

### Step 1 — Clone the repository

```bash
git clone https://github.com/OmerGamil/customer-churn-predictor.git
cd customer-churn-predictor
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs all required libraries including `ipykernel`, which is needed to run the Jupyter notebooks in VS Code or Gitpod.

### Step 3 — Add the dataset

Place the raw CSV file into the collection folder:

```
outputs/datasets/collection/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

> The CSV files are excluded from version control via `.gitignore` because they contain the original dataset. You must place the file manually before running the notebooks.

### Step 4 — Run the notebooks in order

Open each notebook and select the correct Python kernel (`.venv` or the active environment). Run them **in order**:

| Notebook | Purpose | Output |
|----------|---------|--------|
| `01_DataCollection.ipynb` | Load and inspect the raw data | `telco_churn_raw.csv` |
| `02_DataCleaning.ipynb` | Clean and encode the data | `telco_churn_cleaned.csv` |
| `03_EDA.ipynb` | Generate all EDA plots | PNG files in `outputs/eda/` |
| `04_ModellingEvaluation.ipynb` | Train model and evaluate | `clf_pipeline.pkl`, `confusion_matrix.png`, `classification_report.csv` |

### Step 5 — Launch the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## Deployment

The application is deployed to **Heroku** using the following steps:

1. Ensure `Procfile`, `runtime.txt`, `requirements.txt`, and `setup.sh` are present in the root of the repository.
2. Log in to Heroku: `heroku login`
3. Create a new Heroku app: `heroku create your-app-name`
4. Push the code: `git push heroku main`
5. Open the app: `heroku open`

The `Procfile` runs `setup.sh` to configure Streamlit's headless server settings before launching the app.

**Python version**: 3.12 (specified in `runtime.txt`)

---

## Known Issues and Fixes

### Issue: Old package versions fail to install on Python 3.12

**Symptom**: Running `pip install -r requirements.txt` raises a `BackendUnavailable: Cannot import 'setuptools.build_meta'` error.

**Cause**: The original CI template uses very old pinned versions (e.g. `numpy==1.19.5`, `streamlit==0.85.0`) that were released before Python 3.12 and cannot be built on it.

**Fix**: The `requirements.txt` in this project has been updated to use Python 3.12-compatible versions of all packages. If you still encounter build errors, run:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

### Issue: Jupyter kernel not found in VS Code or Gitpod

**Symptom**: Opening a notebook shows the error:
> *"Running cells with '.venv (Python 3.x)' requires the ipykernel package. Install 'ipykernel' into the Python environment."*

**Cause**: The virtual environment exists but `ipykernel` has not been registered with Jupyter.

**Fix**: `ipykernel` is now included in `requirements.txt`. After installing dependencies, register the kernel manually if the error persists:

```bash
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

Then in VS Code, press `Ctrl+Shift+P` → **Notebook: Select Notebook Kernel** → choose **Python (.venv)**.

---

### Issue: Dataset CSV not found when running notebooks

**Symptom**: `FileNotFoundError` when running the first cell of `01_DataCollection.ipynb`.

**Cause**: The CSV files are excluded from version control via `.gitignore`.

**Fix**: Manually place the raw dataset file at:
```
outputs/datasets/collection/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

In Gitpod, drag and drop the file directly into the folder using the file explorer panel on the left.

---

### Issue: EDA plots or pipeline not found on dashboard pages

**Symptom**: Dashboard pages show yellow warning boxes instead of plots or model metrics.

**Cause**: The notebooks have not been run yet to generate the output files.

**Fix**: Run all four notebooks in order (01 → 04) before launching the Streamlit app. See [How to Run the Project](#how-to-run-the-project) above.

---

## Main Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Numerical operations |
| pandas | 2.2.2 | Data loading and manipulation |
| matplotlib | 3.9.0 | Static plotting |
| seaborn | 0.13.2 | Statistical visualisation |
| plotly | 5.22.0 | Interactive charts |
| scikit-learn | 1.5.0 | ML pipeline, preprocessing, evaluation |
| imbalanced-learn | 0.12.3 | SMOTE oversampling |
| joblib | 1.4.2 | Pipeline serialisation |
| feature-engine | 1.8.1 | Feature engineering utilities |
| streamlit | 1.35.0 | Web dashboard framework |
| Pillow | 10.3.0 | Image loading for dashboard plots |
| ipykernel | 6.29.4 | Jupyter notebook kernel support |

---

## Credits

- **Dataset**: IBM Telco Customer Churn dataset, sourced from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Course and project framework**: [Code Institute](https://codeinstitute.net/) — Diploma in Full Stack Software Development (Predictive Analytics pathway)
- **Mentor**: Mr Mo Shami — for guidance, support, and feedback throughout the project
