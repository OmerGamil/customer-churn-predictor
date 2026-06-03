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

The dataset is sourced from IBM's **Telco Customer Churn** dataset, available publicly on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn). It contains **7,043 records** and **20 features** describing customer demographics, account information, and services subscribed to. The target variable is **Churn**, which indicates whether the customer left the company within the last month.

| Feature Group | Features |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account info | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Target | Churn (Yes / No, encoded as 1 / 0) |

The dataset is publicly available and contains no personally identifiable information. No ethical or privacy concerns apply.

---

## Business Requirements

**Business Requirement 1 — Correlation Study**
> The client wants to understand which customer attributes most strongly correlate with churn, so that the retention team can identify and prioritise at-risk customer segments. This is addressed through conventional data analysis.

**Business Requirement 2 — Churn Prediction**
> The client wants to predict whether a given customer will churn, enabling proactive retention action before the customer leaves. The model must achieve **at least 75% recall on the Churn class** to be considered a successful deployment.

---

## Hypothesis and Validation

### Hypothesis 1 — Contract Type and Churn

**Statement**: Customers with month-to-month contracts are significantly more likely to churn than those on one-year or two-year contracts.

**Rationale**: Month-to-month customers have no long-term commitment, making it frictionless to cancel at any time. Annual contracts create a financial and procedural barrier to leaving.

**Validation process**: Bar chart of churn rate per contract type, computed from the cleaned dataset.

| Contract Type | Churn Rate |
|---|---|
| Month-to-month | ~42% |
| One year | ~11% |
| Two year | ~3% |

**Outcome**: ✅ **CONFIRMED** — Month-to-month customers churn at roughly 14× the rate of two-year contract customers. Contract type is among the strongest categorical predictors of churn in the dataset.

---

### Hypothesis 2 — Monthly Charges and Churn

**Statement**: Customers with higher monthly charges are more likely to churn than customers paying lower amounts.

**Rationale**: Customers who perceive the service as expensive relative to its value are more motivated to seek cheaper alternatives.

**Validation process**: Pearson correlation between MonthlyCharges and the binary Churn column; distribution plot comparing MonthlyCharges for churners vs non-churners.

- **Pearson r = +0.19** (positive correlation with Churn)
- Churned customers have a higher average monthly charge (~$74) compared to retained customers (~$61)

**Outcome**: ✅ **CONFIRMED** — MonthlyCharges is positively correlated with churn. Higher charges are a measurable risk factor.

---

### Hypothesis 3 — Tenure and Churn

**Statement**: Customers who have been with the company for a shorter time (low tenure) are more likely to churn.

**Rationale**: New customers have not yet built loyalty and may still be evaluating competing providers. Long-tenured customers have demonstrated commitment and are more embedded in the service.

**Validation process**: Pearson correlation between tenure and the binary Churn column; box plot comparing tenure distributions for churners vs non-churners.

- **Pearson r = −0.35** (strongest numeric correlation with Churn)
- Churned customers have a mean tenure of ~18 months vs ~38 months for retained customers

**Outcome**: ✅ **CONFIRMED** — Tenure is the strongest individual numeric predictor of churn. Early-stage customers are the highest-risk segment.

---

## Rationale to Map Business Requirements to Data Visualisations and ML Tasks

### Business Requirement 1 — Correlation Study

**User Story 1**: As a data analyst at the telecom company, I want to see which customer attributes are most correlated with churn, so that I can brief the retention team on which customer segments to prioritise.
- A Pearson correlation heatmap was produced for all numeric features against the Churn target.
- Distribution plots (box plots and histograms) were created for the top correlated numeric features: tenure and MonthlyCharges.
- Bar charts of churn rate were created for key categorical features: Contract type, TechSupport, and InternetService.
- All plots are displayed on the **EDA & Correlation Study** dashboard page with written interpretations beneath each.

**User Story 2**: As a product manager, I want to understand why customers with certain service configurations are at higher risk, so that I can design targeted retention campaigns.
- The correlation study identifies service-related features (TechSupport, InternetService type) that are associated with higher churn rates, giving the product team actionable recommendations.

### Business Requirement 2 — Churn Prediction

**User Story 3**: As a customer success manager, I want to enter a customer's details and receive an instant churn prediction, so that I can proactively reach out to at-risk customers before they cancel.
- A supervised binary classification ML pipeline was built and trained on the cleaned dataset.
- The pipeline is embedded in the **Churn Predictor** dashboard page, where staff can input 19 customer attributes and receive an instant prediction ("Likely to churn" / "Not likely to churn").

**User Story 4**: As a business stakeholder, I want to know whether the ML model meets the agreed performance target, so that I can decide whether to deploy it to the retention team.
- The **Model Performance** dashboard page displays the full evaluation report, confusion matrix, and a clear ✅/❌ statement confirming whether the ≥75% recall target was met.

---

## ML Business Case

### Objective

Develop a **supervised binary classification** model that predicts whether a telecom customer will churn (target label = 1) or remain (target label = 0), based on 19 customer attributes.

### Learning Method

Supervised learning using a **GradientBoostingClassifier** — an ensemble method that builds decision trees sequentially, each correcting the residual errors of the previous one. This approach is well-suited to tabular data with mixed feature types and class imbalance.

### Ideal Outcome

A trained ML pipeline that the retention team can use to flag at-risk customers before they leave, enabling proactive intervention.

### Success Metrics

| Metric | Target | Priority |
|---|---|---|
| Recall (Churn class) | ≥ 75% | Primary — minimises missed churners (false negatives) |
| Precision (Churn class) | Maximise | Secondary — reduces wasted retention effort |
| F1-score | Track | Balances precision and recall |

Recall is the primary metric because a **false negative** (missing a customer who churns) is more costly to the business than a **false positive** (unnecessarily contacting a customer who would have stayed). Maximising recall is the business requirement.

### Model Output

A **binary class prediction** (0 = no churn, 1 = churn) for a single customer record, generated using `predict_proba` with a tuned decision threshold (rather than the default 0.5 cutoff). The threshold is chosen from the **precision-recall curve** on the validation set to be the highest value that still achieves ≥75% recall.

### ML Pipeline Steps

1. **Feature Encoding**: `OneHotEncoder` (drop first) for categorical features, applied via `ColumnTransformer`.
2. **Feature Scaling**: `StandardScaler` for numeric features, applied in the same `ColumnTransformer`.
3. **Oversampling**: `SMOTE` (Synthetic Minority Oversampling Technique) applied only within training folds to address the ~26.5% class imbalance, preventing data leakage into the validation set.
4. **Classifier**: `GradientBoostingClassifier` with hyperparameters tuned via `GridSearchCV` (5-fold cross-validation, scoring = recall).
5. **Threshold Tuning**: The decision threshold is tuned on the validation set using `precision_recall_curve`, then evaluated on the held-out test set.

### Hyperparameter Optimisation Rationale

| Hyperparameter | Values Searched | Rationale |
|---|---|---|
| `n_estimators` | 50, 100, 140 | More trees generally improve recall but increase training time |
| `learning_rate` | 0.001, 0.01, 0.1 | Lower rates require more trees; higher rates risk overfitting |
| `max_depth` | 3, 10, 15 | Controls tree complexity; deeper trees capture more patterns but may overfit |
| `min_samples_split` | 2, 20, 50 | Higher values prevent the tree from splitting on noise |
| `min_samples_leaf` | 1, 10, 50 | Minimum leaf size smooths predictions and prevents overfitting |
| `max_leaf_nodes` | None, 25, 50 | Limits total leaves; None = unlimited, others cap tree complexity |

### Heuristics

Without an ML model, the retention team would need to manually review all customer accounts or rely on anecdotal signals such as recent support tickets or contract renewal dates. A simple rule-of-thumb based on contract type alone would miss many at-risk customers in other segments.

### Training Data

`telco_churn_cleaned.csv` — 7,043 customer records with 19 features and one binary target column. Data was split into **60% train / 20% validation / 20% test** sets, stratified on the target to preserve the class distribution across all splits.

### Failure Condition

If the final model does not achieve ≥75% recall on the Churn class of the held-out test set, the pipeline is considered unsuitable for deployment and further iteration is required (e.g., additional threshold adjustment or an alternative classifier).

---

## Dashboard Design

The Streamlit dashboard consists of five pages, accessible via the sidebar navigation.

### Page 1 — Project Summary

Addresses **criterion 1.1** by presenting the project context to any first-time visitor.

- Short description of the Telco Customer Churn dataset (source, size, key feature groups)
- Summary of both business requirements displayed in green info boxes
- Link to the project README for further detail

### Page 2 — EDA & Correlation Study

Addresses **Business Requirement 1** by presenting the full correlation analysis visually.

- Info box restating Business Requirement 1
- **Churn distribution** bar chart with class imbalance note and interpretation
- **Numeric correlation heatmap** (Pearson) for all numeric features against Churn, with interpretation
- **tenure vs Churn** distribution plot — showing churners skew toward low tenure
- **MonthlyCharges vs Churn** distribution plot — showing churners pay more on average
- **Contract type vs Churn** bar chart — showing month-to-month has highest churn rate
- **TechSupport vs Churn** bar chart — showing lack of support correlates with churn
- **InternetService vs Churn** bar chart — showing Fibre optic has elevated churn rate
- Written interpretation beneath every plot
- Conclusion box directly answering Business Requirement 1

### Page 3 — Project Hypothesis

Documents the three analytical assumptions and their statistical validation.

- **Hypothesis 1**: Contract type and churn — churn rates per contract tier, **CONFIRMED**
- **Hypothesis 2**: Monthly charges and churn — Pearson r, average charges per class, **CONFIRMED**
- **Hypothesis 3**: Tenure and churn — Pearson r, average tenure per class, **CONFIRMED**
- Each hypothesis includes the original statement, rationale, statistical evidence, and outcome

### Page 4 — Model Performance

Addresses **Business Requirement 2** — model evaluation.

- Pipeline architecture overview (preprocessing → SMOTE → GradientBoostingClassifier)
- Optimal decision threshold value (loaded dynamically from `optimal_threshold.pkl`)
- **Precision-Recall trade-off** line chart (threshold_tuning.png) with 75% target line
- **Confusion matrix** heatmap for the test set (confusion_matrix.png)
- **Classification report** table (loaded from `classification_report.csv`)
- **Feature importance** bar chart (feature_importance.png) — top 15 most influential features
- Dynamic success/failure statement: ✅ if Churn recall ≥ 75%, ❌ otherwise

### Page 5 — Churn Predictor

Addresses **Business Requirement 2** — live prediction interface.

- Input widgets (dropdowns and sliders) for all 19 customer attributes, laid out in three columns
- A **Predict** button that assembles the inputs into a one-row DataFrame and passes it to the pipeline
- The submitted customer profile is echoed back as a table before the prediction result
- Clear result: **"This customer is likely to churn"** (error box) or **"not likely to churn"** (success box)
- A retention recommendation is shown for at-risk customers

---

## How to Run the Project

### Prerequisites

- Python 3.12
- The raw dataset file `WA_Fn-UseC_-Telco-Customer-Churn.csv` (place it in `outputs/datasets/collection/` — not included in the repo)

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

Place the raw CSV file at:

```
outputs/datasets/collection/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

> CSV files are excluded from version control via `.gitignore`. Place the file manually before running the notebooks.

### Step 4 — Run the notebooks in order

Open each notebook and select the correct Python kernel (`.venv` or the active environment). Run them **in order**:

| Notebook | Purpose | Output |
|----------|---------|--------|
| `01_DataCollection.ipynb` | Load and inspect the raw data | `telco_churn_raw.csv` |
| `02_DataCleaning.ipynb` | Clean and encode the data | `telco_churn_cleaned.csv` |
| `03_EDA.ipynb` | Generate all EDA plots | PNG files in `outputs/eda/` |
| `04_FeatureEngineering.ipynb` | Assess feature transformations | Documents transformation decisions |
| `05_ModellingEvaluation.ipynb` | Train model and evaluate | `clf_pipeline.pkl`, `optimal_threshold.pkl`, evaluation artefacts |

### Step 5 — Launch the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Deployment

The application is deployed to **Heroku** using the following configuration:

| File | Purpose |
|---|---|
| `Procfile` | Tells Heroku to run `setup.sh` then `streamlit run app.py` |
| `setup.sh` | Writes Streamlit server settings for headless deployment |
| `runtime.txt` | Specifies `python-3.12.1` |
| `requirements.txt` | All Python dependencies |

**Deploy steps:**

```bash
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

---

## Known Issues and Fixes

### Issue: Old package versions fail to install on Python 3.12

**Symptom**: `pip install -r requirements.txt` raises `BackendUnavailable: Cannot import 'setuptools.build_meta'`.

**Cause**: Old pinned versions (e.g. `numpy==1.19.5`) cannot be built on Python 3.12.

**Fix**: `requirements.txt` uses Python 3.12-compatible versions. If errors persist:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

### Issue: Jupyter kernel not found in VS Code or Gitpod

**Symptom**: *"Running cells with '.venv' requires the ipykernel package."*

**Fix**: `ipykernel` is in `requirements.txt`. If the error persists after installing:

```bash
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

Then in VS Code: `Ctrl+Shift+P` → **Notebook: Select Notebook Kernel** → choose **Python (.venv)**.

---

### Issue: Dataset CSV not found

**Symptom**: `FileNotFoundError` when running `01_DataCollection.ipynb`.

**Fix**: Place the raw dataset at `outputs/datasets/collection/WA_Fn-UseC_-Telco-Customer-Churn.csv`. In Gitpod, drag and drop the file into the folder via the file explorer.

---

### Issue: Dashboard shows warnings instead of plots

**Symptom**: Yellow warning boxes on EDA or Model Performance pages.

**Fix**: Run all notebooks in order (01 → 05) before launching the app. Notebook outputs are excluded from version control.

---

## Main Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Numerical operations |
| pandas | 2.2.2 | Data loading and manipulation |
| matplotlib | 3.9.0 | Static plotting |
| seaborn | 0.13.2 | Statistical visualisation |
| plotly | 5.22.0 | Interactive charts |
| scikit-learn | 1.5.0 | ML pipeline, preprocessing, GridSearchCV, evaluation |
| imbalanced-learn | 0.12.3 | SMOTE oversampling |
| joblib | 1.4.2 | Pipeline serialisation |
| feature-engine | 1.8.1 | Feature engineering utilities |
| scipy | 1.13.0 | Statistical tests |
| streamlit | 1.35.0 | Web dashboard framework |
| Pillow | 10.3.0 | Image loading for dashboard plots |
| ipykernel | 6.29.4 | Jupyter notebook kernel support |

---

## Credits

- **Dataset**: IBM Telco Customer Churn dataset, sourced from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn). The dataset is publicly available with no privacy concerns.
- **Course and project framework**: [Code Institute](https://codeinstitute.net/) — Diploma in Full Stack Software Development (Predictive Analytics pathway)
- **Mentor**: Mr Mo Shami — for guidance, support, and feedback throughout the project
