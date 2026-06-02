# Customer Churn Predictor

A machine learning web application that predicts whether a telecom customer will churn, built as part of the Code Institute Diploma in Full Stack Software Development (Predictive Analytics).

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

The Streamlit dashboard consists of five pages:

### Page 1 — Project Summary
- Description of the dataset (source, size, features)
- Summary of both business requirements
- Link to README

### Page 2 — EDA & Correlation Study
- Addresses Business Requirement 1
- Churn distribution bar chart with class imbalance note
- Numeric correlation heatmap
- Distribution plots for top features: tenure, MonthlyCharges
- Churn rate bar charts for Contract, TechSupport, and InternetService
- Written interpretation beneath each plot
- Conclusion statement answering Business Requirement 1

### Page 3 — Project Hypothesis
- Hypothesis 1: Contract type and churn (CONFIRMED)
- Hypothesis 2: Monthly charges and churn (CONFIRMED)
- Hypothesis 3: Tenure and churn (CONFIRMED)
- Each hypothesis includes a rationale and validation statement

### Page 4 — Model Performance
- Addresses Business Requirement 2
- Pipeline architecture overview
- Confusion matrix image
- Classification report table
- Clear statement whether the 75% recall target was met

### Page 5 — Churn Predictor
- Input widgets for all 19 customer features (dropdowns and sliders)
- Predict button
- Loads the trained pipeline from `clf_pipeline.pkl`
- Displays prediction: "This customer is likely to churn" or "not likely to churn"

---

## Deployment

The application is deployed to **Heroku** using the following steps:

1. Ensure `Procfile`, `runtime.txt`, `requirements.txt`, and `setup.sh` are present in the root of the repository.
2. Log in to Heroku: `heroku login`
3. Create a new Heroku app: `heroku create your-app-name`
4. Push the code: `git push heroku main`
5. Open the app: `heroku open`

The `Procfile` runs `setup.sh` to configure Streamlit's headless server settings before launching the app.

**Python version**: 3.8.18 (specified in `runtime.txt`)

---

## Main Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | 1.3.5 | Data loading and manipulation |
| numpy | 1.19.5 | Numerical operations |
| matplotlib | 3.3.1 | Static plotting |
| seaborn | 0.11.0 | Statistical visualisation |
| plotly | 4.12.0 | Interactive charts |
| scikit-learn | 0.24.2 | ML pipeline, preprocessing, evaluation |
| imbalanced-learn | 0.8.0 | SMOTE oversampling |
| joblib | 1.0.1 | Pipeline serialisation |
| feature-engine | 1.0.2 | Feature engineering utilities |
| streamlit | 0.85.0 | Web dashboard framework |

---

## Credits

- **Dataset**: IBM Telco Customer Churn dataset, sourced from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Project structure and methodology**: Inspired by Code Institute Predictive Analytics walkthrough projects
- **CRISP-DM methodology**: Applied throughout notebooks and dashboard design
- **Reference projects** (structure only, no code copied):
  - [Brain Tumor Detector](https://github.com/tomdu3/brain-tumor-detector)
  - [Plant Disease Classification](https://github.com/teman67/Plant-Disease-Classification-Project)
  - [Cherry Leaf Mildew Detection](https://github.com/barty-s/cherry-leaf-mildew-detection-project)
