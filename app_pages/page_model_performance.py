"""
page_model_performance.py — Streamlit page: ML Model Performance.

Shows the trained pipeline configuration, precision-recall trade-off
chart, confusion matrix, classification report, feature importance
chart, and the business requirement outcome for the Customer Churn
Predictor model.
"""

import streamlit as st
import os
import joblib
import pandas as pd
from PIL import Image


PIPELINE_DIR = "outputs/ml_pipeline/predict_churn/v1"


def page_model_performance_body():
    """Render the Model Performance page content."""

    st.write("## ML Model Performance")

    st.info(
        "**Business Requirement 2**: The client wants to predict whether "
        "a given customer will churn. The success metric is "
        "**≥75% recall** on the Churn class (label = 1)."
    )

    # --- Pipeline overview ---
    st.write("---")
    st.write("### Pipeline Steps")
    st.write(
        "The ML pipeline consists of three stages applied in order:\n\n"
        "1. **Feature Engineering** (`ColumnTransformer`)\n"
        "   - `OneHotEncoder(drop='first')` for categorical features\n"
        "   - `StandardScaler` for numeric features\n"
        "2. **Oversampling** — `SMOTE` applied only within the "
        "training set to address the ~26.5% class imbalance.\n"
        "3. **Classifier** — `GradientBoostingClassifier`, tuned "
        "via `GridSearchCV` (5-fold CV, scoring = recall) across 6 "
        "hyperparameters × 3 values each.\n\n"
        "After training, the **decision threshold** is tuned on the "
        "validation set using the precision-recall curve to find the "
        "highest threshold that still meets ≥75% recall."
    )

    # Show optimal threshold if saved
    threshold_path = os.path.join(PIPELINE_DIR, "optimal_threshold.pkl")
    if os.path.exists(threshold_path):
        threshold = joblib.load(threshold_path)
        st.write(
            f"**Optimal decision threshold**: `{threshold:.4f}` "
            "(lowered from the default 0.5 to increase recall on the "
            "Churn class)"
        )

    # --- Precision-Recall trade-off ---
    st.write("---")
    st.write("### Precision-Recall Trade-off (Validation Set)")
    pr_path = os.path.join(PIPELINE_DIR, "threshold_tuning.png")
    if os.path.exists(pr_path):
        st.image(Image.open(pr_path))
        st.write(
            "The chart shows how precision and recall change as the "
            "decision threshold varies. The green dashed line marks the "
            "75% recall target. The optimal threshold is the highest "
            "value that still keeps recall at or above this target, "
            "balancing the need to catch churners while limiting "
            "unnecessary retention outreach."
        )
    else:
        st.warning(
            "Plot not found. "
            "Run Notebook 05_ModellingEvaluation.ipynb first."
        )

    # --- Confusion matrix ---
    st.write("---")
    st.write("### Confusion Matrix (Test Set)")
    cm_path = os.path.join(PIPELINE_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(Image.open(cm_path))
        st.write(
            "**True Positives** (churners correctly flagged) are the "
            "priority — these are the customers the retention team "
            "can act on. **False Negatives** (missed churners) are the "
            "costliest error: the model missed them and no retention "
            "action was taken. **False Positives** (non-churners "
            "incorrectly flagged) waste resources but do not result in "
            "lost customers."
        )
    else:
        st.warning(
            "Confusion matrix not found. "
            "Run Notebook 05_ModellingEvaluation.ipynb first."
        )

    # --- Classification report ---
    st.write("---")
    st.write("### Classification Report (Test Set)")
    report_path = os.path.join(PIPELINE_DIR, "classification_report.csv")
    if os.path.exists(report_path):
        report_df = pd.read_csv(report_path, index_col=0)
        st.dataframe(report_df)
        st.write(
            "The table shows precision, recall, and F1-score for both "
            "classes on the held-out test set. The **Recall** column for "
            "the Churn row is the key metric against the business "
            "requirement."
        )
    else:
        st.warning(
            "Classification report not found. "
            "Run Notebook 05_ModellingEvaluation.ipynb first."
        )

    # --- Feature importance ---
    st.write("---")
    st.write("### Feature Importance")
    fi_path = os.path.join(PIPELINE_DIR, "feature_importance.png")
    if os.path.exists(fi_path):
        st.image(Image.open(fi_path))
        st.write(
            "Feature importance is measured by the mean decrease in "
            "impurity (MDI) across all trees in the "
            "GradientBoostingClassifier. Higher bars indicate features "
            "the model relied on most when splitting. Contract type, "
            "tenure, and MonthlyCharges are typically the top drivers "
            "— consistent with the EDA findings."
        )
    else:
        st.warning(
            "Feature importance plot not found. "
            "Run Notebook 05_ModellingEvaluation.ipynb first."
        )

    # --- Business requirement outcome ---
    st.write("---")
    st.write("### Business Requirement Outcome")

    if os.path.exists(report_path):
        report_df = pd.read_csv(report_path, index_col=0)
        if "Churn" in report_df.index:
            churn_recall = report_df.loc["Churn", "recall"]
            churn_precision = report_df.loc["Churn", "precision"]
            target = 0.75
            if churn_recall >= target:
                st.success(
                    f"**Business Requirement MET** \n\n"
                    f"The model achieved "
                    f"**{churn_recall * 100:.1f}% recall** and "
                    f"**{churn_precision * 100:.1f}% precision** on the "
                    f"Churn class on the held-out test set, exceeding "
                    f"the ≥75% recall target. The pipeline is "
                    f"suitable for deployment."
                )
            else:
                st.error(
                    f"**Business Requirement NOT MET** \n\n"
                    f"The model achieved "
                    f"**{churn_recall * 100:.1f}% recall** on the "
                    f"Churn class, which is below the ≥75% target."
                )
    else:
        st.warning(
            "Run Notebook 05_ModellingEvaluation.ipynb to train the "
            "model. Results will appear here automatically once complete."
        )
