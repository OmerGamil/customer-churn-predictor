"""
page_model_performance.py — Streamlit page: ML Model Performance.

Shows the trained pipeline configuration, precision-recall trade-off chart,
confusion matrix, classification report, and business requirement outcome
for the Customer Churn Predictor model.
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
        "**Business Requirement 2**: The client wants to predict whether a given "
        "customer will churn. The success metric is **≥75% recall** on the "
        "Churn class (label = 1)."
    )

    st.write("---")
    st.write("### Pipeline Overview")
    st.write(
        "The ML pipeline consists of three stages:\n"
        "1. **Feature Engineering**: OneHotEncoder for categorical features, "
        "StandardScaler for numerical features.\n"
        "2. **Oversampling**: SMOTE applied on the training set to handle the "
        "class imbalance (~26.5% churn rate).\n"
        "3. **Classifier**: GradientBoostingClassifier, tuned via GridSearchCV "
        "optimising for recall on the Churn class.\n\n"
        "After training, a **precision-recall curve** was used to find the optimal "
        "decision threshold that meets the ≥75% recall business requirement."
    )

    threshold_path = os.path.join(PIPELINE_DIR, "optimal_threshold.pkl")
    if os.path.exists(threshold_path):
        threshold = joblib.load(threshold_path)
        st.write(f"**Optimal decision threshold**: `{threshold:.4f}` "
                 f"(default is 0.5 — lowered to increase recall on Churn class)")

    st.write("---")
    st.write("### Precision-Recall Trade-off")
    pr_path = os.path.join(PIPELINE_DIR, "threshold_tuning.png")
    if os.path.exists(pr_path):
        st.image(Image.open(pr_path))
        st.write(
            "The chart shows how precision and recall change as the decision "
            "threshold varies. The green dashed line marks the 75% recall target. "
            "The optimal threshold was selected as the highest value that still "
            "keeps recall at or above this target."
        )

    st.write("---")
    st.write("### Confusion Matrix")
    cm_path = os.path.join(PIPELINE_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(Image.open(cm_path))
        st.write(
            "The confusion matrix shows model predictions on the held-out test set. "
            "True Positives (churners correctly identified) are the priority metric. "
            "False Negatives (missed churners) are the costliest error for the business."
        )
    else:
        st.warning(
            f"Confusion matrix not found. "
            "Run Notebook 04_ModellingEvaluation.ipynb first."
        )

    st.write("---")
    st.write("### Classification Report")
    report_path = os.path.join(PIPELINE_DIR, "classification_report.csv")
    if os.path.exists(report_path):
        report_df = pd.read_csv(report_path, index_col=0)
        st.dataframe(report_df)
    else:
        st.warning(
            "Classification report not found. "
            "Run Notebook 04_ModellingEvaluation.ipynb first."
        )

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
                    f"The model achieved **{churn_recall*100:.1f}% recall** and "
                    f"**{churn_precision*100:.1f}% precision** on the Churn class, "
                    f"exceeding the ≥75% recall target. The pipeline is suitable "
                    f"for deployment."
                )
            else:
                st.error(
                    f"**Business Requirement NOT MET** \n\n"
                    f"The model achieved **{churn_recall*100:.1f}% recall** on the "
                    f"Churn class, which is below the ≥75% target."
                )
    else:
        st.warning(
            "Run Notebook 04_ModellingEvaluation.ipynb to train the model. "
            "Results will appear here automatically once complete."
        )
