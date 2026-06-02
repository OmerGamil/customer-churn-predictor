import streamlit as st
import os
import pandas as pd
from PIL import Image


PIPELINE_DIR = "outputs/ml_pipeline/predict_churn/v1"


def page_model_performance_body():
    st.write("## ML Model Performance")

    st.info(
        "**Business Requirement 2**: The client wants to predict whether a given "
        "customer will churn. The success metric is **≥75% recall** on the "
        "Churn class (label = 1)."
    )

    st.write("---")
    st.write("### Pipeline Overview")
    st.write(
        "The ML pipeline consists of two stages:\n"
        "1. **Feature Engineering**: OneHotEncoder for categorical features, "
        "StandardScaler for numerical features.\n"
        "2. **Classifier**: RandomForestClassifier, tuned via GridSearchCV."
    )

    st.write("---")
    st.write("### Confusion Matrix")
    cm_path = os.path.join(PIPELINE_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        image = Image.open(cm_path)
        st.image(image)
        st.write(
            "The confusion matrix shows the model's predictions on the test set. "
            "True Positives (churners correctly identified) are critical for the "
            "business use case, as missing a churner (False Negative) is more "
            "costly than a false alarm."
        )
    else:
        st.warning(
            f"Confusion matrix not found at `{cm_path}`. "
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
            f"Classification report not found at `{report_path}`. "
            "Run Notebook 04_ModellingEvaluation.ipynb first."
        )

    st.write("---")
    st.write("### Business Requirement Outcome")

    report_path = os.path.join(PIPELINE_DIR, "classification_report.csv")
    if os.path.exists(report_path):
        report_df = pd.read_csv(report_path, index_col=0)
        if "Churn" in report_df.index:
            churn_recall = report_df.loc["Churn", "recall"]
            target = 0.75
            if churn_recall >= target:
                st.success(
                    f"**Business Requirement MET** \n\n"
                    f"The model achieved **{churn_recall*100:.1f}% recall** on the "
                    f"Churn class, exceeding the ≥75% target. The pipeline is "
                    f"suitable for deployment."
                )
            else:
                st.error(
                    f"**Business Requirement NOT MET** \n\n"
                    f"The model achieved **{churn_recall*100:.1f}% recall** on the "
                    f"Churn class, which is below the ≥75% target. Consider "
                    f"adjusting the decision threshold or trying a different model."
                )
        else:
            st.warning("Could not find 'Churn' row in classification report.")
    else:
        st.warning(
            "Run Notebook 04_ModellingEvaluation.ipynb to train the model. "
            "The business requirement outcome will be displayed here automatically "
            "once the classification report is generated."
        )
