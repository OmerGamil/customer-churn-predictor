"""
page_churn_predictor.py — Streamlit page: Churn Predictor.

Provides an interactive form for entering customer attributes and
predicts churn probability using the saved ML pipeline and optimal threshold.
Addresses Business Requirement 2 of the Customer Churn Predictor project.
"""

import streamlit as st
import pandas as pd
from src.data_management import load_pkl_file
from src.machine_learning.predictive_analysis import predict_churn


PIPELINE_PATH = "outputs/ml_pipeline/predict_churn/v1/clf_pipeline.pkl"


def page_churn_predictor_body():
    """Render the Churn Predictor page with an input form and prediction output."""
    st.write("## Churn Predictor")
    st.info(
        "Enter customer details below and click **Predict** to find out whether "
        "this customer is likely to churn."
    )

    st.write("---")
    st.write("### Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", options=["No", "Yes"])
        partner = st.selectbox("Partner", options=["No", "Yes"])
        dependents = st.selectbox("Dependents", options=["No", "Yes"])
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone Service", options=["No", "Yes"])
        multiple_lines = st.selectbox(
            "Multiple Lines", options=["No", "Yes", "No phone service"]
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service", options=["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", options=["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", options=["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", options=["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", options=["No", "Yes", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV", options=["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", options=["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", options=["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless Billing", options=["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.slider(
            "Monthly Charges ($)", min_value=18.0, max_value=119.0, value=65.0
        )
        total_charges = st.slider(
            "Total Charges ($)", min_value=18.0, max_value=8685.0, value=1500.0
        )

    st.write("---")

    if st.button("Predict Churn"):
        X_live = pd.DataFrame(
            {
                "gender": [gender],
                "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": [phone_service],
                "MultipleLines": [multiple_lines],
                "InternetService": [internet_service],
                "OnlineSecurity": [online_security],
                "OnlineBackup": [online_backup],
                "DeviceProtection": [device_protection],
                "TechSupport": [tech_support],
                "StreamingTV": [streaming_tv],
                "StreamingMovies": [streaming_movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless_billing],
                "PaymentMethod": [payment_method],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges],
            }
        )

        st.write("#### Customer Profile Submitted")
        st.dataframe(X_live.T.rename(columns={0: "Value"}))

        try:
            pipeline = load_pkl_file(PIPELINE_PATH)
            prediction_label = predict_churn(X_live, pipeline)
            if prediction_label == 1:
                st.error(
                    "**Prediction: This customer is likely to churn.** \n\n"
                    "We recommend proactive retention action such as a personalised "
                    "offer or a customer satisfaction check-in."
                )
            else:
                st.success(
                    "**Prediction: This customer is not likely to churn.** \n\n"
                    "This customer appears to be at low risk of leaving."
                )
        except FileNotFoundError:
            st.warning(
                f"Pipeline file not found at `{PIPELINE_PATH}`. "
                "Please run Notebook 04_ModellingEvaluation.ipynb to train and "
                "save the model first."
            )
