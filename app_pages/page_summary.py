"""
page_summary.py — Streamlit page: Project Summary.

Displays the dataset overview, business requirements, and links
to the project README for the Customer Churn Predictor dashboard.
"""

import streamlit as st


def page_summary_body():
    """Render the Project Summary page content."""
    st.write("## Customer Churn Prediction")

    st.info(
        "### Dataset Content\n"
        "The dataset is sourced from IBM's Telco Customer Churn dataset. "
        "It contains **7,043 records** and **20 features** describing customer "
        "demographics, account information, and services subscribed. "
        "The target variable is **Churn**, indicating whether the customer left "
        "within the last month.\n\n"
        "Key features include:\n"
        "- **Demographics**: gender, SeniorCitizen, Partner, Dependents\n"
        "- **Account info**: tenure, Contract, PaperlessBilling, PaymentMethod, "
        "MonthlyCharges, TotalCharges\n"
        "- **Services**: PhoneService, MultipleLines, InternetService, "
        "OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, "
        "StreamingTV, StreamingMovies"
    )

    st.write("---")
    st.write("### Business Requirements")

    st.success(
        "**Business Requirement 1** \n\n"
        "The client wants to understand which customer attributes most strongly "
        "correlate with churn, so that the retention team can identify at-risk "
        "customer segments."
    )

    st.success(
        "**Business Requirement 2** \n\n"
        "The client wants to predict whether a given customer will churn, so that "
        "proactive retention actions can be taken before the customer leaves. "
        "The model must achieve at least **75% recall** on the Churn class to be "
        "considered successful."
    )

    st.write("---")
    st.write(
        "For additional information, please visit the "
        "[Project README](https://github.com)."
    )
