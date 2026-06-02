"""
page_eda.py — Streamlit page: EDA & Correlation Study.

Displays exploratory data analysis plots and interpretations that
address Business Requirement 1 (identify attributes correlated with churn).
"""

import streamlit as st
import os
from PIL import Image


EDA_DIR = "outputs/eda"


def page_eda_body():
    """Render the EDA & Correlation Study page content."""
    st.write("## Customer Churn Correlation Study")

    st.info(
        "**Business Requirement 1**: The client wants to understand which customer "
        "attributes most strongly correlate with churn."
    )

    st.write(
        "An exploratory data analysis (EDA) was conducted on the Telco Customer "
        "Churn dataset. The study examines the relationship between customer "
        "attributes and the Churn target variable using correlation analysis "
        "and visualisations."
    )

    st.write("---")
    st.write("### Churn Distribution")
    _display_plot(
        "churn_distribution.png",
        "The dataset shows a class imbalance: approximately 26.5% of customers "
        "churned. This imbalance was addressed during modelling using SMOTE "
        "oversampling to prevent bias towards the majority (non-churn) class."
    )

    st.write("---")
    st.write("### Correlation Heatmap (Numeric Features vs Churn)")
    _display_plot(
        "correlation_heatmap.png",
        "The heatmap reveals that **tenure** has the strongest negative correlation "
        "with Churn (-0.35), meaning longer-tenured customers are less likely to "
        "leave. **MonthlyCharges** shows a positive correlation (+0.19), indicating "
        "that higher monthly charges are associated with higher churn risk. "
        "**TotalCharges** is negatively correlated, largely driven by tenure."
    )

    st.write("---")
    st.write("### Top Correlated Features vs Churn")

    plots = [
        ("tenure_vs_churn.png",
         "Customers who churn have significantly lower average tenure. The "
         "distribution of tenure for churners is heavily skewed towards shorter "
         "service periods, confirming that new customers are at higher risk."),
        ("monthlycharges_vs_churn.png",
         "Churned customers pay higher average monthly charges. This suggests that "
         "pricing sensitivity is a key driver of churn and that premium-tier customers "
         "are at elevated risk."),
        ("contract_vs_churn.png",
         "Month-to-month contract customers churn at a far higher rate than those "
         "on one-year or two-year contracts. Long-term contracts act as a strong "
         "retention mechanism."),
        ("techsupport_vs_churn.png",
         "Customers without tech support churn at a higher rate. This suggests that "
         "support availability contributes to customer satisfaction and retention."),
        ("internetservice_vs_churn.png",
         "Fibre optic internet customers churn more than DSL or no-internet customers. "
         "This may reflect higher price sensitivity or unmet expectations for this "
         "premium service tier."),
    ]

    for filename, interpretation in plots:
        _display_plot(filename, interpretation)

    st.write("---")
    st.success(
        "**Conclusion (Business Requirement 1)**: \n\n"
        "The correlation study confirms that **tenure**, **MonthlyCharges**, "
        "**Contract type**, **TechSupport**, and **InternetService** are the "
        "strongest predictors of customer churn. These findings give the client "
        "clear, actionable insights into which customer segments are most at risk."
    )


def _display_plot(filename, interpretation):
    """Load and display a saved EDA plot with a text interpretation below it.

    Args:
        filename (str): Image filename inside EDA_DIR.
        interpretation (str): Text displayed beneath the plot.
    """
    path = os.path.join(EDA_DIR, filename)
    if os.path.exists(path):
        image = Image.open(path)
        st.image(image)
        st.write(interpretation)
    else:
        st.warning(
            f"Plot not found: `{path}`. "
            "Run Notebook 03_EDA.ipynb to generate EDA plots."
        )
