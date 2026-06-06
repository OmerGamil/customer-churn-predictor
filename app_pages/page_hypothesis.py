"""
page_hypothesis.py — Streamlit page: Project Hypothesis and Validation.

States three project hypotheses (contract type, monthly charges, tenure)
and validates each one using statistical evidence from the EDA notebook.
Addresses Merit criterion 2.3 (statistical validation of hypotheses).
"""

import streamlit as st


def page_hypothesis_body():
    """Render the Project Hypothesis and Validation page content."""

    st.write("## Project Hypothesis and Validation")

    st.write(
        "Three hypotheses were formulated before the exploratory data "
        "analysis. Each is validated below using correlation statistics "
        "and churn-rate comparisons derived from the cleaned Telco "
        "Customer Churn dataset."
    )

    # --- Hypothesis 1 ---
    st.write("---")
    st.write("### Hypothesis 1: Contract Type and Churn")
    st.info(
        "**Hypothesis**: Customers with month-to-month contracts are "
        "significantly more likely to churn than customers on one-year "
        "or two-year contracts.\n\n"
        "**Rationale**: Month-to-month customers have no long-term "
        "commitment, making it frictionless to cancel at any time. "
        "Annual contracts create a financial and procedural barrier "
        "to leaving."
    )
    st.write("**Statistical evidence**:")
    st.table({
        "Contract Type": [
            "Month-to-month", "One year", "Two year"
        ],
        "Churn Rate": ["~42%", "~11%", "~3%"],
    })
    st.write(
        "Month-to-month customers churn at roughly **14× the rate** of "
        "two-year contract customers. Contract type is the strongest "
        "categorical predictor of churn in the dataset."
    )
    st.success(
        "**Outcome: CONFIRMED** \n\n"
        "The EDA bar chart and churn-rate table both confirm the "
        "hypothesis. Contract type is one of the top features in the "
        "trained GradientBoostingClassifier."
    )

    # --- Hypothesis 2 ---
    st.write("---")
    st.write("### Hypothesis 2: Monthly Charges and Churn")
    st.info(
        "**Hypothesis**: Customers with higher monthly charges are more "
        "likely to churn than customers paying lower amounts.\n\n"
        "**Rationale**: Customers who perceive the service as expensive "
        "relative to its value are more motivated to seek cheaper "
        "alternatives."
    )
    st.write("**Statistical evidence**:")
    st.table({
        "Group": ["Churned customers", "Retained customers"],
        "Mean Monthly Charges": ["~$74", "~$61"],
        "Pearson r (vs Churn)": ["+0.19", "—"],
    })
    st.write(
        "The Pearson correlation between MonthlyCharges and the binary "
        "Churn target is **r = +0.19**, indicating a positive "
        "relationship. Churned customers pay on average $13 more per "
        "month than retained customers."
    )
    st.success(
        "**Outcome: CONFIRMED** \n\n"
        "MonthlyCharges shows a positive correlation with Churn. The "
        "distribution plot in the EDA page confirms that churners cluster "
        "at higher monthly charge values."
    )

    # --- Hypothesis 3 ---
    st.write("---")
    st.write("### Hypothesis 3: Tenure and Churn")
    st.info(
        "**Hypothesis**: Customers who have been with the company for a "
        "shorter time (low tenure) are more likely to churn.\n\n"
        "**Rationale**: New customers have not yet built loyalty and may "
        "still be evaluating competing providers. Long-tenured customers "
        "are more embedded in the service."
    )
    st.write("**Statistical evidence**:")
    st.table({
        "Group": ["Churned customers", "Retained customers"],
        "Mean Tenure (months)": ["~18", "~38"],
        "Pearson r (vs Churn)": ["−0.35", "—"],
    })
    st.write(
        "Tenure has a **Pearson r = −0.35** with Churn — the "
        "strongest individual numeric correlation in the dataset. "
        "Churned customers have a mean tenure of 18 months, less than "
        "half that of retained customers (38 months)."
    )
    st.success(
        "**Outcome: CONFIRMED** \n\n"
        "Tenure is the strongest individual numeric predictor of churn. "
        "Early-stage customers (low tenure) represent the highest-risk "
        "segment for the retention team."
    )
