import streamlit as st


def page_hypothesis_body():
    st.write("## Project Hypothesis and Validation")

    st.write("---")
    st.write("### Hypothesis 1: Contract Type and Churn")
    st.info(
        "**Hypothesis**: Customers with month-to-month contracts are significantly "
        "more likely to churn than customers on one-year or two-year contracts.\n\n"
        "**Rationale**: Month-to-month customers have no long-term commitment, "
        "making it easier for them to cancel at any time."
    )
    st.success(
        "**Validation**: CONFIRMED \n\n"
        "The EDA correlation study and bar plots confirm that month-to-month "
        "contract holders churn at a substantially higher rate. "
        "Contract type is among the strongest predictors of churn in the dataset."
    )

    st.write("---")
    st.write("### Hypothesis 2: Monthly Charges and Churn")
    st.info(
        "**Hypothesis**: Customers with higher monthly charges are more likely "
        "to churn than customers paying lower amounts.\n\n"
        "**Rationale**: Customers who perceive the service as expensive relative "
        "to its value are more motivated to seek cheaper alternatives."
    )
    st.success(
        "**Validation**: CONFIRMED \n\n"
        "The correlation heatmap and distribution plots show that churned customers "
        "have, on average, higher monthly charges. MonthlyCharges is one of the top "
        "numeric features positively correlated with the Churn target variable."
    )

    st.write("---")
    st.write("### Hypothesis 3: Tenure and Churn")
    st.info(
        "**Hypothesis**: Customers who have been with the company for a shorter "
        "time (low tenure) are more likely to churn.\n\n"
        "**Rationale**: New customers have not yet built loyalty and may still be "
        "evaluating competing providers."
    )
    st.success(
        "**Validation**: CONFIRMED \n\n"
        "Tenure shows a strong negative correlation with Churn. Customers who churn "
        "tend to have significantly lower average tenure, confirming that early "
        "retention efforts are especially important."
    )
