"""
app.py — Entry point for the Customer Churn Predictor Streamlit dashboard.

Registers all five pages and launches the multipage app.
"""

from app_pages.multipage import MultiPage
from app_pages.page_summary import page_summary_body
from app_pages.page_eda import page_eda_body
from app_pages.page_hypothesis import page_hypothesis_body
from app_pages.page_model_performance import page_model_performance_body
from app_pages.page_churn_predictor import page_churn_predictor_body

# Initialise the app with a title shown in the browser tab
app = MultiPage(app_name="Customer Churn Predictor")

# Register pages in the order they appear in the sidebar
app.add_page("Project Summary", page_summary_body)
app.add_page("EDA & Correlation Study", page_eda_body)
app.add_page("Project Hypothesis", page_hypothesis_body)
app.add_page("Model Performance", page_model_performance_body)
app.add_page("Churn Predictor", page_churn_predictor_body)

app.run()
