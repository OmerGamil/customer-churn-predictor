import pandas as pd
import joblib


def load_churn_data():
    """Load the cleaned churn dataset."""
    return pd.read_csv(
        "outputs/datasets/collection/telco_churn_cleaned.csv"
    )


def load_pkl_file(file_path):
    """Load a joblib-serialised pipeline or model from disk."""
    return joblib.load(filename=file_path)
