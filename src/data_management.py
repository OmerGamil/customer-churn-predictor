"""
data_management.py — Utility functions for loading data and model artifacts.

Used by the Streamlit dashboard pages to load the cleaned dataset
and the serialised ML pipeline / threshold files.
"""

import pandas as pd
import joblib


def load_churn_data():
    """
    Load the cleaned Telco Customer Churn dataset from disk.

    Returns:
        pd.DataFrame: Cleaned dataset with 20 feature columns and Churn target.
    """
    return pd.read_csv(
        "outputs/datasets/collection/telco_churn_cleaned.csv"
    )


def load_pkl_file(file_path):
    """
    Load a joblib-serialised object from disk.

    Used to load both the ML pipeline (clf_pipeline.pkl) and the
    optimal decision threshold (optimal_threshold.pkl).

    Args:
        file_path (str): Path to the .pkl file.

    Returns:
        object: The deserialised Python object (pipeline, float, etc.).
    """
    return joblib.load(filename=file_path)
