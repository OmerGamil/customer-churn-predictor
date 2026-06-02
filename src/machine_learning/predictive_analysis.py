"""
predictive_analysis.py — Live prediction helper for the churn dashboard.

Loads the saved optimal threshold and applies it to the fitted pipeline's
predict_proba output so the decision boundary matches what was tuned
during training.
"""

import joblib
import os

THRESHOLD_PATH = "outputs/ml_pipeline/predict_churn/v1/optimal_threshold.pkl"


def predict_churn(X_live, pipeline):
    """
    Run the fitted ML pipeline on a live input DataFrame and return
    the predicted class label (0 = No churn, 1 = Churn).

    Uses the saved optimal threshold (tuned to meet the ≥75% recall
    business requirement) instead of the default 0.5 cutoff.
    """
    if os.path.exists(THRESHOLD_PATH):
        threshold = joblib.load(THRESHOLD_PATH)
    else:
        threshold = 0.5

    proba = pipeline.predict_proba(X_live)[:, 1]
    prediction = (proba >= threshold).astype(int)
    return int(prediction[0])
