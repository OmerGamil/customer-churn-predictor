"""
evaluate_clf.py — CI-style evaluation helpers for classification pipelines.

Provides confusion_matrix_and_report and clf_performance, adapted from
the Code Institute helper functions for use with sklearn predict_proba
and a configurable decision threshold.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def confusion_matrix_and_report(X, y, pipeline, label_map, threshold=0.5):
    """
    Print a styled confusion matrix and classification report.

    Uses predict_proba so the threshold can be tuned independently of
    the model — important when the business requirement demands high
    recall.

    Args:
        X: Feature matrix (DataFrame or array).
        y: True labels.
        pipeline: Fitted sklearn/imblearn pipeline with predict_proba.
        label_map (list): Class names e.g. ['No Churn', 'Churn'].
        threshold (float): Probability cutoff for positive class.
    """
    # Get class probabilities and apply the threshold
    y_proba = pipeline.predict_proba(X)[:, 1]
    prediction = (y_proba >= threshold).astype(int)

    print("---  Confusion Matrix  ---")
    print(pd.DataFrame(
        confusion_matrix(y_true=y, y_pred=prediction),
        columns=["Actual " + sub for sub in label_map],
        index=["Predicted " + sub for sub in label_map]
    ))
    print("\n")

    print("---  Classification Report  ---")
    print(classification_report(y, prediction, target_names=label_map),
          "\n")


def clf_performance(X_train, y_train, X_val, y_val, X_test, y_test,
                    pipeline, label_map, threshold=0.5):
    """
    Evaluate a pipeline across train, validation, and test splits.

    Prints the confusion matrix and classification report for each
    split, making it easy to spot overfitting by comparing train vs
    val vs test.

    Args:
        X_train, y_train: Training split.
        X_val, y_val:     Validation split.
        X_test, y_test:   Test split.
        pipeline:         Fitted sklearn/imblearn pipeline.
        label_map (list): Class names e.g. ['No Churn', 'Churn'].
        threshold (float): Probability cutoff for the positive class.
    """
    print("#### Train Set #### \n")
    confusion_matrix_and_report(
        X_train, y_train, pipeline, label_map, threshold
    )

    print("#### Validation Set #### \n")
    confusion_matrix_and_report(
        X_val, y_val, pipeline, label_map, threshold
    )

    print("#### Test Set ####\n")
    confusion_matrix_and_report(
        X_test, y_test, pipeline, label_map, threshold
    )
