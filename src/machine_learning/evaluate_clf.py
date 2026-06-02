import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def confusion_matrix_and_report(X, y, pipeline, label_map, threshold=0.5):
    """
    Display a styled confusion matrix and classification report
    for a given dataset split and fitted pipeline.

    Uses predict_proba with a configurable threshold so the same
    function works whether the default 0.5 or a tuned threshold is used.
    """
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
    print(classification_report(y, prediction, target_names=label_map), "\n")


def clf_performance(X_train, y_train, X_val, y_val, X_test, y_test,
                    pipeline, label_map, threshold=0.5):
    """
    Evaluate pipeline performance across train, validation, and test splits.
    Prints confusion matrix and classification report for each split.
    """
    print("#### Train Set ####\n")
    confusion_matrix_and_report(X_train, y_train, pipeline, label_map, threshold)

    print("#### Validation Set ####\n")
    confusion_matrix_and_report(X_val, y_val, pipeline, label_map, threshold)

    print("#### Test Set ####\n")
    confusion_matrix_and_report(X_test, y_test, pipeline, label_map, threshold)
