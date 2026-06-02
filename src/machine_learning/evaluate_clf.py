import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def confusion_matrix_and_report(X, y, pipeline, label_map):
    """
    Print classification report and display confusion matrix
    for a given dataset split and fitted pipeline.
    """
    y_pred = pipeline.predict(X)

    print("Classification Report")
    print(classification_report(y, y_pred, target_names=label_map))

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_map
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return fig


def clf_performance(X_train, y_train, X_test, y_test, pipeline, label_map):
    """Display model performance on both train and test splits."""
    print("=" * 50)
    print("Train Set")
    print("=" * 50)
    confusion_matrix_and_report(X_train, y_train, pipeline, label_map)

    print("=" * 50)
    print("Test Set")
    print("=" * 50)
    fig = confusion_matrix_and_report(X_test, y_test, pipeline, label_map)

    return fig
