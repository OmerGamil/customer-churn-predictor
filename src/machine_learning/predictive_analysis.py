def predict_churn(X_live, pipeline):
    """
    Run the fitted ML pipeline on a live input DataFrame and
    return the predicted class label (0 = No churn, 1 = Churn).
    """
    prediction = pipeline.predict(X_live)
    return int(prediction[0])
