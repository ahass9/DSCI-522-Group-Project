from typing import Any, Dict

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)


def evaluation_metrics(model: Any, X_test, y_test) -> Dict[str, Any]:
    """
    Fit a classification model on test data and compute evaluation metrics.

    This function fits the given model on X_test and y_test, then uses the
    fitted model to predict on X_test and calculate common metrics for
    binary classification.

    Parameters
    ----------
    model : object
        A classifier that implements 'fit', 'predict', and 'predict_proba'.
    X_test : pandas.DataFrame or numpy.ndarray
        Feature data used for fitting and evaluating the model.
    y_test : pandas.Series, numpy.ndarray or list
        True labels for X_test.

    Returns
    -------
    dict
        A dictionary with:
        - "accuracy" : float
            Accuracy score on X_test.
        - "confusion_matrix" : numpy.ndarray
            Confusion matrix for predictions on X_test.
        - "fpr" : numpy.ndarray
            False positive rates for the ROC curve.
        - "tpr" : numpy.ndarray
            True positive rates for the ROC curve.
        - "auc" : float
            Area under the ROC curve.
        - "y_pred" : numpy.ndarray
            Predicted class labels for X_test.
        - "y_proba" : numpy.ndarray
            Predicted probabilities for the positive class.

    Raises
    ------
    ValueError
        If X_test or y_test are None, empty, have different lengths, or if the
        model does not implement the required methods.
    """
    
    if X_test is None or y_test is None:
      raise ValueError("X_test and y_test must not be None.")

    if len(X_test) == 0:
      raise ValueError("X_test is empty.")

    if len(y_test) == 0:
      raise ValueError("y_test is empty.")

    if len(X_test) != len(y_test):
      raise ValueError("X_test and y_test must have the same number of rows.")

    if not hasattr(model, "fit") or not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
     raise ValueError(
       "Model must implement 'fit', 'predict', and 'predict_proba' methods."
       )

    model.fit(X_test, y_test)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }