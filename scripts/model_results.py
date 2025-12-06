"""
Model Training and Evaluation (KNN)

This script loads preprocessed training and test feature sets along with their
corresponding targets, fits a K-Nearest Neighbors (KNN) classifier, evaluates
its performance on the test set, and saves:

1. Test accuracy (as a CSV file).
2. Confusion matrix (as a CSV file).
3. Confusion matrix plot (as a PNG file).
4. ROC curve plot (as a PNG file).
"""

import os
import click
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


@click.command()
@click.option("--x-train-path", required=True)
@click.option("--x-test-path", required=True)
@click.option("--y-train-path", required=True)
@click.option("--y-test-path", required=True)
@click.option("--metrics-output-path", required=True)
@click.option("--cm-csv-output-path", required=True)
@click.option("--cm-figure-output-path", required=True)
@click.option("--roc-figure-output-path", required=True)
def model_results(
    x_train_path,
    x_test_path,
    y_train_path,
    y_test_path,
    metrics_output_path,
    cm_csv_output_path,
    cm_figure_output_path,
    roc_figure_output_path,
):
    """
    Train and evaluate a KNN model on preprocessed hotel cancellation data.
    """

    def ensure_dir(path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    # ----- Load data -----
    X_train = pd.read_csv(x_train_path).to_numpy()
    X_test = pd.read_csv(x_test_path).to_numpy()

    y_train = pd.read_csv(y_train_path).squeeze("columns").to_numpy()
    y_test = pd.read_csv(y_test_path).squeeze("columns").to_numpy()

    # ----- Define and fit KNN (k=15) -----
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)

    # ----- Predictions and metrics -----
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    # ----- Save test accuracy as CSV -----
    ensure_dir(metrics_output_path)
    metrics_df = pd.DataFrame(
        {"metric": ["test_accuracy"], "value": [test_accuracy]}
    )
    metrics_df.to_csv(metrics_output_path, index=False)

    # ----- Save confusion matrix as CSV -----
    ensure_dir(cm_csv_output_path)
    labels = ["not_canceled", "canceled"]
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )
    cm_df.to_csv(cm_csv_output_path)

    # ----- Save confusion matrix plot as PNG -----
    ensure_dir(cm_figure_output_path)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax_cm, values_format="d", cmap="Blues", colorbar=False)
    ax_cm.set_title("Confusion Matrix - KNN (Test Set)")
    fig_cm.tight_layout()
    fig_cm.savefig(cm_figure_output_path)
    plt.close(fig_cm)

    # ----- Save ROC curve plot as PNG -----
    ensure_dir(roc_figure_output_path)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")

    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve - KNN Model")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)

    fig_roc.tight_layout()
    fig_roc.savefig(roc_figure_output_path)
    plt.close(fig_roc)


if __name__ == "__main__":
    model_results()