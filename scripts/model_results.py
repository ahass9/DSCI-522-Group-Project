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
from src.evaluation_metrics import evaluation_metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay


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

    Parameters
    ----------
    x_train_path : str
        Path to the transformed training features CSV file.
    x_test_path : str
        Path to the transformed test features CSV file.
    y_train_path : str
        Path to the training target values CSV file.
    y_test_path : str
        Path to the test target values CSV file.
    metrics_output_path : str
        Path to save the test accuracy CSV file.
    cm_csv_output_path : str
        Path to save the confusion matrix CSV file.
    cm_figure_output_path : str
        Path to save the confusion matrix PNG figure.
    roc_figure_output_path : str
        Path to save the ROC curve PNG figure.

    Returns
    -------
    None
        Function does not return anything, it only saves metrics and figures
        to the specified output paths.

    Examples
    --------
    From the command line, run:

    python scripts/model_results.py \
        --x-train-path data/processed/X_train_transformed.csv \
        --x-test-path data/processed/X_test_transformed.csv \
        --y-train-path data/processed/y_train.csv \
        --y-test-path data/processed/y_test.csv \
        --metrics-output-path results/tables/test_accuracy.csv \
        --cm-csv-output-path results/tables/confusion_matrix.csv \
        --cm-figure-output-path results/figures/confusion_matrix_knn.png \
        --roc-figure-output-path results/figures/roc_curve_knn.png
    """

    def ensure_dir(path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    # ----- Load data -----
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    # ----- Define and fit KNN (k=15) on training data -----
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)

    # ----- Use evaluation_metrics to compute metrics on the test set -----
    metrics = evaluation_metrics(knn, X_test, y_test)

    test_accuracy = metrics["accuracy"]
    cm = metrics["confusion_matrix"]
    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    auc = metrics["auc"]
    # y_pred and y_proba are available too if you ever need them:
    # y_pred = metrics["y_pred"]
    # y_proba = metrics["y_proba"]

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