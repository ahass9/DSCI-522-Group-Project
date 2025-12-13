"""
Correlation matrix for numeric features

This function uses the numeric features in a given dataframe and generates a pairwise
correlation matrix plot.
"""


import pandas as pd
import matplotlib.pyplot as plt


def compute_numeric_correlation(
    df: pd.DataFrame,
    method: str = "pearson",
    figsize: tuple = (10, 8),
    cmap: str = "RdYlGn",
    vmin: float = -1.0,
    vmax: float = 1.0,
    output_path = None) -> pd.DataFrame:
    """
    Compute and plot a correlation matrix for numeric features in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    method : str, default="pearson"
        Correlation method ("pearson", "spearman", "kendall")
    figsize : tuple, default=(10, 8)
        Figure size for the correlation plot
    annot : bool, default=False
        Whether to annotate correlation values on the heatmap

    Returns
    -------
    pd.DataFrame
        Correlation matrix of numeric features
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ValueError("DataFrame must contain at least two numeric columns")

    corr = numeric_df.corr(method=method)

    # Plot
    if output_path is not None:
        plt.figure(figsize=figsize)
        im = plt.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im)
        plt.title("Correlation Matrix")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    return corr
