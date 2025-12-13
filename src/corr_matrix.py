import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_numeric_correlation(
    df: pd.DataFrame,
    method: str = "pearson",
    figsize: tuple = (10, 8),
    annot: bool = False
) -> pd.DataFrame:
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

    corr_matrix = numeric_df.corr(method=method)

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        square=True,
        annot=annot,
        fmt=".2f",
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f"{method.capitalize()} Correlation Matrix")
    plt.tight_layout()
    plt.show()

    return corr_matrix
