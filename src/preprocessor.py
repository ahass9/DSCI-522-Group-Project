"""
Preprocessor module

This file defines a function to create a preprocessing pipeline
that applies StandardScaler to numeric features, passthrough to binary features, and OneHotEncoder to categorical features.
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


def create_preprocessor(num_features, cat_features, binary_features):
    """
    Create a preprocessing pipeline for numeric, categorical, and binary features.

    Parameters
    ----------
    num_features : list of str
        Names of numeric columns to scale.
    cat_features : list of str
        Names of categorical columns to one-hot encode.
    binary_features : list of str
        Names of binary columns to passthrough.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        A fitted column transformer object ready for training or testing data.
    
    Raises
    ------
    ValueError
        If all feature lists are empty.
    TypeError
        If feature lists are not lists of strings.

        
    Examples
    --------
    >>> num_features = ["age", "income"]
    >>> cat_features = ["city"]
    >>> binary_features = ["is_member"]
    >>> preprocessor = create_preprocessor(num_features, cat_features, binary_features)
    """
    
    # Validation checks

    # Edge case of empty list
    if not num_features and not cat_features and not binary_features:
        raise ValueError("At least one feature type must be provided.")
    
    #Error cases: wrong types and wrong entries
    for features, name in [(num_features, "num_features"), (cat_features, "cat_features"), (binary_features, "binary_features"),]:
        if features is not None:
            if not isinstance(features, list):
                raise TypeError(f"{name} must be a list of strings.")
            if any(not isinstance(f, str) for f in features):
                raise TypeError(f"All entries in {name} must be strings.")
                
    preprocessor = make_column_transformer(
        (StandardScaler(), num_features),
        ("passthrough", binary_features),
        (OneHotEncoder(handle_unknown="ignore"), cat_features)
    )
    return preprocessor
