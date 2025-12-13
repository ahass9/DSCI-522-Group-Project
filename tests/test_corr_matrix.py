import pandas as pd
import numpy as np
import pytest

from src.eda import compute_numeric_correlation


# EXPECTED USE CASE TESTS

def test_returns_correlation_dataframe_for_numeric_input():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [4, 3, 2, 1]
    })

    corr = compute_numeric_correlation(df)

    assert isinstance(corr, pd.DataFrame)
    assert corr.shape == (2, 2)


def test_ignores_non_numeric_columns():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [3, 2, 1],
        "category": ["a", "b", "c"]
    })

    corr = compute_numeric_correlation(df)

    assert list(corr.columns) == ["x", "y"]


# EDGE CASE TESTS

def test_exactly_two_numeric_columns():
    df = pd.DataFrame({
        "m": [1, 2, 3],
        "n": [3, 2, 1]
    })

    corr = compute_numeric_correlation(df)

    assert corr.shape == (2, 2)


def test_constant_column_produces_nan_correlation():
    df = pd.DataFrame({
        "a": [1, 1, 1, 1],
        "b": [1, 2, 3, 4]
    })

    corr = compute_numeric_correlation(df)

    assert np.isnan(corr.loc["a", "b"])


# ERROR CASE TESTS

def test_non_dataframe_input_raises_type_error():
    with pytest.raises(TypeError):
        compute_numeric_correlation([1, 2, 3])


def test_less_than_two_numeric_columns_raises_value_error():
    df = pd.DataFrame({
        "only_numeric": [1, 2, 3],
        "category": ["a", "b", "c"]
    })

    with pytest.raises(ValueError):
        compute_numeric_correlation(df)
