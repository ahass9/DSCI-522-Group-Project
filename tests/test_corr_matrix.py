"""
Unit tests for compute_numeric_correlation function

These tests verify that the function correctly produces a correlation matrix of all 
numeric features in a given dataframe. It checks that non numeric columns are ignored 
and that at least 2 numeric features are present for the correlation
"""

import pandas as pd
import numpy as np
import pytest

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.corr_matrix import compute_numeric_correlation


# EXPECTED USE CASE TESTS

def test_returns_correlation_dataframe_for_numeric_input():
    """
    returns a pandas DataFrame with correct shape
    when input contains only numeric columns.
    """
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [4, 3, 2, 1]
    })

    corr = compute_numeric_correlation(df)

    assert isinstance(corr, pd.DataFrame)
    assert corr.shape == (2, 2)


def test_ignores_non_numeric_columns():
    """
    Function checks that Non-numeric columns are excluded from the correlation matrix.
    """
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [3, 2, 1],
        "category": ["a", "b", "c"]
    })

    corr = compute_numeric_correlation(df)

    assert list(corr.columns) == ["x", "y"]


# EDGE CASE TESTS

def test_exactly_two_numeric_columns():
    """
    Checks that function works correctly when exactly two numeric columns are provided.
    """
    df = pd.DataFrame({
        "m": [1, 2, 3],
        "n": [3, 2, 1]
    })

    corr = compute_numeric_correlation(df)

    assert corr.shape == (2, 2)


def test_constant_column_produces_nan_correlation():
    """
    Checks that function works correctly when exactly two numeric columns are provided.
    """
    df = pd.DataFrame({
        "a": [1, 1, 1, 1],
        "b": [1, 2, 3, 4]
    })

    corr = compute_numeric_correlation(df)

    assert np.isnan(corr.loc["a", "b"])


# ERROR CASE TESTS

def test_non_dataframe_input_raises_type_error():
    """
    Checks that passing non dataframe input produces a Typeerror
    """
    with pytest.raises(TypeError):
        compute_numeric_correlation([1, 2, 3])


def test_less_than_two_numeric_columns_raises_value_error():
    """
    Function produces ValueError when fewer than two numeric columns are passed
    """
    df = pd.DataFrame({
        "only_numeric": [1, 2, 3],
        "category": ["a", "b", "c"]
    })

    with pytest.raises(ValueError):
        compute_numeric_correlation(df)
