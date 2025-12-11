"""
Unit tests for the create_barplot function.

The tests verify that the create_barplot function correctly generates
Altair categorical bar plots under expected conditions, handles edge cases
and raises clear errors for invalid inputs.
"""

import pytest
import pandas as pd
import altair as alt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eda_barplot import create_barplot

#expected cases
def test_create_barplot_basic_categorical():
    """
    Basic bar plot with a single categorical variable.
    """
    df = pd.DataFrame({"category": ["A", "A", "B", "C", "C"]})

    chart = create_barplot(df, x_col="category")

    assert isinstance(chart, alt.Chart)
    assert chart.mark == "bar"
    assert chart.encoding.x.shorthand == "category"
    assert chart.encoding.x.to_dict()["type"] == "nominal"
    assert chart.encoding.y.to_dict()["aggregate"] == "count"

def test_create_barplot_with_color_grouping():
    """
    Grouped bar plot with a categorical color variable.
    """
    df = pd.DataFrame({
        "group": ["X", "X", "Y", "Y"],
        "label": ["yes", "no", "yes", "no"]
    })

    chart = create_barplot(df, x_col="group", color_col="label")

    assert chart.mark == "bar"
    assert chart.encoding.x.shorthand == "group"
    assert chart.encoding.x.to_dict()["type"] == "nominal"
    assert chart.encoding.color.shorthand == "label"
    assert chart.encoding.color.to_dict()["type"] == "nominal"
    assert chart.encoding.y.to_dict()["aggregate"] == "count"

#edge cases
def test_create_barplot_single_unique_value():
    """
    Edge case where the categorical column has only one unique value.
    """
    df = pd.DataFrame({"type": ["only_one", "only_one", "only_one"]})

    chart = create_barplot(df, x_col="type")

    assert chart.encoding.x.shorthand == "type"
    assert chart.encoding.x.to_dict()["type"] == "nominal"
    assert chart.encoding.y.to_dict()["aggregate"] == "count"

def test_create_barplot_multiple_rows_same_category_and_color():
    """
    Edge case with repeated category + color combinations.
    """
    df = pd.DataFrame({
        "category": ["A", "A", "A"],
        "group": ["G1", "G1", "G1"]
    })

    chart = create_barplot(df, x_col="category", color_col="group")

    assert chart.encoding.x.shorthand == "category"
    assert chart.encoding.color.shorthand == "group"
    assert chart.encoding.y.to_dict()["aggregate"] == "count"

#error cases
def test_create_barplot_empty_dataframe():
    """
    Tests if empty dataframe fails with a clear error message.
    """
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Input dataframe is empty."):
        create_barplot(df, x_col="anything")


def test_create_barplot_missing_x_column():
    """
    Raises error when x column not present in dataframe.
    """
    df = pd.DataFrame({
        "col1": ["A", "B", "C"]
    })

    with pytest.raises(ValueError, match="Column 'missing' not found in dataframe."):
        create_barplot(df, x_col="missing")


def test_create_barplot_missing_color_column():
    """
    Raises error when color column not present in dataframe.
    """
    df = pd.DataFrame({
        "category": ["A", "B", "C"]
    })

    with pytest.raises(ValueError, match="Column 'color' not found in dataframe."):
        create_barplot(df, x_col="category", color_col="color")