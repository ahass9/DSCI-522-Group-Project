"""
Creating a barplot for a categorical variable

This function takes a dataframe, a column name corresponding to a categorical variable, returning
an altair bar chart showing the count of observations. It also takes an optional color column to be 
mapped onto the color encoding if needed.
"""

import os
import pandas as pd
import altair as alt

def create_barplot(df, x_col, color_col=None, x_title=None,
    color_title=None,
    title=None, output_path=None):
    """
    Creates a categorical bar plot using Altair.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    x_col : str
        Categorical column to map to x-axis.
    color_col : str, optional
        Column to map to color encoding.
    x_title : str, optional
        Custom x-axis title.
    color_title : str, optional
        Custom legend title.
    title : str, optional
        Plot title.
    output_path : str, optional
        If provided, save the chart to this path.

    Returns
    -------
    alt.Chart
        Altair bar chart object.

    Raises
    ------
    ValueError
        If required columns are not present in the dataframe.
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in dataframe.")
    
    if color_col and color_col not in df.columns:
        raise ValueError(f"Column '{color_col}' not found in dataframe.")

    encoding = {"x": alt.X(x_col,type="nominal",title=x_title if x_title else x_col),
                "y": alt.Y("count()", aggregate="count")}
    
    if color_col:
        encoding["color"] = alt.Color(color_col,type="nominal",title=color_title if color_title else color_col)
    
    chart = (alt.Chart(df)
            .mark_bar()
            .encode(**encoding)
            .properties(width=300, height=250, title=title))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        chart.save(output_path)

    return chart