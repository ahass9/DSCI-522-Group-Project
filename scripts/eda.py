"""
Exploratory Data Analysis

This script examines the distributions of different features across the classes of our target variable,
is_canceled. The function also creates a correlation matrix to examine whether nay of our numeric 
features are highly correlated.
"""

import pandas as pd
import click
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import sys
import seaborn as sns
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eda_barplot import create_barplot

alt.data_transformers.disable_max_rows()

@click.command()
@click.option("--input-path")
@click.option("--figure-prefix")
def create_eda_plot(input_path, figure_prefix):
    """
    Function generates plots that assist with EDA and examining feature distributions

    Parameters
    ----------
    input_path : str
        Path to training dataset
    
    figure_prefix: str
        Path to output folder where png image will be saved including filename prefix
         
    Returns
    --------
    None: Function does not return anything, generates and saves EDA plots to output folder with given prefix.

    Examples:
    ---------
    From command line, run:

    >>> python scripts/eda.py --input-path data/processed/hotel_train_df.csv --figure-prefix results/figures/eda
    """

    data = pd.read_csv(input_path) #training data only

    from src.corr_matrix import compute_numeric_correlation

    corr = compute_numeric_correlation(
        data,
        method="pearson",
        annot=False
    )
    
    plt.savefig(f"{figure_prefix}_correlation_matrix.png")
    plt.close()
    click.echo("Correlation matrix of numerical features saved!")

    create_barplot(data,x_col="is_canceled",output_path=f"{figure_prefix}_target_var_distribution.png")
    click.echo(f" Bar chart of target variable (hotel cancellations) counts  saved!")

    create_barplot(data,x_col="hotel",color_col="is_canceled",output_path=f"{figure_prefix}_hotel_vs_cancellations.png")
    click.echo('Bar Chart of distribution of cancellations across hotel types saved!')

    lead_time_plot = alt.Chart(data).transform_density(
    'lead_time',
        groupby= ['is_canceled'],
        as_=['lead_time', 'density'],
        ).mark_area().encode(
        x="lead_time:Q",
        y='density:Q',
        color='is_canceled:N'
        )

    lead_time_plot.save(f"{figure_prefix}_lead_time_density.png")
    click.echo('Lead time density plot across cancellation classes saved!')

    create_barplot(data,x_col="is_repeated_guest",color_col="is_canceled",output_path=f"{figure_prefix}_repeated_guest_vs_cancellations_count.png")
    click.echo('Bar chart of repeat guest counts vs cancellations saved!')

    create_barplot(data,x_col="deposit_type",color_col="is_canceled",output_path=f"{figure_prefix}_deposit_type_vs_cancellations_count.png")
    click.echo("Bar chart of deposit type vs cancellations saved!")


    create_barplot(data,x_col="reserved_room_type",color_col="is_canceled",output_path=f"{figure_prefix}_reserved_room_type_vs_cancellations.png")
    click.echo("Bar chart of reserved room type vs cancellations saved!")

if __name__ == '__main__':
    create_eda_plot()

