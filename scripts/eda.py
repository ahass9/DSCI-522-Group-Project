"""
Exploratory Data Analysis

This script examines the distributions of different features across the classes of our target variable,
is_canceled. The function also creates a correlation matrix to examine whether nay of our numeric 
features are highly correlated.
"""

import pandas as pd
import click
import os
import matplotlib.pyplot as plt
import altair as alt
import numpy as np

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

    corr = data.select_dtypes("number").corr()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.title("Correlation Matrix")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()

    plt.savefig(f"{figure_prefix}_correlation_matrix.png")
    plt.close()
    click.echo(f"Correlation matrix of numerical features saved!")

    target_dist = alt.Chart(data).mark_bar().encode(
        x='is_canceled:N',
        y='count()'
    ).properties(width=300, height=250)

    target_dist.save(f"{figure_prefix}_target_var_distribution.png")

    click.echo(f" Bar chart of target variable (hotel cancellations) counts  saved!")

    target_dist_hotel_type = alt.Chart(data).mark_bar().encode(
        x='hotel:N',
        y='count()',
        color='is_canceled:N'
    ).properties(width=300, height=250)

    target_dist_hotel_type.save(f"{figure_prefix}_hotel_vs_cancellations.png")
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

    repeated_guest = alt.Chart(data[['is_repeated_guest', 'deposit_type', 'reserved_room_type', 'is_canceled']]).mark_bar().encode(
        x='is_repeated_guest:N', 
        y='count()',
        color='is_canceled:N'
        ).properties(width=250)

    repeated_guest.save(f"{figure_prefix}_repeated_guest_vs_cancellations_count.png")
    click.echo('Bar chart of repeat guest counts vs cancellations saved!')

    deposit_type = alt.Chart(data).mark_bar().encode(
        x='deposit_type:N',
        y='count()',
        color='is_canceled:N'
        ).properties(width=250)
    
    deposit_type.save(f"{figure_prefix}_deposit_type_vs_cancellations_count.png")
    click.echo("Bar chart of deposit type vs cancellations saved!")


    room_type = alt.Chart(data).mark_bar().encode(
        x='reserved_room_type:N',
        y='count()',
        color='is_canceled:N'
        ).properties(width=250)
    
    room_type.save(f"{figure_prefix}_reserved_room_type_vs_cancellations.png")
    click.echo("Bar chart of reserved room type vs cancellations saved!")
    
if __name__ == '__main__':
    create_eda_plot()



