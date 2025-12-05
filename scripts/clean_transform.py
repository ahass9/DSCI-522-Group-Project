"""
Clean Hotel Dataset

This script filters the hotel bookings dataset for observations from the year 2017 
and selects only columns relevant to our focused research question. It also removes 
rows with missing values and saves the processed csv.
"""

import pandas as pd
import click
import os

cols_of_interest = [
    "hotel",
    "is_canceled",
    "lead_time",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "total_of_special_requests",
    "is_repeated_guest",
    "deposit_type",
    "reserved_room_type"
]

@click.command()
@click.option("--input-path")
@click.option("--output-path")

def clean_data(input_path, output_path):
    """
    Clean hotel dataset and filtering for desired columns and rows

    Parameters
    ----------
    input_path : str
        Path to unprocessed data from local data folder
    
    output_path : str
        File path to save the cleaned data to.
    
    Returns
    --------
    None: Function does not return anything, simply saves cleaned dataset to output path.

    Examples:
    ---------
    From command line, run:

    >>> python scripts/clean_transform.py --input-path data/raw/hotel_data.csv --output-path data/processed/hotel_data_cleaned.csv
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    hotel_data = pd.read_csv(input_path)
    
    #filtering for year 2017 only (based on research question constraints)
    data_2017 = hotel_data[hotel_data["arrival_date_year"] == 2017]

    #selecting our coloumns of interest and dropping nulls
    hotel_data_cleaned = data_2017[cols_of_interest].dropna()

    hotel_data_cleaned.to_csv(output_path, index=False)
    click.echo(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    clean_data()
    

