"""
Downloading Raw Hotel Data

This script reads hotel booking data from a given input path
(URL or local CSV file) and saves a cleaned subset of columns to a specified output path 
(data/raw folder)
"""

import pandas as pd
import click
import os

@click.command()
@click.option("--input-path")
@click.option("--output-path")

# Function for Downloading raw hotel dataset
def download_extract(input_path, output_path):
    """Hotel Data Downloader 
    Reads in data from url/file path and saves it to the output path (in the data/raw folder)

    Parameters
    ----------
    input_path : str
        Path to input data (URL or csv file).
    
    output_path : str
        File path to save the raw data to.
    
    Returns
    --------
    None: Function does not return anything, simply saves dataset to output path.

    Examples:
    ---------
    From command line, run:

    >>> python src/download_data.py --input-path https://exampleurl.com/data.csv --output-path data/raw/hotel_data.csv
    >>> python src/download_data.py --input-path data/hotel_booking.csv --output-path data/raw/hotel_data.csv
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hotel_data = pd.read_csv(input_path)

    hotel_data.to_csv(output_path, index=False)
    click.echo(f"Saved data to: {output_path}")

if __name__ == "__main__":
    download_extract()

