"""
Splitting Hotel Data into train and test 

The script reads the cleaned hotel booking dataset, splits it into
training and test sets (70/30 split), and saves the resulting datasets to the
output directory (processed data folder).
"""

import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command()
@click.option("--input-path")
@click.option("--train-output-path")
@click.option("--test-output-path")
def create_train_test_split(input_path, train_output_path, test_output_path):
    """
    Function splits cleaned dataset into train and test splits

    Parameters
    ----------
    input_path : str
        Path to cleaned dataset
    
    train_output_path : str
        Path to save the training data
    
    test_output_path : str
        Path to save the test data
         
    Returns
    --------
    None: Function does not return anything, saves training and test dataset to output path.

    Examples:
    ---------
    From command line, run:

    >>> python scripts/split_data.py --input-path data/processed/hotel_data_cleaned.csv --train-output-path data/processed/hotel_train_df.csv --test-output-path data/processed/hotel_test_df.csv
    """
    unsplit_data = pd.read_csv(input_path)
    train_df, test_df = train_test_split(unsplit_data, test_size=0.3, random_state=123)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    click.echo(f"Training data saved to: {train_output_path}")
    click.echo(f"Test data saved to: {test_output_path}")

if __name__ == "__main__":
    create_train_test_split()