"""
Feature Preprocessing

This script loads the train and test data, applies feature preprocessing
which includes scaling for numeric features and onehotencoding for categorical and binary features.
It saves transformed feature and target training and test sets separately, to be used later
for model fitting evaluation.
"""

import click
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


@click.command()
@click.option("--train-input-path")
@click.option("--test-input-path")
@click.option("--x-train-transformed-output-path")
@click.option("--x-test-transformed-output-path")
@click.option("--y-train-output-path")
@click.option("--y-test-output-path")
@click.option("--preprocessor-output-path")
def preprocess_features(train_input_path,test_input_path,x_train_transformed_output_path,x_test_transformed_output_path,y_train_output_path,y_test_output_path,preprocessor_output_path):
    """
    Function preprocesses train/test data for model training and evaluation.

    Parameters
    ----------
    train_input_path : str
        path to training dataset.
    test_input_path : str
        path to test dataset.
    x_train_transformed_output_path : str
        path to save transformed training features.
    x_test_transformed_output_path : str
        path to save transformed testing features.
    y_train_output_path : str
        path to save the training target values.
    y_test_output_path : str
        path to save the testing target values.
    preprocessor_output_path : str
        path to save the fitted preprocessing object (pickle file).
    
    Returns
    --------
    None: Function does not return anything, generates and saves transformed features and raw target
    files separately.

    Examples:
    ---------
    From command line, run:

    >>> python scripts/feature_preprocessing.py \
    --train-input-path data/processed/hotel_train_df.csv \
    --test-input-path data/processed/hotel_test_df.csv \
    --x-train-transformed-output-path data/processed/X_train_transformed.csv \
    --x-test-transformed-output-path data/processed/X_test_transformed.csv \
    --y-train-output-path data/processed/y_train.csv \
    --y-test-output-path data/processed/y_test.csv \
    --preprocessor-output-path results/models/preprocessor.pkl
    """
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    X_train = train_df.drop(columns=["is_canceled"])
    y_train = train_df["is_canceled"]
    X_test = test_df.drop(columns=["is_canceled"])
    y_test = test_df["is_canceled"]

    # Feature groups
    num_features = ["lead_time", "previous_cancellations", 'previous_bookings_not_canceled', 'booking_changes', 'total_of_special_requests']
    cat_features = ["reserved_room_type", "deposit_type"]
    binary_features = ['is_repeated_guest']

    # Defining transformations in column transformer
    preprocessor = make_column_transformer(
        (StandardScaler(), num_features),
        ("passthrough", binary_features),
        (OneHotEncoder(handle_unknown="ignore"), cat_features))

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    pd.DataFrame(X_train_processed).to_csv(x_train_transformed_output_path, index=False)
    pd.DataFrame(X_test_processed).to_csv(x_test_transformed_output_path, index=False)

    click.echo(f"Preprocessed and transformed training and test features saved!")

    y_train.to_csv(y_train_output_path, index=False)
    y_test.to_csv(y_test_output_path, index=False)

    click.echo(f"Preprocessed and transformed targets saved!")

    #saving preprocessor incase needed for later us
    with open(preprocessor_output_path, "wb") as f:
        pickle.dump(preprocessor, f)

    click.echo(f"Preprocessor saved!")


if __name__ == "__main__":
    preprocess_features()