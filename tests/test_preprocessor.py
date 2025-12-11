'''
Script to test the preprocessor function.

Run "PYTHONPATH=. pytest -v" in root of project to run the tests
PYTHONPATH= is needed to get package from src folder
'''

import pytest
import pandas as pd
from src.preprocessor import create_preprocessor

def test_preprocessor_numeric_scaling():
    num_features = ["age"]
    cat_features = ["city"]
    binary_features = ["is_member"]

    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["A", "B", "A"],
        "is_member": [1, 0, 1]
    })

    preprocessor = create_preprocessor(num_features, cat_features, binary_features)
    transformed = preprocessor.fit_transform(df)

    # Check shape: Should be 3 rows, 1 scaled numeric + 1 binary + 2 OHE categories
    assert transformed.shape == (3, 4)


def test_preprocessor_handles_unknown_category():
    num_features = ["age"]
    cat_features = ["city"]
    binary_features = ["is_member"]

    train_df = pd.DataFrame({
        "age": [25, 35],
        "city": ["A", "B"],
        "is_member": [1, 0]
    })

    test_df = pd.DataFrame({
        "age": [45],
        "city": ["C"],  # unseen category
        "is_member": [1]
    })

    preprocessor = create_preprocessor(num_features, cat_features, binary_features)
    preprocessor.fit(train_df)
    transformed = preprocessor.transform(test_df)

    # Should still work with unseen category
    assert transformed.shape[1] == 4


if __name__ == "__main__":
    test_preprocessor_numeric_scaling()
    test_preprocessor_handles_unknown_category()
    print("All tests ran successfully!")
