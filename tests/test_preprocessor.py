'''
Script to test the preprocessor function.

Run "pytest -v" in root of project to run the tests
'''

import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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

def test_preprocessor_empty_features():
    '''
    Check for error handling in empty feature lists
    '''
    with pytest.raises(ValueError, match="At least one feature type must be provided"):
        create_preprocessor([], [], [])

def test_preprocessor_invalid_feature_type():
    '''
    Check for invalid feature type (not a list)
    '''
    with pytest.raises(TypeError, match="num_features must be a list of strings"):
        create_preprocessor("age", ["city"], ["is_member"])


def test_preprocessor_invalid_feature_entries():
    '''
    Check for invalid feature entries (non-string)
    '''
    with pytest.raises(TypeError, match="All entries in cat_features must be strings"):
        create_preprocessor(["age"], [123], ["is_member"])


if __name__ == "__main__":
    test_preprocessor_numeric_scaling()
    test_preprocessor_handles_unknown_category()
    print("All tests ran successfully!")
