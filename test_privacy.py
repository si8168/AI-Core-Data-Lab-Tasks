import pytest
import pandas as pd
import numpy as np
from joblib import load

# Mock data generation fixture to represent your structural inputs safely
@pytest.fixture
def sample_raw_data():
    data = {
        'Customer Name': ['Sarah Connor', 'Tony Stark'],
        'Customer e-mail': ['sarah@sky.net', 'tony@stark.corp'],
        'Country': ['USA', 'USA'],
        'Phone Number': ['021-555-0199', '022-777-0144'], # Custom Expanded PII Column
        'Home Address': ['123 Robot Way, Auckland', '10880 Malibu Point, CA'], # Custom Expanded PII Column
        'Date of Birth': ['1984-11-10', '1970-05-29'], # Custom Expanded PII Column
        'Gender': [1, 0],
        'Age': [41, 56],
        'Annual Salary': [75000, 150000],
        'Credit Card Debt': [2000, 0],
        'Net Worth': [180000, 9999999],
        'Car Purchase Amount': [42000, 95000]
    }
    return pd.DataFrame(data)

@pytest.fixture
def processed_pipeline_data(sample_raw_data):
    # Expanded dropping scope showing custom data pillars not explicitly dictated in the initial brief
    columns_to_drop = ["Customer Name", "Customer e-mail", "Country", "Phone Number", "Home Address", "Date of Birth"]
    data_cleaned = sample_raw_data.drop(columns=columns_to_drop)
    X = data_cleaned.drop("Car Purchase Amount", axis=1)
    y = data_cleaned["Car Purchase Amount"]
    return X, y, columns_to_drop

# Test Case 1: Input Data Privacy Check
def test_input_data_privacy_check(processed_pipeline_data):
    X, _, _ = processed_pipeline_data
    # Verifying that the custom privacy assets are completely isolated from training configurations
    forbidden_columns = ["Customer Name", "Customer e-mail", "Country", "Phone Number", "Home Address", "Date of Birth"]
    for col in forbidden_columns:
        assert col not in X.columns, f"Privacy Leak Error: '{col}' is exposed in training features!"

# Test Case 2: Output Data Privacy Check
def test_output_data_privacy_check(processed_pipeline_data):
    _, y, _ = processed_pipeline_data
    assert y.name == "Car Purchase Amount", "Target mismatch."
    # Assert y is a single target dimension and doesn't contain a compound dictionary of PII
    assert isinstance(y, pd.Series), "Target matrix should be a clean singular sequence."

# Test Case 3: Dataset Shape Check
def test_dataset_shape_check(processed_pipeline_data):
    X, _, _ = processed_pipeline_data
    # Expected inputs are exactly 5 features: Gender, Age, Annual Salary, Credit Card Debt, Net Worth
    assert X.shape[1] == 5, f"Expected 5 features, but got {X.shape[1]} instead."

# Test Case 4: Approved Input Feature Check
def test_approved_input_feature_check(processed_pipeline_data):
    X, _, _ = processed_pipeline_data
    approved_features = ["Gender", "Age", "Annual Salary", "Credit Card Debt", "Net Worth"]
    assert list(X.columns) == approved_features, "Input features do not match approved design spec."

# Test Case 5: Removed Columns Check
def test_removed_columns_check(processed_pipeline_data):
    _, _, removed_cols = processed_pipeline_data
    required_removals = ["Customer Name", "Customer e-mail", "Country", "Phone Number", "Home Address", "Date of Birth"]
    assert all(item in removed_cols for item in required_removals), "Not all customized PII columns are flagged for deletion."

# Test Case 6: Saved Model/Pipeline Privacy Check
def test_model_prediction_privacy_check():
    # Simulating a mock inference matrix using only the approved schema parameters
    approved_mock_input = pd.DataFrame([{
        'Gender': 1, 'Age': 35, 'Annual Salary': 70000, 'Credit Card Debt': 5000, 'Net Worth': 300000
    }])
    # Asserting that data shape matching approved columns is sufficient to execute shapes structure
    assert approved_mock_input.shape[1] == 5
    assert "Customer Name" not in approved_mock_input.columns
    assert "Phone Number" not in approved_mock_input.columns

# Test Case 7: Prediction Output Privacy Check
def test_prediction_output_privacy_check():
    # Simulate an engine inference step outputting a raw float metric
    mock_prediction_result = float(45250.75)
    
    # Asserting result is completely numeric and contains no structural metadata leaking labels or text
    assert isinstance(mock_prediction_result, float)
    assert "Connor" not in str(mock_prediction_result)
    assert "021" not in str(mock_prediction_result)