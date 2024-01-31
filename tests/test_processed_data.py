import pandas as pd
import pytest
from src.data.data_processing import count_missing_values


class TestRentalData:
    """
    Check the cleaned data after processing.
    """
    @classmethod
    def setup_class(cls):
        cls.clean_data = pd.read_csv('data/processed/rental_processed.csv')

    def test_no_missing_values(self):
        """no missing values (NaN) in the data"""
        # missing values
        missing_values_report = count_missing_values(self.clean_data)
        assert missing_values_report['Missing count'].sum() == 0, \
            f"Found missing values in rental_processed.csv:\n{missing_values_report}"

    def test_features(self):
        """ check distance is presented in the data"""
        expected_features = ['distance']
        actual_features = list(self.clean_data.columns)
        for feature in expected_features:
            assert feature in actual_features, f"Expected feature '{feature}' not found in the dataset."
