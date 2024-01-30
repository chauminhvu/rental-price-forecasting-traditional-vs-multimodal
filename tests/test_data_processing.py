from os import name
import sys
import pandas as pd
import pytest

sys.path.insert(0, "./src")
from data_processing import count_missing_values, haversine_distance


class TestCountMissingValues:
    def test_empty_dataframe(self):
        """
        Test an empty DataFrame is passed.
        """
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            count_missing_values(empty_df)

    def test_empty_series(self):
        """
        Test an empty Series is passed.
        """
        empty_series = pd.Series([], dtype=int)
        with pytest.raises(ValueError):
            count_missing_values(empty_series)

    def test_missing_values_in_series(self):
        """
        Test calculates missing values in a Series.
        """
        series_missing_values = pd.Series([1, None, 3, None, 5], name='A')
        # import pdb;pdb.set_trace()
        result = count_missing_values(series_missing_values)

        expected_result = pd.DataFrame({'Missing count': [2],
                                        'Percentage': [40.0]},
                                       index=['A'])
        pd.testing.assert_frame_equal(result, expected_result)

    def test_missing_values_in_dataframe(self):
        """
        Test calculates missing values in a DataFrame.
        """
        df_missing_values = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': [None, 2, 3, 4, 5],
            'C': [1, 2, 3, None, None],
            })
        result = count_missing_values(df_missing_values)
        expected_result = pd.DataFrame({
            'Missing count': [1, 1, 2],
            'Percentage': [20.0, 20.0, 40.0]
        }, index=['A', 'B', 'C'])
        expected_result = expected_result.sort_values(by='Percentage',
                                                      ascending=False)
        pd.testing.assert_frame_equal(result, expected_result)


class TestHaversineDistance:

    def test_known_coordinates(self):
        """
        Test haversine_distance function with known coordinates.
        """
        ref_lat = 48.7758
        ref_lon = 9.1829
        lat_berlin = 52.520008
        lon_berlin = 13.404954
        expected_distance_km = 511

        calculated_distance_km = haversine_distance(
            ref_lat, ref_lon, lat_berlin, lon_berlin).get()

        assert pytest.approx(calculated_distance_km, 1) == expected_distance_km
