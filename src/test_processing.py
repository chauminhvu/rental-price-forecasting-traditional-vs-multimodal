import sys
sys.path.append("..")
import unittest
from data_processing import *


class TestMadFunction(unittest.TestCase):
    
    def test_haversine_distance(self):
        # Reference coordinates for Stuttgart
        ref_lat = 48.7758
        ref_lon = 9.1829

        # Coordinates for Berlin
        lat_berlin = 52.520008
        lon_berlin = 13.404954

        expected_distance_km = 511

        calculated_distance_km = haversine_distance(
            ref_lat, ref_lon, lat_berlin, lon_berlin)

        self.assertAlmostEqual(calculated_distance_km,
                               expected_distance_km, delta=1)

    # def test_numeric_data(self):
    #     series = pd.Series([1, 3, 3, 6, 8, 10, 10, 1000])
    #     result = compute_mad_1d(series)
    #     self.assertAlmostEqual(result, 5.1891)

    # def test_nonnumeric_data(self):
    #     series = pd.Series(['a', 'b', 'c'])

    #     # This should raise a TypeError because the data is not numeric.
    #     with self.assertRaises(TypeError):
    #         compute_mad_1d(series)



if __name__ == "__main__":
    unittest.main()
