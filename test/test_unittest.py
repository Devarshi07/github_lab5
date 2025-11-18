"""
Unittest tests for F1 Race Data Analyzer
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import f1_analyzer


class TestF1Analyzer(unittest.TestCase):
    """Test cases for F1 Analyzer functions"""
    
    def setUp(self):
        """Set up mock data before each test"""
        self.mock_lap_data = pd.DataFrame({
            'LapNumber': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'LapTime': [92.5, 91.2, 90.8, 90.5, 90.3, 90.1, 89.9, 89.8, 89.7, 89.6]
        })
    
    def test_calculate_average_lap_time(self):
        """Test average lap time calculation"""
        avg_time = f1_analyzer.calculate_average_lap_time(self.mock_lap_data)
        expected = self.mock_lap_data['LapTime'].mean()
        self.assertAlmostEqual(avg_time, expected, places=2,
                              msg="Average lap time calculation is incorrect")
    
    def test_calculate_average_lap_time_single_lap(self):
        """Test average lap time with single lap"""
        single_lap_data = pd.DataFrame({
            'LapNumber': [1],
            'LapTime': [90.5]
        })
        avg_time = f1_analyzer.calculate_average_lap_time(single_lap_data)
        self.assertEqual(avg_time, 90.5,
                        "Single lap average should equal the lap time")
    
    def test_calculate_average_lap_time_multiple_values(self):
        """Test average calculation with known values"""
        test_data_1 = pd.DataFrame({
            'LapNumber': [1, 2, 3],
            'LapTime': [90.0, 90.0, 90.0]
        })
        self.assertEqual(f1_analyzer.calculate_average_lap_time(test_data_1), 90.0)
        
        test_data_2 = pd.DataFrame({
            'LapNumber': [1, 2, 3],
            'LapTime': [89.0, 90.0, 91.0]
        })
        self.assertEqual(f1_analyzer.calculate_average_lap_time(test_data_2), 90.0)
        
        test_data_3 = pd.DataFrame({
            'LapNumber': [1, 2, 3],
            'LapTime': [85.5, 86.5, 87.5]
        })
        self.assertEqual(f1_analyzer.calculate_average_lap_time(test_data_3), 86.5)
    
    def test_predict_lap_times_structure(self):
        """Test that predict_lap_times returns correct structure"""
        result = f1_analyzer.predict_lap_times(self.mock_lap_data, future_laps=3)
        
        self.assertIn('model_score', result,
                     "Result should contain model_score")
        self.assertIn('predictions', result,
                     "Result should contain predictions")
        self.assertIn('future_lap_numbers', result,
                     "Result should contain future_lap_numbers")
        self.assertEqual(len(result['predictions']), 3,
                        "Should predict 3 future laps")
        self.assertEqual(len(result['future_lap_numbers']), 3,
                        "Should have 3 future lap numbers")
    
    def test_predict_lap_times_insufficient_data(self):
        """Test that predict_lap_times raises error with insufficient data"""
        small_data = pd.DataFrame({
            'LapNumber': [1, 2, 3],
            'LapTime': [90.5, 90.3, 90.1]
        })
        
        with self.assertRaises(ValueError) as context:
            f1_analyzer.predict_lap_times(small_data)
        
        self.assertIn("Need at least 5 laps", str(context.exception))
    
    def test_predict_lap_times_returns_floats(self):
        """Test that predictions are float values"""
        result = f1_analyzer.predict_lap_times(self.mock_lap_data, future_laps=2)
        predictions = result['predictions']
        
        for pred in predictions:
            self.assertIsInstance(pred, float,
                                "Each prediction should be a float")
    
    def test_predict_lap_times_reasonable_values(self):
        """Test that predictions are within reasonable range"""
        result = f1_analyzer.predict_lap_times(self.mock_lap_data, future_laps=2)
        predictions = result['predictions']
        
        for pred in predictions:
            self.assertGreater(pred, 85,
                             "Prediction should be greater than 85 seconds")
            self.assertLess(pred, 95,
                          "Prediction should be less than 95 seconds")
    
    def test_predict_lap_times_with_different_future_laps(self):
        """Test predictions with different numbers of future laps"""
        result_2 = f1_analyzer.predict_lap_times(self.mock_lap_data, future_laps=2)
        result_5 = f1_analyzer.predict_lap_times(self.mock_lap_data, future_laps=5)
        
        self.assertEqual(len(result_2['predictions']), 2)
        self.assertEqual(len(result_5['predictions']), 5)
    
    def test_model_score_range(self):
        """Test that model score is between -1 and 1"""
        result = f1_analyzer.predict_lap_times(self.mock_lap_data, future_laps=2)
        score = result['model_score']
        
        self.assertGreaterEqual(score, -1,
                               "Model score should be >= -1")
        self.assertLessEqual(score, 1,
                           "Model score should be <= 1")
    
    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame"""
        empty_data = pd.DataFrame({
            'LapNumber': [],
            'LapTime': []
        })
        result = f1_analyzer.calculate_average_lap_time(empty_data)
        self.assertTrue(pd.isna(result),
                       "Empty data should return NaN")


if __name__ == '__main__':
    unittest.main()