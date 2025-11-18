"""
Pytest tests for F1 Race Data Analyzer
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Add src directory to path
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

import f1_analyzer


# Mock data for testing
@pytest.fixture
def mock_lap_data():
    """Create mock lap time data for testing"""
    return pd.DataFrame({
        'LapNumber': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'LapTime': [92.5, 91.2, 90.8, 90.5, 90.3, 90.1, 89.9, 89.8, 89.7, 89.6]
    })


def test_calculate_average_lap_time(mock_lap_data):
    """Test average lap time calculation"""
    avg_time = f1_analyzer.calculate_average_lap_time(mock_lap_data)
    expected = mock_lap_data['LapTime'].mean()
    assert abs(avg_time - expected) < 0.01, "Average lap time calculation is incorrect"


def test_calculate_average_lap_time_single_lap():
    """Test average lap time with single lap"""
    single_lap_data = pd.DataFrame({
        'LapNumber': [1],
        'LapTime': [90.5]
    })
    avg_time = f1_analyzer.calculate_average_lap_time(single_lap_data)
    assert avg_time == 90.5, "Single lap average should equal the lap time"


def test_predict_lap_times_structure(mock_lap_data):
    """Test that predict_lap_times returns correct structure"""
    result = f1_analyzer.predict_lap_times(mock_lap_data, future_laps=3)
    
    assert 'model_score' in result, "Result should contain model_score"
    assert 'predictions' in result, "Result should contain predictions"
    assert 'future_lap_numbers' in result, "Result should contain future_lap_numbers"
    assert len(result['predictions']) == 3, "Should predict 3 future laps"
    assert len(result['future_lap_numbers']) == 3, "Should have 3 future lap numbers"


def test_predict_lap_times_insufficient_data():
    """Test that predict_lap_times raises error with insufficient data"""
    small_data = pd.DataFrame({
        'LapNumber': [1, 2, 3],
        'LapTime': [90.5, 90.3, 90.1]
    })
    
    with pytest.raises(ValueError, match="Need at least 5 laps"):
        f1_analyzer.predict_lap_times(small_data)


def test_predict_lap_times_trend(mock_lap_data):
    """Test that predictions follow the expected trend"""
    result = f1_analyzer.predict_lap_times(mock_lap_data, future_laps=2)
    predictions = result['predictions']
    
    # Since lap times are decreasing, predictions should be reasonable
    assert all(isinstance(p, float) for p in predictions), "Predictions should be floats"
    assert predictions[0] < 95, "First prediction should be reasonable"
    assert predictions[0] > 85, "First prediction should be reasonable"


def test_predict_lap_times_different_future_laps(mock_lap_data):
    """Test predictions with different numbers of future laps"""
    result_2 = f1_analyzer.predict_lap_times(mock_lap_data, future_laps=2)
    result_5 = f1_analyzer.predict_lap_times(mock_lap_data, future_laps=5)
    
    assert len(result_2['predictions']) == 2
    assert len(result_5['predictions']) == 5


def test_model_score_range(mock_lap_data):
    """Test that model score is between -1 and 1"""
    result = f1_analyzer.predict_lap_times(mock_lap_data, future_laps=2)
    score = result['model_score']
    
    assert score >= -1, "Model score should be >= -1"
    assert score <= 1, "Model score should be <= 1"


# Parametrized test example
@pytest.mark.parametrize("lap_times,expected_avg", [
    ([90.0, 90.0, 90.0], 90.0),
    ([89.0, 90.0, 91.0], 90.0),
    ([85.5, 86.5, 87.5], 86.5),
])
def test_calculate_average_parametrized(lap_times, expected_avg):
    """Parametrized test for average calculation"""
    data = pd.DataFrame({
        'LapNumber': range(1, len(lap_times) + 1),
        'LapTime': lap_times
    })
    avg = f1_analyzer.calculate_average_lap_time(data)
    assert abs(avg - expected_avg) < 0.01


def test_empty_dataframe():
    """Test behavior with empty DataFrame"""
    empty_data = pd.DataFrame({
        'LapNumber': [],
        'LapTime': []
    })
    # This should return NaN
    result = f1_analyzer.calculate_average_lap_time(empty_data)
    assert pd.isna(result), "Empty data should return NaN"