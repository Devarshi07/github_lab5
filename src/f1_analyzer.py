"""
F1 Race Data Analyzer
This module provides functions to analyze Formula 1 race data using FastF1
"""

import fastf1
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def load_session_data(year, race, session_type='R'):
    """
    Load F1 session data for a specific race
    
    Args:
        year (int): Year of the race
        race (int or str): Race number or name
        session_type (str): Type of session ('R' for Race, 'Q' for Qualifying)
    
    Returns:
        fastf1.core.Session: Session object with loaded data
    """
    session = fastf1.get_session(year, race, session_type)
    session.load()
    return session


def get_lap_times(session, driver):
    """
    Extract lap times for a specific driver
    
    Args:
        session: FastF1 session object
        driver (str): Driver code (e.g., 'VER', 'HAM')
    
    Returns:
        pd.DataFrame: DataFrame with lap numbers and times
    """
    laps = session.laps.pick_driver(driver)
    lap_data = pd.DataFrame({
        'LapNumber': laps['LapNumber'],
        'LapTime': laps['LapTime'].dt.total_seconds()
    })
    return lap_data.dropna()


def calculate_average_lap_time(lap_data):
    """
    Calculate average lap time from lap data
    
    Args:
        lap_data (pd.DataFrame): DataFrame with lap times
    
    Returns:
        float: Average lap time in seconds
    """
    return lap_data['LapTime'].mean()


def predict_lap_times(lap_data, future_laps=5):
    """
    Build a simple linear regression model to predict future lap times
    
    Args:
        lap_data (pd.DataFrame): DataFrame with lap numbers and times
        future_laps (int): Number of future laps to predict
    
    Returns:
        dict: Dictionary containing model score and predictions
    """
    if len(lap_data) < 5:
        raise ValueError("Need at least 5 laps of data for prediction")
    
    X = lap_data['LapNumber'].values.reshape(-1, 1)
    y = lap_data['LapTime'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate score
    score = model.score(X_test, y_test)
    
    # Predict future laps
    last_lap = lap_data['LapNumber'].max()
    future_lap_numbers = np.arange(
        last_lap + 1, 
        last_lap + future_laps + 1
    ).reshape(-1, 1)
    
    predictions = model.predict(future_lap_numbers)
    
    return {
        'model_score': score,
        'predictions': predictions.tolist(),
        'future_lap_numbers': future_lap_numbers.flatten().tolist()
    }


def compare_drivers(session, driver1, driver2):
    """
    Compare average lap times between two drivers
    
    Args:
        session: FastF1 session object
        driver1 (str): First driver code
        driver2 (str): Second driver code
    
    Returns:
        dict: Comparison results with average times and difference
    """
    laps1 = get_lap_times(session, driver1)
    laps2 = get_lap_times(session, driver2)
    
    avg1 = calculate_average_lap_time(laps1)
    avg2 = calculate_average_lap_time(laps2)
    
    return {
        f'{driver1}_avg': avg1,
        f'{driver2}_avg': avg2,
        'difference': abs(avg1 - avg2),
        'faster_driver': driver1 if avg1 < avg2 else driver2
    }