# src/features.py

import pandas as pd
import numpy as np

def add_lag_features(df, lags=[1, 2, 24, 48, 168]):
    for lag in lags:
        df[f'lag_{lag}'] = df['load'].shift(lag)
    return df

def add_rolling_features(df, windows=[24, 48, 168]):
    for window in windows:
        df[f'rolling_mean_{window}'] = df['load'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df['load'].shift(1).rolling(window).std()
    return df

def add_calendar_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df
