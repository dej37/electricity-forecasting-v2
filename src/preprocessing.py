import pandas as pd
import numpy as np

path=r"C:\Users\Top Prix\OneDrive\New folder\Thesis_Project\notebooks\processed_rin_data.csv"
def load_data(path):
    df = pd.read_csv(path)

    # adjust column names if needed
    df.columns = ['timestamp', 'load']

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()

    return df


def regularize_time_index(df):
    # enforce strict 15-min grid
    df = df.resample('15T').mean()
    return df


def detect_and_fix_anomalies(df):
    # Example: remove zeros (invalid for national load)
    df.loc[df['load'] == 0, 'load'] = np.nan

    # Example: remove abnormal low values (you can adjust threshold)
    threshold = 2000
    df.loc[df['load'] < threshold, 'load'] = np.nan

    # Interpolate missing values
    df['load'] = df['load'].interpolate(method='linear')

    return df


def aggregate_hourly(df):
    df_hourly = df.resample('H').mean()
    return df_hourly