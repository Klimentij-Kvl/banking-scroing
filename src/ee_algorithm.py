import pandas as pd
from sklearn.covariance import EllipticEnvelope

def ee_algorithm(data: pd.DataFrame)->pd.DataFrame:
    ee = EllipticEnvelope(random_state=30,contamination=0.1)
    ee.fit(data)
    labels = ee.predict(data)
    return labels

def censoring_data(data: pd.DataFrame, selected_columns: list)->pd.DataFrame:
    labels = ee_algorithm(data[selected_columns])
    data_clean = data[labels != -1].copy()
    return data_clean

def normalize_data(data: pd.DataFrame)->pd.DataFrame:
    data_normalized = data.copy()
    for col in data.columns:
        k_min, k_max = data[col].min(), data[col].max()
        #data_normalized[col] = (data[col] - k_min)/(k_max-k_min)
        data_normalized[col] = (data[col]-k_min)/(k_max-k_min)
    return data_normalized

def prepare_data(data: pd.DataFrame, selected_columns: list)->pd.DataFrame:
    return normalize_data(censoring_data(data, selected_columns))