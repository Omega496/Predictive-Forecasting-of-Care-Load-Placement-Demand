"""
Validation and Evaluation Metrics Engine
UAC Program Care Load & Placement Demand Forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Generator
from sklearn.model_selection import TimeSeriesSplit

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_baseline_lag: np.ndarray = None) -> Dict[str, float]:
    """
    Computes standard forecasting error and business KPI metrics:
    MAE, RMSE, MAPE, Accuracy (%), Directional Accuracy (%), and Stability.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Avoid zero-division in MAPE
    mask = np.abs(y_true) > 0
    if not np.any(mask):
        mape = np.nan
        accuracy = np.nan
    else:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100.0)
        accuracy = max(0.0, float(100.0 - mape))
        
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # Directional Accuracy (trend direction vs previous lag)
    if y_baseline_lag is not None and len(y_baseline_lag) == len(y_true):
        actual_dir = np.sign(y_true - y_baseline_lag)
        pred_dir = np.sign(y_pred - y_baseline_lag)
        directional_acc = float(np.mean(actual_dir == pred_dir) * 100.0)
    else:
        directional_acc = np.nan
        
    # Forecast Stability Index (FSI): 1 - Var(d_pred)/Var(y_true)
    if len(y_pred) > 2 and np.var(y_true) > 0:
        fsi = float(max(0.0, 1.0 - (np.var(np.diff(y_pred)) / (np.var(y_true) + 1e-6))))
    else:
        fsi = 1.0

    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2),
        'Accuracy': round(accuracy, 2),
        'Directional_Accuracy': round(directional_acc, 2) if not np.isnan(directional_acc) else None,
        'Stability_Index': round(fsi, 3)
    }

def get_time_series_splits(n_samples: int, n_splits: int = 5, test_size: int = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields strictly forward-walking expanding-window indices (train_idx, test_idx).
    Ensures zero temporal leakage from future into past.
    """
    if test_size is not None:
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
    indices = np.arange(n_samples)
    for train_idx, test_idx in tscv.split(indices):
        yield train_idx, test_idx

def get_train_test_split_by_date(df: pd.DataFrame, test_days: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into chronological train and test subsets based on the last N days.
    """
    df_sorted = df.sort_values('date').reset_index(drop=True)
    cutoff_date = df_sorted['date'].max() - pd.Timedelta(days=test_days)
    
    train = df_sorted[df_sorted['date'] <= cutoff_date].copy()
    test = df_sorted[df_sorted['date'] > cutoff_date].copy()
    
    return train, test

if __name__ == '__main__':
    y_t = np.array([100.0, 105.0, 102.0, 110.0, 108.0])
    y_p = np.array([98.0, 104.0, 105.0, 109.0, 107.0])
    lag = np.array([99.0, 100.0, 105.0, 102.0, 110.0])
    metrics = calculate_metrics(y_t, y_p, lag)
    print("Test calculation metrics:", metrics)
