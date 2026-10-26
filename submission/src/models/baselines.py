"""
Baseline Forecasting Models
UAC Program Care Load & Placement Demand Forecasting
"""

import numpy as np
import pandas as pd

class NaivePersistenceModel:
    """Predicts future care load / discharges as the most recent known value (t)."""
    def __init__(self):
        self.last_value = None

    def fit(self, y: pd.Series):
        self.last_value = float(y.iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last_value)

class MovingAverageModel:
    """Predicts future values as the rolling average of the last W observed days."""
    def __init__(self, window: int = 7):
        self.window = window
        self.mean_value = None

    def fit(self, y: pd.Series):
        recent = y.iloc[-self.window:]
        self.mean_value = float(recent.mean())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.mean_value)
