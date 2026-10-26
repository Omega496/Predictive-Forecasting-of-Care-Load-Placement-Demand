"""
Statistical and Classical Time-Series Forecasting Models
UAC Program Care Load & Placement Demand Forecasting
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Optional

class HoltWintersModel:
    """Holt-Winters Exponential Smoothing with additive trend and weekly seasonality."""
    def __init__(self, seasonal_periods: int = 7, trend: str = 'add', seasonal: str = 'add'):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.fitted_model = None
        self.residual_std = 0.0

    def fit(self, y: pd.Series):
        # Ensure positive values for stability
        y_clean = y.dropna().astype(float)
        # Use simple exponential smoothing or additive Holt-Winters
        try:
            model = ExponentialSmoothing(
                y_clean,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method='estimated'
            )
            self.fitted_model = model.fit(optimized=True)
            residuals = y_clean - self.fitted_model.fittedvalues
            self.residual_std = float(residuals.std())
        except Exception:
            # Fallback to simple exponential smoothing if seasonal fit fails
            model = ExponentialSmoothing(y_clean, trend='add')
            self.fitted_model = model.fit()
            residuals = y_clean - self.fitted_model.fittedvalues
            self.residual_std = float(residuals.std())
        return self

    def predict(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (point_predictions, lower_95_ci, upper_95_ci)."""
        preds = self.fitted_model.forecast(horizon).values
        # Expand uncertainty over horizon: std * sqrt(1 + h * 0.05)
        h_factors = np.sqrt(1 + np.arange(horizon) * 0.05)
        lower_95 = np.maximum(0, preds - 1.96 * self.residual_std * h_factors)
        upper_95 = preds + 1.96 * self.residual_std * h_factors
        return preds, lower_95, upper_95

class ARIMAModel:
    """SARIMAX (Autoregressive Integrated Moving Average) model."""
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None

    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None):
        y_clean = y.dropna().astype(float)
        try:
            model = SARIMAX(
                y_clean,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = model.fit(disp=False, maxiter=50)
        except Exception:
            # Fallback to simple ARIMA(1, 1, 0)
            model = SARIMAX(y_clean, order=(1, 1, 0), enforce_stationarity=False)
            self.fitted_model = model.fit(disp=False, maxiter=50)
        return self

    def predict(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (point_predictions, lower_95_ci, upper_95_ci)."""
        forecast_res = self.fitted_model.get_forecast(steps=horizon)
        preds = forecast_res.predicted_mean.values
        ci = forecast_res.conf_int(alpha=0.05)
        lower_95 = np.maximum(0, ci.iloc[:, 0].values)
        upper_95 = ci.iloc[:, 1].values
        return preds, lower_95, upper_95
