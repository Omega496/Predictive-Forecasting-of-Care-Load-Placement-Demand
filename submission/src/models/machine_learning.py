"""
Machine Learning Forecasting Models (Direct Multi-Horizon)
UAC Program Care Load & Placement Demand Forecasting
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from src.feature_engineering import get_feature_columns, prepare_xy_for_horizon
from src.validation import calculate_metrics

class DirectMLForecaster:
    """
    Direct Multi-Horizon Forecaster training dedicated models
    for specific forecast horizons (e.g., 7, 14, 30 days).
    """
    def __init__(self, model_type: str = 'xgboost', target_type: str = 'care_load', horizons: List[int] = [7, 14, 30], random_state: int = 42):
        self.model_type = model_type.lower()
        self.target_type = target_type
        self.horizons = horizons
        self.random_state = random_state
        self.models: Dict[int, Any] = {}
        self.residual_stds: Dict[int, float] = {}
        self.feature_names: List[str] = []
        self.feature_importances: Dict[int, pd.Series] = {}

    def _instantiate_model(self) -> Any:
        if self.model_type == 'rf' or self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=120,
                max_depth=10,
                min_samples_split=4,
                n_jobs=-1,
                random_state=self.random_state
            )
        elif self.model_type == 'gbr' or self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state
            )
        elif self.model_type == 'xgboost' or self.model_type == 'xgb':
            return XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(self, df_feat: pd.DataFrame):
        """Fits dedicated model for each configured horizon."""
        self.feature_names = get_feature_columns(df_feat)
        
        for h in self.horizons:
            X, y, _ = prepare_xy_for_horizon(df_feat, target_type=self.target_type, horizon=h)
            
            # Temporal train/eval split for residual uncertainty estimation
            split_idx = int(len(X) * 0.85)
            X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
            X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
            
            model = self._instantiate_model()
            model.fit(X_train, y_train)
            
            # Compute evaluation residuals for prediction intervals
            val_preds = model.predict(X_val)
            res_std = float(np.std(y_val - val_preds))
            if self.target_type in ['discharges', 'net_flow']:
                self.residual_stds[h] = max(res_std, 1.0)
            else:
                self.residual_stds[h] = max(res_std, 10.0) # floor std to 10 for care load
            
            # Retrain on full available historical dataset for operational deployment
            model.fit(X, y)
            self.models[h] = model
            
            # Store feature importances
            if hasattr(model, 'feature_importances_'):
                self.feature_importances[h] = pd.Series(
                    model.feature_importances_,
                    index=self.feature_names
                ).sort_values(ascending=False)

        return self

    def predict_horizon(self, current_features: pd.DataFrame, horizon: int) -> Tuple[float, float, float, float, float]:
        """
        Generates point forecast and 80% & 95% confidence intervals for a specific horizon step.
        Returns: (point_pred, lower_80, upper_80, lower_95, upper_95)
        """
        if horizon not in self.models:
            raise ValueError(f"Model for horizon {horizon} not fitted.")
            
        model = self.models[horizon]
        feat_vector = current_features[self.feature_names].iloc[[-1]]
        point_pred = float(model.predict(feat_vector)[0])
        
        std = self.residual_stds.get(horizon, 100.0)
        
        # Conformal / Gaussian intervals (allow negative values for net flow)
        if self.target_type == 'net_flow':
            lower_80 = point_pred - 1.28 * std
            lower_95 = point_pred - 1.96 * std
        else:
            lower_80 = max(0.0, point_pred - 1.28 * std)
            lower_95 = max(0.0, point_pred - 1.96 * std)
        upper_80 = point_pred + 1.28 * std
        upper_95 = point_pred + 1.96 * std
        
        return point_pred, lower_80, upper_80, lower_95, upper_95

    def predict_trajectory(self, current_features: pd.DataFrame) -> pd.DataFrame:
        """
        Produces multi-horizon trajectory across all configured horizons.
        """
        rows = []
        for h in self.horizons:
            pred, l80, u80, l95, u95 = self.predict_horizon(current_features, h)
            rows.append({
                'horizon_days': h,
                'point_forecast': round(pred, 1),
                'lower_80': round(l80, 1),
                'upper_80': round(u80, 1),
                'lower_95': round(l95, 1),
                'upper_95': round(u95, 1),
                'pred_std': round(self.residual_stds[h], 1)
            })
        return pd.DataFrame(rows)

    def save(self, filepath: str):
        """Serializes forecaster to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'DirectMLForecaster':
        """Loads serialized forecaster from disk."""
        return joblib.load(filepath)
