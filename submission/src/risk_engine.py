"""
Capacity Risk & Early-Warning Surge Engine
UAC Program Care Load & Placement Demand Forecasting
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Any, Optional

class CapacityRiskEngine:
    """
    Computes Capacity Breach Probability, Surge Lead Time, and Alert Tiers
    based on multi-horizon forecasted care load and predictive uncertainty distributions.
    """
    def __init__(self, operational_capacity: float = 3000.0, crisis_capacity: float = 9762.0):
        self.operational_capacity = float(operational_capacity)
        self.crisis_capacity = float(crisis_capacity)

    def calculate_point_breach_probability(self, point_forecast: float, pred_std: float, threshold: Optional[float] = None) -> float:
        """
        Computes P(Y > C) = 1 - Phi((C - y_hat) / sigma) for a single forecast step.
        """
        c = float(threshold if threshold is not None else self.operational_capacity)
        sigma = max(float(pred_std), 1.0) # Avoid zero division
        z = (c - point_forecast) / sigma
        prob = float(1.0 - norm.cdf(z))
        return float(np.clip(prob, 0.0, 1.0))

    def calculate_forecast_stability_index(self, y_pred: np.ndarray, y_historical: np.ndarray) -> float:
        """
        Computes Forecast Stability Index (FSI):
        FSI = max(0.0, 1.0 - (Var(diff(y_pred)) / (Var(y_historical) + 1e-6)))
        """
        y_p = np.asarray(y_pred, dtype=float)
        y_h = np.asarray(y_historical, dtype=float)
        if len(y_p) < 2 or np.var(y_h) <= 0:
            return 1.0
        fsi = float(max(0.0, 1.0 - (np.var(np.diff(y_p)) / (np.var(y_h) + 1e-6))))
        return float(round(fsi, 3))

    def evaluate_forecast_trajectory(
        self,
        forecast_df: pd.DataFrame,
        capacity_threshold: Optional[float] = None,
        risk_tolerance: float = 0.50
    ) -> Dict[str, Any]:
        """
        Evaluates an entire forecast trajectory across horizons (e.g. 7, 14, 30 days).
        forecast_df must contain: ['horizon_days', 'point_forecast', 'pred_std']
        
        Returns:
            - point_probabilities: list of step-level breach probabilities
            - cumulative_breach_probability: cumulative risk across entire horizon
            - surge_lead_time_days: days until breach or safe indicator
            - alert_tier: 'GREEN', 'YELLOW', 'ORANGE', or 'RED'
            - alert_color: HEX color code
            - alert_message: Human-readable operational recommendation
        """
        c = float(capacity_threshold if capacity_threshold is not None else self.operational_capacity)
        
        step_probs = []
        breach_horizon = None
        max_horizon = int(forecast_df['horizon_days'].max())
        
        for _, row in forecast_df.sort_values('horizon_days').iterrows():
            h = int(row['horizon_days'])
            pred = float(row['point_forecast'])
            std = float(row.get('pred_std', 100.0))
            
            p_step = self.calculate_point_breach_probability(pred, std, threshold=c)
            step_probs.append({'horizon_days': h, 'breach_probability': round(p_step * 100.0, 1)})
            
            # Check if breach condition met: point forecast >= capacity OR breach prob >= risk tolerance
            if breach_horizon is None and (pred >= c or p_step >= risk_tolerance):
                breach_horizon = h

        # Cumulative probability across horizons: 1 - prod(1 - p_i)
        probs_array = np.array([item['breach_probability'] / 100.0 for item in step_probs])
        cum_prob = float(1.0 - np.prod(1.0 - probs_array))
        cum_prob_pct = round(cum_prob * 100.0, 1)

        # Determine Surge Lead Time
        if breach_horizon is not None:
            surge_lead_time = breach_horizon
            lead_time_display = f"{surge_lead_time} Days"
        else:
            surge_lead_time = None
            lead_time_display = f"> {max_horizon} Days (Safe)"

        # Classify Alert Tier
        if cum_prob_pct >= 60.0 or (surge_lead_time is not None and surge_lead_time <= 7):
            alert_tier = "RED"
            alert_color = "#d62728"
            alert_title = "CRITICAL SURGE WARNING"
            alert_message = (
                f"Severe capacity strain detected. High likelihood of exceeding {c:,.0f} beds within "
                f"{surge_lead_time if surge_lead_time else 'horizon'} days. Immediately activate Emergency Influx Sites (EIS) "
                "and deploy emergency caseworker squads to accelerate sponsor discharges."
            )
        elif cum_prob_pct >= 30.0 or (surge_lead_time is not None and surge_lead_time <= 14):
            alert_tier = "ORANGE"
            alert_color = "#ff7f0e"
            alert_title = "ELEVATED CAPACITY ALERT"
            alert_message = (
                f"Shelter utilization trending toward {c:,.0f} beds. Lead time: {lead_time_display}. "
                "Authorize caseworker overtime, expedite fingerprinting/sponsor checks, and prepare contingency influx beds."
            )
        elif cum_prob_pct >= 10.0 or (surge_lead_time is not None and surge_lead_time <= 30):
            alert_tier = "YELLOW"
            alert_color = "#f1c40f"
            alert_title = "WATCH / MONITORING"
            alert_message = (
                f"Moderate accumulation pressure. Monitor CBP intake trends and maintain steady sponsor discharge velocity."
            )
        else:
            alert_tier = "GREEN"
            alert_color = "#2ecc71"
            alert_title = "NORMAL OPERATIONS"
            alert_message = (
                f"Care load well within safe margins below {c:,.0f} beds. Discharge capacity currently matches incoming intake."
            )

        return {
            'capacity_threshold': c,
            'step_probabilities': step_probs,
            'cumulative_breach_probability_pct': cum_prob_pct,
            'surge_lead_time_days': surge_lead_time,
            'surge_lead_time_display': lead_time_display,
            'alert_tier': alert_tier,
            'alert_color': alert_color,
            'alert_title': alert_title,
            'alert_message': alert_message
        }

if __name__ == '__main__':
    engine = CapacityRiskEngine(operational_capacity=3000.0)
    mock_forecast = pd.DataFrame([
        {'horizon_days': 7, 'point_forecast': 2520.0, 'pred_std': 85.0},
        {'horizon_days': 14, 'point_forecast': 2680.0, 'pred_std': 120.0},
        {'horizon_days': 30, 'point_forecast': 3050.0, 'pred_std': 180.0}
    ])
    result = engine.evaluate_forecast_trajectory(mock_forecast, capacity_threshold=3000.0)
    print("Risk Engine Evaluation Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
