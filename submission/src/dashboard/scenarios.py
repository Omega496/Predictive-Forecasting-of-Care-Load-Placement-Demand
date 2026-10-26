"""
Scenario Simulator & Stress-Testing Sandbox
UAC Program Care Load & Placement Demand Forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

def simulate_care_load_scenario(
    base_forecast_df: pd.DataFrame,
    current_care_load: float,
    current_transfers: float,
    current_discharges: float,
    inflow_pct_change: float = 0.0,
    discharge_pct_change: float = 0.0,
    capacity_threshold: float = 3000.0
) -> pd.DataFrame:
    """
    Simulates cumulative care load trajectory under scenario adjustments:
    - inflow_pct_change: e.g. +25.0 means +25% transfers from CBP
    - discharge_pct_change: e.g. -20.0 means -20% discharges to sponsors
    
    Uses system flow conservation:
    delta_t = (transfers * (1 + inflow_pct)) - (discharges * (1 + discharge_pct))
    care_load_t = care_load_t-1 + delta_t
    """
    adj_transfers = current_transfers * (1.0 + inflow_pct_change / 100.0)
    adj_discharges = max(1.0, current_discharges * (1.0 + discharge_pct_change / 100.0))
    daily_net_delta = adj_transfers - adj_discharges
    
    sim_rows = []
    for _, row in base_forecast_df.sort_values('horizon_days').iterrows():
        h = int(row['horizon_days'])
        base_pred = float(row['point_forecast'])
        std = float(row.get('pred_std', 100.0))
        
        # Cumulative flow shift across h days
        flow_adjustment = daily_net_delta * h
        scenario_pred = max(0.0, current_care_load + flow_adjustment)
        
        # Blend ML base prediction with dynamic flow adjustment
        # Weighting: 40% ML baseline + 60% dynamic cumulative physical flow shock
        blended_pred = 0.35 * (base_pred + flow_adjustment * 0.4) + 0.65 * scenario_pred
        
        # Scenario uncertainty expands with shock magnitude
        shock_volatility = std * (1.0 + 0.01 * (abs(inflow_pct_change) + abs(discharge_pct_change)))
        
        sim_rows.append({
            'horizon_days': h,
            'base_forecast': round(base_pred, 1),
            'point_forecast': round(blended_pred, 1),
            'lower_80': round(max(0.0, blended_pred - 1.28 * shock_volatility), 1),
            'upper_80': round(blended_pred + 1.28 * shock_volatility, 1),
            'lower_95': round(max(0.0, blended_pred - 1.96 * shock_volatility), 1),
            'upper_95': round(blended_pred + 1.96 * shock_volatility, 1),
            'pred_std': round(shock_volatility, 1)
        })
        
    return pd.DataFrame(sim_rows)
