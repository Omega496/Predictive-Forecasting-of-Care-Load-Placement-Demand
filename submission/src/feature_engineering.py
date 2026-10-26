"""
Feature Engineering Module for Multi-Horizon Forecasting
UAC Program Care Load & Placement Demand Forecasting
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def add_autoregressive_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Computes autoregressive lag features for care load, flows, and custody."""
    out = df.copy()
    
    # Care Load Lags
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        out[f'care_load_lag_{lag}'] = out['care_load_hhs'].shift(lag)
        
    # Flow and Placement Lags
    for lag in [1, 2, 3, 7, 14, 21]:
        out[f'transfers_lag_{lag}'] = out['transfers_cbp_to_hhs'].shift(lag)
        out[f'discharges_lag_{lag}'] = out['discharges_hhs'].shift(lag)
        out[f'net_flow_lag_{lag}'] = out['net_flow'].shift(lag)
        
    # Upstream CBP Lags
    for lag in [1, 2, 7, 14]:
        out[f'apprehensions_lag_{lag}'] = out['apprehensions_cbp'].shift(lag)
        out[f'custody_lag_{lag}'] = out['custody_cbp'].shift(lag)
        
    return out

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes rolling statistical aggregates over multiple time windows."""
    out = df.copy()
    
    # Windows: 7, 14, 30 days
    for w in [7, 14, 30]:
        # Care load statistics (shifted by 1 to prevent lookahead)
        care_shifted = out['care_load_hhs'].shift(1)
        out[f'care_load_roll_mean_{w}'] = care_shifted.rolling(w).mean()
        out[f'care_load_roll_std_{w}'] = care_shifted.rolling(w).std()
        out[f'care_load_roll_min_{w}'] = care_shifted.rolling(w).min()
        out[f'care_load_roll_max_{w}'] = care_shifted.rolling(w).max()
        out[f'care_load_roll_range_{w}'] = out[f'care_load_roll_max_{w}'] - out[f'care_load_roll_min_{w}']
        
        # Transfers statistics
        trans_shifted = out['transfers_cbp_to_hhs'].shift(1)
        out[f'transfers_roll_mean_{w}'] = trans_shifted.rolling(w).mean()
        out[f'transfers_roll_sum_{w}'] = trans_shifted.rolling(w).sum()
        
        # Discharges statistics
        disc_shifted = out['discharges_hhs'].shift(1)
        out[f'discharges_roll_mean_{w}'] = disc_shifted.rolling(w).mean()
        out[f'discharges_roll_sum_{w}'] = disc_shifted.rolling(w).sum()
        
        # Net flow statistics
        net_shifted = out['net_flow'].shift(1)
        out[f'net_flow_roll_mean_{w}'] = net_shifted.rolling(w).mean()
        out[f'net_flow_roll_sum_{w}'] = net_shifted.rolling(w).sum()
        
    # Momentum & Velocity Features
    out['care_load_velocity_7_14'] = out['care_load_roll_mean_7'] - out['care_load_roll_mean_14']
    out['care_load_velocity_14_30'] = out['care_load_roll_mean_14'] - out['care_load_roll_mean_30']
    out['net_flow_momentum_7'] = out['net_flow_lag_1'] - out['net_flow_roll_mean_7']
    
    return out

def add_flow_dynamics_and_upstream(df: pd.DataFrame) -> pd.DataFrame:
    """Computes system equilibrium ratios, pressure indicators, and upstream conversion proxies."""
    out = df.copy()
    
    # Inflow-to-discharge pressure ratio
    out['flow_pressure_ratio_7'] = (out['transfers_roll_sum_7'] + 1.0) / (out['discharges_roll_sum_7'] + 1.0)
    out['flow_pressure_ratio_14'] = (out['transfers_roll_sum_14'] + 1.0) / (out['discharges_roll_sum_14'] + 1.0)
    
    # CBP Custody to Transfer conversion rate proxy
    out['cbp_transfer_velocity'] = out['transfers_lag_1'] / (out['custody_lag_1'] + 1.0)
    out['cbp_apprehension_to_custody'] = out['custody_lag_1'] / (out['apprehensions_lag_1'] + 1.0)
    
    # Day-over-day acceleration
    out['transfers_diff_1'] = out['transfers_lag_1'] - out['transfers_lag_2']
    out['discharges_diff_1'] = out['discharges_lag_1'] - out['discharges_lag_2']
    out['flow_acceleration'] = out['transfers_diff_1'] - out['discharges_diff_1']
    
    return out

def add_calendar_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes cyclical seasonal transformations for calendar time."""
    out = df.copy()
    
    # Day of week cyclical encoding
    dow = out['date'].dt.dayofweek
    out['sin_dow'] = np.sin(2 * np.pi * dow / 7.0)
    out['cos_dow'] = np.cos(2 * np.pi * dow / 7.0)
    
    # Day of year cyclical encoding
    doy = out['date'].dt.dayofyear
    out['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    out['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
    
    return out

def build_multi_horizon_targets(df: pd.DataFrame, horizons: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """Adds forward-looking multi-horizon target labels for care load, discharge demand, and net flow."""
    out = df.copy()
    for h in horizons:
        out[f'target_care_load_h{h}'] = out['care_load_hhs'].shift(-h)
        out[f'target_discharges_h{h}'] = out['discharges_hhs'].shift(-h)
        out[f'target_net_flow_h{h}'] = out['net_flow'].shift(-h)
    return out

def get_target_columns(horizons: List[int] = [7, 14, 30]) -> List[str]:
    """Returns list of canonical multi-horizon target column names across care load, discharges, and net flow."""
    target_types = ['care_load', 'discharges', 'net_flow']
    return [f'target_{t}_h{h}' for t in target_types for h in horizons]

def create_feature_pipeline(df: pd.DataFrame, horizons: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """
    Executes complete end-to-end feature pipeline, assembling lags, rolling stats,
    flow dynamics, cyclical features, and forward targets.
    """
    df_feat = add_autoregressive_lags(df)
    df_feat = add_rolling_features(df_feat)
    df_feat = add_flow_dynamics_and_upstream(df_feat)
    df_feat = add_calendar_cyclical_features(df_feat)
    df_feat = build_multi_horizon_targets(df_feat, horizons=horizons)
    
    return df_feat

def get_feature_columns(df_feat: pd.DataFrame) -> List[str]:
    """Returns list of predictive feature names, excluding dates and target labels."""
    excluded_prefixes = ['target_', 'date']
    excluded_exact = ['target_care_load_h7', 'target_care_load_h14', 'target_care_load_h30',
                      'target_discharges_h7', 'target_discharges_h14', 'target_discharges_h30',
                      'target_net_flow_h7', 'target_net_flow_h14', 'target_net_flow_h30',
                      'date']
    
    feature_cols = []
    for col in df_feat.columns:
        if col in excluded_exact:
            continue
        if any(col.startswith(p) for p in excluded_prefixes):
            continue
        if col in ['care_load_hhs', 'discharges_hhs', 'transfers_cbp_to_hhs', 'custody_cbp', 'apprehensions_cbp', 'net_flow']:
            # Contemporary target values at time t are NOT features when forecasting into the future without shifting,
            # but lag_1, roll_mean etc are already shifted properly. To avoid leak, exclude contemporary unlagged values!
            continue
        feature_cols.append(col)
        
    return feature_cols

def prepare_xy_for_horizon(df_feat: pd.DataFrame, target_type: str = 'care_load', horizon: int = 7) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepares clean (X, y) matrix for a given target ('care_load', 'discharges', or 'net_flow')
    and horizon (7, 14, or 30), dropping unpopulated initial lag rows and final forward target rows.
    """
    target_col = f'target_{target_type}_h{horizon}'
    if target_col not in df_feat.columns:
        raise ValueError(f"Target {target_col} not found in dataframe.")
        
    feature_cols = get_feature_columns(df_feat)
    
    # Valid subset where both features and target are non-null
    valid_mask = df_feat[feature_cols + [target_col]].notna().all(axis=1)
    df_valid = df_feat[valid_mask].copy()
    
    X = df_valid[feature_cols]
    y = df_valid[target_col]
    
    return X, y, feature_cols

if __name__ == '__main__':
    from src.data_loader import prepare_and_save_data
    raw_df, aligned_df = prepare_and_save_data('HHS_Unaccompanied_Alien_Children_Program.csv')
    feat_df = create_feature_pipeline(aligned_df, horizons=[7, 14, 30])
    
    print(f"Total features created: {feat_df.shape[1]}")
    features = get_feature_columns(feat_df)
    print(f"Predictive feature count: {len(features)}")
    print("Sample feature names:", features[:10])
    
    for h in [7, 14, 30]:
        X, y, _ = prepare_xy_for_horizon(feat_df, target_type='care_load', horizon=h)
        print(f"Horizon {h}d Care Load: X shape = {X.shape}, y shape = {y.shape}")
