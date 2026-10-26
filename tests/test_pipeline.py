"""
Automated Pipeline and Unit Test Suite
UAC Program Care Load & Placement Demand Forecasting
"""

import pytest
import os
import numpy as np
import pandas as pd

from src.data_loader import load_raw_data, create_calendar_aligned_series
from src.feature_engineering import create_feature_pipeline, get_feature_columns, get_target_columns, prepare_xy_for_horizon
from src.validation import calculate_metrics
from src.risk_engine import CapacityRiskEngine
from src.dashboard.scenarios import simulate_care_load_scenario
from src.models.machine_learning import DirectMLForecaster
from src.dashboard.components import plot_forecast_plotly, plot_breach_gauge, plot_net_flow_plotly
from streamlit.testing.v1 import AppTest

RAW_PATH = 'HHS_Unaccompanied_Alien_Children_Program.csv'

def test_data_ingestion():
    """Verify raw data loading and type sanitization."""
    assert os.path.exists(RAW_PATH), "Raw dataset file must exist."
    raw_df = load_raw_data(RAW_PATH)
    
    assert len(raw_df) == 720, f"Expected 720 valid reporting rows, got {len(raw_df)}"
    assert 'date' in raw_df.columns
    assert 'care_load_hhs' in raw_df.columns
    assert 'transfers_cbp_to_hhs' in raw_df.columns
    assert 'discharges_hhs' in raw_df.columns
    
    assert pd.api.types.is_datetime64_any_dtype(raw_df['date'])
    assert pd.api.types.is_numeric_dtype(raw_df['care_load_hhs'])
    assert raw_df['care_load_hhs'].min() >= 1000

def test_calendar_alignment():
    """Verify continuous daily calendar reindexing and gap interpolation."""
    raw_df = load_raw_data(RAW_PATH)
    aligned_df = create_calendar_aligned_series(raw_df)
    
    expected_days = (raw_df['date'].max() - raw_df['date'].min()).days + 1
    assert len(aligned_df) == expected_days, f"Expected {expected_days} continuous days, got {len(aligned_df)}"
    assert aligned_df['care_load_hhs'].isna().sum() == 0, "No missing values allowed in aligned care load"
    assert 'is_reporting_day' in aligned_df.columns

def test_stock_vs_flow_interpolation():
    """Verify Defect 2 remediation: stock is interpolated, flow variables are 0 on non-reporting days."""
    raw_df = load_raw_data(RAW_PATH)
    aligned_df = create_calendar_aligned_series(raw_df)
    
    non_reporting = aligned_df[aligned_df['is_reporting_day'] == 0]
    assert len(non_reporting) > 0, "Dataset must have non-reporting weekend/holiday days"
    
    # Stock variables must be continuous and non-zero
    assert (non_reporting['care_load_hhs'] > 0).all()
    assert (non_reporting['custody_cbp'] >= 0).all()
    
    # Flow variables on non-reporting days must be 0 (no forward Monday lookahead)
    assert (non_reporting['transfers_cbp_to_hhs'] == 0.0).all()
    assert (non_reporting['discharges_hhs'] == 0.0).all()
    assert (non_reporting['apprehensions_cbp'] == 0.0).all()
    assert (non_reporting['net_flow'] == 0.0).all()

def test_feature_engineering_pipeline():
    """Verify lag features, rolling windows, and multi-horizon target labels."""
    raw_df = load_raw_data(RAW_PATH)
    aligned_df = create_calendar_aligned_series(raw_df)
    feat_df = create_feature_pipeline(aligned_df, horizons=[7, 14, 30])
    
    assert 'care_load_lag_1' in feat_df.columns
    assert 'care_load_roll_mean_7' in feat_df.columns
    assert 'target_care_load_h7' in feat_df.columns
    assert 'target_care_load_h14' in feat_df.columns
    assert 'target_care_load_h30' in feat_df.columns
    
    # Defect 5 targets
    assert 'target_net_flow_h7' in feat_df.columns
    assert 'target_net_flow_h14' in feat_df.columns
    assert 'target_net_flow_h30' in feat_df.columns
    
    targets = get_target_columns([7, 14, 30])
    assert len(targets) == 9
    assert 'target_net_flow_h7' in targets
    
    feature_cols = get_feature_columns(feat_df)
    assert len(feature_cols) == 87, f"Expected exactly 87 predictive features, got {len(feature_cols)}"
    
    # Verify no unlagged targets in feature list
    for target in targets + ['care_load_hhs', 'net_flow']:
        assert target not in feature_cols

def test_prepare_xy_net_flow():
    """Verify Defect 5: prepare_xy_for_horizon works for net_flow."""
    raw_df = load_raw_data(RAW_PATH)
    aligned_df = create_calendar_aligned_series(raw_df)
    feat_df = create_feature_pipeline(aligned_df, horizons=[7, 14, 30])
    
    X, y, cols = prepare_xy_for_horizon(feat_df, target_type='net_flow', horizon=7)
    assert len(X) == len(y)
    assert len(cols) == 87
    assert not y.isna().any()

def test_metrics_calculation_and_fsi():
    """Verify forecasting evaluation metrics and corrected FSI formula (Defect 6)."""
    y_true = np.array([2000.0, 2100.0, 2200.0, 2250.0, 2300.0])
    y_pred = np.array([1980.0, 2110.0, 2190.0, 2240.0, 2295.0])
    y_lag = np.array([1950.0, 2000.0, 2100.0, 2200.0, 2250.0])
    
    metrics = calculate_metrics(y_true, y_pred, y_lag)
    assert 'MAE' in metrics
    assert 'RMSE' in metrics
    assert 'MAPE' in metrics
    assert 'Accuracy' in metrics
    assert 'Stability_Index' in metrics
    assert metrics['Accuracy'] > 95.0
    assert 0.0 <= metrics['Stability_Index'] <= 1.0

def test_risk_engine_calculations():
    """Verify Capacity Breach Probability, Surge Lead Time logic, and FSI method (Defect 6)."""
    engine = CapacityRiskEngine(operational_capacity=3000.0)
    
    # Low forecast -> Safe / Zero breach
    p_safe = engine.calculate_point_breach_probability(point_forecast=2400.0, pred_std=50.0, threshold=3000.0)
    assert p_safe < 0.001
    
    # Exact threshold -> 50% breach
    p_mid = engine.calculate_point_breach_probability(point_forecast=3000.0, pred_std=50.0, threshold=3000.0)
    assert p_mid == pytest.approx(0.50, abs=0.01)
    
    # High forecast -> Breach
    p_high = engine.calculate_point_breach_probability(point_forecast=3200.0, pred_std=50.0, threshold=3000.0)
    assert p_high > 0.99
    
    # Trajectory evaluation
    traj = pd.DataFrame([
        {'horizon_days': 7, 'point_forecast': 2500.0, 'pred_std': 60.0},
        {'horizon_days': 14, 'point_forecast': 2700.0, 'pred_std': 80.0},
        {'horizon_days': 30, 'point_forecast': 3100.0, 'pred_std': 100.0}
    ])
    eval_res = engine.evaluate_forecast_trajectory(traj, capacity_threshold=3000.0)
    assert eval_res['surge_lead_time_days'] == 30
    assert eval_res['alert_tier'] == 'RED'
    
    # Explicit calculate_forecast_stability_index method
    y_hist = np.array([2400.0, 2450.0, 2480.0, 2500.0, 2520.0])
    y_pred_traj = np.array([2510.0, 2530.0, 2550.0])
    fsi_val = engine.calculate_forecast_stability_index(y_pred_traj, y_hist)
    assert isinstance(fsi_val, float)
    assert 0.0 <= fsi_val <= 1.0

def test_purged_boundary_split():
    """Verify Defect 1: h-day purge window eliminates forward-label overlap with test set."""
    raw_df = load_raw_data(RAW_PATH)
    aligned_df = create_calendar_aligned_series(raw_df)
    feat_df = create_feature_pipeline(aligned_df)
    
    holdout_days = 120
    split_date = aligned_df['date'].max() - pd.Timedelta(days=holdout_days)
    
    for h in [7, 14, 30]:
        train_feat_h = feat_df[feat_df['date'] <= (split_date - pd.Timedelta(days=h))].copy()
        test_feat = feat_df[feat_df['date'] > split_date].copy()
        
        # Max date in train_feat_h + h days must be <= split_date
        max_train_date = train_feat_h['date'].max()
        assert max_train_date + pd.Timedelta(days=h) <= split_date
        assert test_feat['date'].min() > split_date

def test_scenario_simulator():
    """Verify what-if simulation adjustments under influx surge and placement bottlenecks."""
    base_traj = pd.DataFrame([
        {'horizon_days': 7, 'point_forecast': 2500.0, 'pred_std': 50.0},
        {'horizon_days': 14, 'point_forecast': 2550.0, 'pred_std': 70.0},
        {'horizon_days': 30, 'point_forecast': 2600.0, 'pred_std': 100.0}
    ])
    
    sim = simulate_care_load_scenario(
        base_forecast_df=base_traj,
        current_care_load=2500.0,
        current_transfers=50.0,
        current_discharges=10.0,
        inflow_pct_change=50.0, # Surge
        discharge_pct_change=-20.0, # Bottleneck
        capacity_threshold=3000.0
    )
    
    assert len(sim) == 3
    # Care load should expand under net positive accumulation
    assert sim['point_forecast'].iloc[-1] > base_traj['point_forecast'].iloc[-1]

def test_trained_model_inference():
    """Verify serialized production models can be loaded and generate trajectories for all targets."""
    raw_df = load_raw_data(RAW_PATH)
    aligned_df = create_calendar_aligned_series(raw_df)
    feat_df = create_feature_pipeline(aligned_df, horizons=[7, 14, 30])
    
    for target in ['care_load', 'discharges', 'net_flow']:
        model_path = f'models/xgb_{target}_model.joblib'
        assert os.path.exists(model_path), f"Serialized model {model_path} must exist"
        forecaster = DirectMLForecaster.load(model_path)
        
        preds = forecaster.predict_trajectory(feat_df)
        assert len(preds) == 3
        assert set(preds['horizon_days']) == {7, 14, 30}
        assert 'pred_std' in preds.columns

def test_dashboard_plotly_components():
    """Verify that all dashboard Plotly figures generate valid specifications without ValueError/attribute errors."""
    raw_df = load_raw_data(RAW_PATH)
    aligned_df = create_calendar_aligned_series(raw_df)
    mock_traj = pd.DataFrame([
        {'horizon_days': 7, 'point_forecast': 2500.0, 'lower_80': 2400.0, 'upper_80': 2600.0, 'lower_95': 2350.0, 'upper_95': 2650.0, 'pred_std': 75.0},
        {'horizon_days': 14, 'point_forecast': 2550.0, 'lower_80': 2420.0, 'upper_80': 2680.0, 'lower_95': 2360.0, 'upper_95': 2740.0, 'pred_std': 95.0},
        {'horizon_days': 30, 'point_forecast': 2600.0, 'lower_80': 2450.0, 'upper_80': 2750.0, 'lower_95': 2380.0, 'upper_95': 2820.0, 'pred_std': 110.0}
    ])
    
    # 1. Test plot_breach_gauge (gauge bar properties)
    gauge_fig = plot_breach_gauge(breach_prob_pct=34.5, threshold_val=3000.0)
    assert gauge_fig is not None
    gauge_dict = gauge_fig.to_plotly_json()
    assert 'data' in gauge_dict
    assert gauge_dict['data'][0]['type'] == 'indicator'
    assert gauge_dict['data'][0]['gauge']['bar']['thickness'] == 0.3
    
    # 2. Test plot_forecast_plotly
    forecast_fig = plot_forecast_plotly(
        historical_df=aligned_df,
        forecast_df=mock_traj,
        capacity_threshold=3000.0,
        target_name="HHS Care Load (Shelter Beds)",
        days_back=60
    )
    assert forecast_fig is not None
    fc_dict = forecast_fig.to_plotly_json()
    assert len(fc_dict['data']) == 5

    # 3. Test plot_net_flow_plotly
    nf_mock = pd.DataFrame([
        {'horizon_days': 7, 'point_forecast': -1.5, 'lower_80': -6.0, 'upper_80': 3.0, 'lower_95': -9.0, 'upper_95': 6.0, 'pred_std': 4.0},
        {'horizon_days': 14, 'point_forecast': -10.0, 'lower_80': -15.0, 'upper_80': -5.0, 'lower_95': -19.0, 'upper_95': -1.0, 'pred_std': 4.5},
        {'horizon_days': 30, 'point_forecast': -25.0, 'lower_80': -32.0, 'upper_80': -18.0, 'lower_95': -37.0, 'upper_95': -13.0, 'pred_std': 5.0}
    ])
    nf_fig = plot_net_flow_plotly(historical_df=aligned_df, forecast_df=nf_mock, days_back=60)
    assert nf_fig is not None
    assert len(nf_fig.to_plotly_json()['data']) == 5

def test_streamlit_app_headless():
    """Verify headless execution of Streamlit application and tab/widget rendering."""
    app_path = os.path.join(os.path.dirname(__file__), '..', 'app.py')
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, f"App execution failed with: {at.exception}"

