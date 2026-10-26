"""
Model Training, Benchmarking, and Serialization Pipeline
UAC Program Care Load & Placement Demand Forecasting
"""

import os
import joblib
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List

from src.data_loader import prepare_and_save_data
from src.feature_engineering import create_feature_pipeline, prepare_xy_for_horizon
from src.validation import calculate_metrics, get_time_series_splits
from src.models.baselines import NaivePersistenceModel, MovingAverageModel
from src.models.statistical import HoltWintersModel, ARIMAModel
from src.models.machine_learning import DirectMLForecaster

# Suppress statsmodels convergence and frequency warnings during batch benchmarking
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def benchmark_models(data_path: str = 'HHS_Unaccompanied_Alien_Children_Program.csv', output_dir: str = 'models') -> pd.DataFrame:
    """
    Executes walk-forward evaluation across all model candidates for horizons 7, 14, 30 days.
    Trains and serializes final models to disk.
    Remediates:
      - Defect 1: Applies h-day purge window on train split (train_feat_h = feat_df[date <= split_date - h])
      - Defect 3: Fixes Holt-Winters static scalar broadcast with rolling out-of-sample forecasting
      - Defect 4: Evaluates SARIMAX / ARIMAModel across all horizons (7, 14, 30 days)
      - Defect 5: Evaluates direct models for net flow imbalance (target_net_flow_h{h})
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("Step 1: Ingesting and aligning data...")
    raw_df, aligned_df = prepare_and_save_data(data_path)
    
    print("Step 2: Generating multi-horizon feature matrix...")
    horizons = [7, 14, 30]
    feat_df = create_feature_pipeline(aligned_df, horizons=horizons)
    
    # Target configurations to benchmark: care_load, discharges, and net_flow
    target_configs = [
        ('care_load', 'Care Load', 'care_load_hhs'),
        ('discharges', 'Discharge Demand', 'discharges_hhs'),
        ('net_flow', 'Net Flow Imbalance', 'net_flow')
    ]
    results = []
    
    # 1. Evaluate on chronological out-of-sample holdout (last 120 days)
    holdout_days = 120
    split_date = aligned_df['date'].max() - pd.Timedelta(days=holdout_days)
    test_feat = feat_df[feat_df['date'] > split_date].copy()
    
    print(f"Step 3: Evaluating models on holdout window ({split_date.date()} to {aligned_df['date'].max().date()})...")
    
    hist_series_dict = {
        col: aligned_df.set_index('date')[col]
        for col in ['care_load_hhs', 'discharges_hhs', 'net_flow']
    }
    
    for target_type, target_name, raw_col in target_configs:
        print(f"\n--- Benchmarking Target: {target_name} ({target_type}) ---")
        
        for h in horizons:
            target_col = f'target_{target_type}_h{h}'
            
            # Defect 1: Apply h-day purge window to prevent boundary leakage into test_feat
            purge_date = split_date - pd.Timedelta(days=h)
            train_feat_h = feat_df[feat_df['date'] <= purge_date].copy()
            
            # Out-of-sample test target and baseline lag
            valid_test = test_feat.dropna(subset=[target_col]).copy()
            y_true = valid_test[target_col].values
            y_base_lag = valid_test[f'{target_type}_lag_1'].values[:len(y_true)]
            
            # Baseline 1: Naïve Persistence
            y_pred_naive = valid_test[raw_col].values[:len(y_true)]
            m_naive = calculate_metrics(y_true, y_pred_naive, y_base_lag)
            results.append({
                'Target': target_name,
                'Horizon': f'{h} Days',
                'Model': 'Naïve Persistence',
                'Model_Family': 'Baseline',
                **m_naive
            })
            
            # Baseline 2: 7-Day Moving Average
            y_pred_ma7 = valid_test[f'{target_type}_roll_mean_7'].values[:len(y_true)]
            m_ma7 = calculate_metrics(y_true, y_pred_ma7, y_base_lag)
            results.append({
                'Target': target_name,
                'Horizon': f'{h} Days',
                'Model': 'Moving Average (7d)',
                'Model_Family': 'Baseline',
                **m_ma7
            })
            
            # Statistical 1: Holt-Winters (Exponential Smoothing)
            # Defect 3: Rolling out-of-sample forecasts instead of static scalar broadcast
            hw = HoltWintersModel()
            hw_preds_list = []
            hist_s = hist_series_dict[raw_col]
            for i, d in enumerate(valid_test['date']):
                if i == 0 or i % 7 == 0:
                    s_train = hist_s.loc[:d]
                    hw.fit(s_train.iloc[-180:])
                p, _, _ = hw.predict(h)
                hw_preds_list.append(p[-1])
            y_pred_hw = np.array(hw_preds_list)
            m_hw = calculate_metrics(y_true, y_pred_hw, y_base_lag)
            results.append({
                'Target': target_name,
                'Horizon': f'{h} Days',
                'Model': 'Holt-Winters Exp Smoothing',
                'Model_Family': 'Statistical',
                **m_hw
            })
            
            # Statistical 2: SARIMAX / ARIMAModel
            # Defect 4: Execute SARIMAX across all horizons (7, 14, 30 days)
            arima = ARIMAModel()
            arima_preds_list = []
            for i, d in enumerate(valid_test['date']):
                if i == 0 or i % 7 == 0:
                    s_train = hist_s.loc[:d]
                    arima.fit(s_train.iloc[-180:])
                p, _, _ = arima.predict(h)
                arima_preds_list.append(p[-1])
            y_pred_arima = np.array(arima_preds_list)
            m_arima = calculate_metrics(y_true, y_pred_arima, y_base_lag)
            results.append({
                'Target': target_name,
                'Horizon': f'{h} Days',
                'Model': 'SARIMAX (ARIMA)',
                'Model_Family': 'Statistical',
                **m_arima
            })
            
            # ML Models: Random Forest, Gradient Boosting, XGBoost
            for ml_name, ml_type in [
                ('Random Forest Regressor', 'rf'),
                ('Gradient Boosting Regressor', 'gbr'),
                ('XGBoost Regressor', 'xgboost')
            ]:
                X_train, y_train, _ = prepare_xy_for_horizon(train_feat_h, target_type=target_type, horizon=h)
                X_test, y_test_eval, feat_cols = prepare_xy_for_horizon(test_feat, target_type=target_type, horizon=h)
                
                forecaster = DirectMLForecaster(model_type=ml_type, target_type=target_type, horizons=[h])
                forecaster.fit(train_feat_h)
                
                # Predict on test features
                preds = forecaster.models[h].predict(X_test)
                m_ml = calculate_metrics(y_test_eval.values, preds, X_test[f'{target_type}_lag_1'].values)
                results.append({
                    'Target': target_name,
                    'Horizon': f'{h} Days',
                    'Model': ml_name,
                    'Model_Family': 'Machine Learning',
                    **m_ml
                })

    df_results = pd.DataFrame(results)
    benchmark_path = 'reports/benchmark_metrics.csv'
    df_results.to_csv(benchmark_path, index=False)
    print(f"\nBenchmark results successfully saved to: {benchmark_path} ({len(df_results)} rows)")
    
    # Step 4: Fit and serialize final production models on 100% available data
    # Defect 5: Serialize models for care_load, discharges, AND net_flow
    print("\nStep 4: Training production models on full timeline...")
    for target in ['care_load', 'discharges', 'net_flow']:
        for ml_type, ml_filename in [('xgboost', f'xgb_{target}_model.joblib'), ('rf', f'rf_{target}_model.joblib')]:
            prod_forecaster = DirectMLForecaster(model_type=ml_type, target_type=target, horizons=horizons)
            prod_forecaster.fit(feat_df)
            save_dest = os.path.join(output_dir, ml_filename)
            prod_forecaster.save(save_dest)
            print(f"  Serialized {ml_type} for {target} to: {save_dest}")
            
    return df_results

if __name__ == '__main__':
    df_res = benchmark_models()
    print("\nBenchmark Summary (Care Load 7-Day Horizon):")
    print(df_res[(df_res['Target'] == 'Care Load') & (df_res['Horizon'] == '7 Days')][['Model', 'MAE', 'RMSE', 'MAPE', 'Accuracy']])
    print("\nBenchmark Summary (Care Load 30-Day Horizon):")
    print(df_res[(df_res['Target'] == 'Care Load') & (df_res['Horizon'] == '30 Days')][['Model', 'MAE', 'RMSE', 'MAPE', 'Accuracy']])
