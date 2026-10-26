"""
Exploratory Data Analysis and Time-Series Diagnostics Module
UAC Program Care Load & Placement Demand Forecasting
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

from src.data_loader import load_raw_data, create_calendar_aligned_series

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def generate_eda_summary(df: pd.DataFrame) -> dict:
    """
    Computes statistical summaries, stationarity tests, and empirical quantiles.
    """
    care_load = df['care_load_hhs'].dropna()
    adf_result = adfuller(care_load)
    
    summary = {
        'total_calendar_days': len(df),
        'min_date': str(df['date'].min().date()),
        'max_date': str(df['date'].max().date()),
        'care_load_mean': float(care_load.mean()),
        'care_load_std': float(care_load.std()),
        'care_load_min': float(care_load.min()),
        'care_load_median': float(care_load.median()),
        'care_load_p75': float(care_load.quantile(0.75)),
        'care_load_p90': float(care_load.quantile(0.90)),
        'care_load_p95': float(care_load.quantile(0.95)),
        'care_load_max': float(care_load.max()),
        'recent_care_load': float(care_load.iloc[-1]),
        'adf_statistic': float(adf_result[0]),
        'adf_pvalue': float(adf_result[1]),
        'is_stationary_adf': bool(adf_result[1] < 0.05),
        'avg_daily_transfers': float(df['transfers_cbp_to_hhs'].mean()),
        'avg_daily_discharges': float(df['discharges_hhs'].mean()),
        'avg_net_flow': float(df['net_flow'].mean())
    }
    return summary

def plot_overview_trends(df: pd.DataFrame, output_dir: str = 'reports/figures') -> str:
    """Plots multi-panel historical overview of all flow metrics."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Panel 1: HHS Care Load
    axes[0].plot(df['date'], df['care_load_hhs'], color='#1f77b4', lw=2, label='HHS Care Load (Active Beds)')
    axes[0].axhline(3000, color='#d62728', linestyle='--', lw=1.5, label='Operational Capacity (3,000 beds)')
    axes[0].axhline(9762, color='#e377c2', linestyle=':', lw=1.5, label='90th Percentile Surge Capacity (9,762 beds)')
    axes[0].set_ylabel('Active Shelter Beds', fontsize=11, fontweight='bold')
    axes[0].set_title('U.S. HHS Unaccompanied Children Care Load & Capacity Thresholds (2023 - 2025)', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True)
    
    # Panel 2: Inflow vs Outflow
    axes[1].plot(df['date'], df['transfers_cbp_to_hhs'], color='#2ca02c', lw=1.5, alpha=0.8, label='Inflow: Transfers to HHS')
    axes[1].plot(df['date'], df['discharges_hhs'], color='#ff7f0e', lw=1.5, alpha=0.8, label='Outflow: Discharges to Sponsors')
    axes[1].set_ylabel('Children / Day', fontsize=11, fontweight='bold')
    axes[1].set_title('Daily System Inflow vs. Outflow Dynamics', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True)
    
    # Panel 3: Net System Pressure
    net_flow = df['net_flow']
    axes[2].fill_between(df['date'], net_flow, 0, where=(net_flow >= 0), color='#d62728', alpha=0.4, label='Net Accumulation (Transfers > Discharges)')
    axes[2].fill_between(df['date'], net_flow, 0, where=(net_flow < 0), color='#1f77b4', alpha=0.4, label='Net Discharge Relief (Discharges > Transfers)')
    axes[2].axhline(0, color='black', lw=0.8)
    axes[2].set_ylabel('Net Daily Change', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[2].set_title('Net System Pressure (Daily Inflow - Outflow)', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig1_historical_overview.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath

def plot_time_series_decomposition(df: pd.DataFrame, output_dir: str = 'reports/figures') -> str:
    """Decomposes the care load series into Trend, Weekly Seasonality, and Residuals."""
    os.makedirs(output_dir, exist_ok=True)
    ts = df.set_index('date')['care_load_hhs']
    
    decomp = seasonal_decompose(ts, model='additive', period=7)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(decomp.observed, color='#1f77b4', lw=1.5)
    axes[0].set_ylabel('Observed', fontweight='bold')
    axes[0].set_title('STL / Seasonal Additive Decomposition of HHS Care Load (Period = 7 Days)', fontsize=13, fontweight='bold')
    
    axes[1].plot(decomp.trend, color='#ff7f0e', lw=2)
    axes[1].set_ylabel('Trend', fontweight='bold')
    
    axes[2].plot(decomp.seasonal, color='#2ca02c', lw=1)
    axes[2].set_ylabel('Weekly Seasonal', fontweight='bold')
    
    axes[3].scatter(decomp.resid.index, decomp.resid, color='#d62728', s=8, alpha=0.6)
    axes[3].axhline(0, color='black', lw=0.8, linestyle='--')
    axes[3].set_ylabel('Residuals', fontweight='bold')
    axes[3].set_xlabel('Date', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig2_time_series_decomposition.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath

def plot_cross_correlation_analysis(df: pd.DataFrame, output_dir: str = 'reports/figures') -> str:
    """Computes and visualizes lead-lag relationships across system stages."""
    os.makedirs(output_dir, exist_ok=True)
    lags = np.arange(-14, 15)
    
    s_app = df['apprehensions_cbp']
    s_trans = df['transfers_cbp_to_hhs']
    s_care = df['care_load_hhs']
    s_disc = df['discharges_hhs']
    
    def ccf(x, y, max_lag=14):
        x = (x - x.mean()) / (x.std() * len(x))
        y = (y - y.mean()) / y.std()
        corr = np.correlate(x, y, mode='full')
        mid = len(corr) // 2
        return corr[mid - max_lag : mid + max_lag + 1]
    
    ccf_app_trans = ccf(s_app, s_trans, 14)
    ccf_trans_care = ccf(s_trans, s_care, 14)
    ccf_care_disc = ccf(s_care, s_disc, 14)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].stem(lags, ccf_app_trans, linefmt='#1f77b4', markerfmt='o', basefmt='k')
    axes[0].axvline(0, color='gray', linestyle='--')
    axes[0].set_title('CBP Apprehensions vs Transfers to HHS', fontweight='bold')
    axes[0].set_xlabel('Lag (Days)')
    axes[0].set_ylabel('Cross-Correlation')
    
    axes[1].stem(lags, ccf_trans_care, linefmt='#2ca02c', markerfmt='o', basefmt='k')
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].set_title('Transfers In vs Total Care Load', fontweight='bold')
    axes[1].set_xlabel('Lag (Days)')
    
    axes[2].stem(lags, ccf_care_disc, linefmt='#ff7f0e', markerfmt='o', basefmt='k')
    axes[2].axvline(0, color='gray', linestyle='--')
    axes[2].set_title('Care Load vs Discharges', fontweight='bold')
    axes[2].set_xlabel('Lag (Days)')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig3_cross_correlation.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath

def plot_capacity_distribution(df: pd.DataFrame, output_dir: str = 'reports/figures') -> str:
    """Plots histogram, KDE, and percentile markers for shelter capacity planning."""
    os.makedirs(output_dir, exist_ok=True)
    care = df['care_load_hhs']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(care, kde=True, color='#1f77b4', bins=40, ax=ax, alpha=0.4)
    
    p50 = care.median()
    p75 = care.quantile(0.75)
    p90 = care.quantile(0.90)
    p95 = care.quantile(0.95)
    op_thresh = 3000
    
    ax.axvline(op_thresh, color='#d62728', linestyle='--', lw=2, label=f'Operational Baseline ({op_thresh:,.0f} beds)')
    ax.axvline(p50, color='#2ca02c', linestyle='-', lw=1.5, label=f'Median ({p50:,.0f} beds)')
    ax.axvline(p75, color='#ff7f0e', linestyle=':', lw=1.5, label=f'75th Pct ({p75:,.0f} beds)')
    ax.axvline(p90, color='#9467bd', linestyle='--', lw=2, label=f'90th Pct ({p90:,.0f} beds)')
    ax.axvline(p95, color='#8c564b', linestyle='-.', lw=2, label=f'95th Pct ({p95:,.0f} beds)')
    
    ax.set_title('Empirical Distribution of HHS Care Load & Capacity Benchmark Thresholds', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Children in HHS Care (Beds)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Observation Count (Days)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig4_capacity_distribution.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath

def run_eda_pipeline(data_path: str = 'data/processed/clean_uac_data.parquet') -> dict:
    """Runs all EDA components and returns analytical summary."""
    df = pd.read_parquet(data_path)
    summary = generate_eda_summary(df)
    
    f1 = plot_overview_trends(df)
    f2 = plot_time_series_decomposition(df)
    f3 = plot_cross_correlation_analysis(df)
    f4 = plot_capacity_distribution(df)
    
    summary['figures'] = [f1, f2, f3, f4]
    return summary

if __name__ == '__main__':
    res = run_eda_pipeline()
    print("EDA Pipeline execution completed successfully.")
    for k, v in res.items():
        if k != 'figures':
            print(f"  {k}: {v}")
    print(f"Generated figures: {res['figures']}")
