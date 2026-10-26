"""
Data Ingestion and Calendar Preprocessing Module
UAC Program Care Load & Placement Demand Forecasting
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict

# Standard canonical column mapping
COLUMN_MAPPING = {
    'Date': 'date',
    'Children apprehended and placed in CBP custody*': 'apprehensions_cbp',
    'Children apprehended and placed in CBP custody': 'apprehensions_cbp',
    'Children in CBP custody': 'custody_cbp',
    'Children transferred out of CBP custody': 'transfers_cbp_to_hhs',
    'Children in HHS Care': 'care_load_hhs',
    'Children discharged from HHS Care': 'discharges_hhs'
}

HUMAN_LABEL_MAPPING = {
    'date': 'Reporting Date',
    'apprehensions_cbp': 'CBP Apprehensions',
    'custody_cbp': 'CBP Active Custody',
    'transfers_cbp_to_hhs': 'Transfers to HHS',
    'care_load_hhs': 'HHS Care Load (Shelter Beds)',
    'discharges_hhs': 'HHS Discharges (Sponsor Placements)',
    'net_flow': 'Net Flow (Transfers - Discharges)'
}

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV data, sanitizes column names, removes comma formatting,
    and converts fields to numeric and datetime representations.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")

    # Read raw CSV
    df = pd.read_csv(filepath)
    df = df.dropna(how='all')

    # Normalize column names
    clean_cols = {}
    for col in df.columns:
        norm_col = col.strip().replace('*', '')
        # Match against dictionary
        matched = False
        for k, v in COLUMN_MAPPING.items():
            if k.strip().replace('*', '').lower() == norm_col.lower():
                clean_cols[col] = v
                matched = True
                break
        if not matched:
            clean_cols[col] = norm_col.lower().replace(' ', '_')

    df = df.rename(columns=clean_cols)

    # Clean numeric fields
    numeric_cols = ['apprehensions_cbp', 'custody_cbp', 'transfers_cbp_to_hhs', 'care_load_hhs', 'discharges_hhs']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    # Net flow: Transfers - Discharges
    df['net_flow'] = df['transfers_cbp_to_hhs'] - df['discharges_hhs']

    return df

def create_calendar_aligned_series(df: pd.DataFrame, fill_method: str = 'interpolate') -> pd.DataFrame:
    """
    Reconciles missing reporting days (weekends/holidays) into a continuous daily time series.
    - Stock variable (care_load_hhs): Interpolated smoothly since children remain in care.
    - Flow variables (transfers, discharges, apprehensions): Interpolated smoothly with day-of-week context.
    - Preserves indicator of original reporting dates via 'is_reporting_day'.
    """
    df_sorted = df.drop_duplicates(subset=['date']).sort_values('date').set_index('date')
    
    # Generate continuous daily date range
    start_date = df_sorted.index.min()
    end_date = df_sorted.index.max()
    full_idx = pd.date_range(start=start_date, end=end_date, freq='D', name='date')
    
    # Reindex
    aligned = df_sorted.reindex(full_idx)
    
    # Create reporting day flag before interpolation
    aligned['is_reporting_day'] = (~aligned['care_load_hhs'].isna()).astype(int)
    
    # Apply interpolation
    if fill_method == 'interpolate':
        # Stock variables: Linear time interpolation for continuous physical quantities
        stock_cols = ['care_load_hhs', 'custody_cbp']
        for col in stock_cols:
            if col in aligned.columns:
                aligned[col] = aligned[col].interpolate(method='time').bfill().ffill().round(1)
        
        # Flow variables: Non-reporting days set to 0.0 to eliminate forward lookahead
        # (accumulated counts are recorded on reporting days)
        flow_cols = ['transfers_cbp_to_hhs', 'discharges_hhs', 'apprehensions_cbp']
        for col in flow_cols:
            if col in aligned.columns:
                aligned[col] = aligned[col].fillna(0.0).round(1)
    elif fill_method == 'forward_fill':
        aligned = aligned.ffill()
    
    # Recompute net flow
    aligned['net_flow'] = aligned['transfers_cbp_to_hhs'] - aligned['discharges_hhs']
    
    # Reset index and add calendar features
    aligned = aligned.reset_index()
    aligned['day_of_week'] = aligned['date'].dt.dayofweek
    aligned['is_weekend'] = aligned['day_of_week'].isin([5, 6]).astype(int)
    aligned['month'] = aligned['date'].dt.month
    aligned['year'] = aligned['date'].dt.year
    aligned['day_of_month'] = aligned['date'].dt.day
    aligned['quarter'] = aligned['date'].dt.quarter
    
    return aligned

def prepare_and_save_data(raw_filepath: str, output_dir: str = 'data/processed') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end data processing function that outputs both reporting-day only
    and calendar-aligned datasets to data/processed/.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_df = load_raw_data(raw_filepath)
    aligned_df = create_calendar_aligned_series(raw_df, fill_method='interpolate')
    
    raw_save_path = os.path.join(output_dir, 'reporting_days_uac.csv')
    aligned_csv_path = os.path.join(output_dir, 'calendar_aligned_uac.csv')
    aligned_parquet_path = os.path.join(output_dir, 'clean_uac_data.parquet')
    
    raw_df.to_csv(raw_save_path, index=False)
    aligned_df.to_csv(aligned_csv_path, index=False)
    aligned_df.to_parquet(aligned_parquet_path, index=False)
    
    return raw_df, aligned_df

if __name__ == '__main__':
    raw_path = 'HHS_Unaccompanied_Alien_Children_Program.csv'
    raw_df, aligned_df = prepare_and_save_data(raw_path)
    print(f"Data ingestion successful.")
    print(f"Reporting days records: {len(raw_df)}, Date range: {raw_df['date'].min()} to {raw_df['date'].max()}")
    print(f"Calendar aligned records: {len(aligned_df)}, Date range: {aligned_df['date'].min()} to {aligned_df['date'].max()}")
    print("Columns:", aligned_df.columns.tolist())
