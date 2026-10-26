"""
Reusable Dashboard UI Components, Plotly Visualizations, and Theme Utilities
UAC Program Care Load & Placement Demand Forecasting
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional

# Custom CSS for styling
CUSTOM_CSS = """
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        color: #f8fafc;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 4px;
    }
    .alert-banner {
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 12px;
    }
</style>
"""

def inject_custom_css():
    """Injects custom CSS styling into Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_alert_banner(alert_info: Dict[str, Any]):
    """Renders high-visibility alert banner corresponding to risk tier."""
    tier = str(alert_info.get('alert_tier', 'GREEN')).upper()
    tier_icons = {
        'GREEN': '✅',
        'YELLOW': '⚠️',
        'ORANGE': '⚡',
        'RED': '🚨'
    }
    icon = tier_icons.get(tier, 'ℹ️')
    
    color = alert_info.get('alert_color', '#2ecc71')
    title = alert_info.get('alert_title', 'NORMAL OPERATIONS')
    msg = alert_info.get('alert_message', '')
    lead_time = alert_info.get('surge_lead_time_display', 'N/A')
    
    html = f"""
    <div style="background-color: {color}22; border-left: 6px solid {color}; border-radius: 8px; padding: 14px 18px; margin: 15px 0;">
        <div style="color: {color}; font-weight: 700; font-size: 1.05rem; margin-bottom: 4px;">
            {icon} {title} (Lead Time: {lead_time})
        </div>
        <div style="color: #e2e8f0; font-size: 0.92rem;">
            {msg}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def plot_forecast_plotly(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    capacity_threshold: Optional[float] = 3000.0,
    target_name: str = "HHS Care Load (Shelter Beds)",
    target_col: str = "care_load_hhs",
    days_back: int = 90
) -> go.Figure:
    """
    Generates interactive Plotly visualization displaying recent historical trend,
    forward multi-horizon point forecast, 80% & 95% confidence intervals, and capacity threshold line.
    """
    if historical_df.empty or forecast_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No forecast data available")
        return fig

    recent_hist = historical_df.dropna(subset=['date']).tail(days_back).copy()
    if target_col not in recent_hist.columns:
        target_col = 'care_load_hhs' if 'care_load_hhs' in recent_hist.columns else recent_hist.select_dtypes(include=[np.number]).columns[0]
        
    last_date = recent_hist['date'].max()
    last_val = float(recent_hist[target_col].iloc[-1])
    
    # Construct forecast timeline
    fc_dates = [last_date]
    fc_points = [last_val]
    fc_l80 = [last_val]
    fc_u80 = [last_val]
    fc_l95 = [last_val]
    fc_u95 = [last_val]
    
    for _, row in forecast_df.sort_values('horizon_days').iterrows():
        h = int(row['horizon_days'])
        fc_dates.append(last_date + pd.Timedelta(days=h))
        fc_points.append(float(row['point_forecast']))
        fc_l80.append(float(row.get('lower_80', row['point_forecast'])))
        fc_u80.append(float(row.get('upper_80', row['point_forecast'])))
        fc_l95.append(float(row.get('lower_95', row['point_forecast'])))
        fc_u95.append(float(row.get('upper_95', row['point_forecast'])))
        
    fig = go.Figure()
    
    # 1. 95% Confidence Band
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1],
        y=fc_u95 + fc_l95[::-1],
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='95% Prediction Interval'
    ))
    
    # 2. 80% Confidence Band
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1],
        y=fc_u80 + fc_l80[::-1],
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.25)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='80% Prediction Interval'
    ))
    
    # 3. Recent Historical Series
    fig.add_trace(go.Scatter(
        x=recent_hist['date'],
        y=recent_hist[target_col],
        mode='lines',
        line=dict(color='#0ea5e9', width=2.5),
        name='Historical Observed'
    ))
    
    # 4. Multi-Horizon Forecast Line
    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=fc_points,
        mode='lines+markers',
        line=dict(color='#f59e0b', width=3, dash='dash'),
        marker=dict(size=8, color='#f59e0b', symbol='diamond'),
        name='Point Forecast (7d, 14d, 30d)'
    ))
    
    # 5. Capacity Threshold Line (optional)
    if capacity_threshold is not None and not np.isnan(capacity_threshold) and capacity_threshold > 0:
        min_date = recent_hist['date'].min() if not recent_hist.empty else last_date
        fig.add_trace(go.Scatter(
            x=[min_date, fc_dates[-1]],
            y=[capacity_threshold, capacity_threshold],
            mode='lines',
            line=dict(color='#ef4444', width=2, dash='dot'),
            name=f'Capacity Threshold ({capacity_threshold:,.0f} beds)'
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{target_name} - Multi-Horizon Forecast & Capacity Bounds</b>",
            font=dict(size=16, color='#f8fafc')
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor='#334155'),
        yaxis=dict(title="Active Beds / Children", showgrid=True, gridcolor='#334155'),
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        paper_bgcolor='rgba(15, 23, 42, 0.0)',
        font=dict(color='#cbd5e1'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(15, 23, 42, 0.8)'
        ),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    return fig

def plot_breach_gauge(breach_prob_pct: float, threshold_val: float) -> go.Figure:
    """Creates needle/gauge meter displaying Capacity Breach Probability."""
    safe_pct = max(0.0, min(100.0, float(breach_prob_pct))) if breach_prob_pct is not None and not np.isnan(breach_prob_pct) else 0.0
    safe_thresh = float(threshold_val) if threshold_val is not None and not np.isnan(threshold_val) else 3000.0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=safe_pct,
        number=dict(suffix="%", font=dict(size=28, color='#f8fafc')),
        title=dict(text=f"<b>Capacity Breach Risk (C={safe_thresh:,.0f})</b>", font=dict(size=14, color='#94a3b8')),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94a3b8"),
            bar=dict(color="#f8fafc", thickness=0.3, line=dict(color="#f8fafc", width=1)),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=2,
            bordercolor="#334155",
            steps=[
                dict(range=[0, 10], color="#22c55e"),
                dict(range=[10, 30], color="#eab308"),
                dict(range=[30, 60], color="#f97316"),
                dict(range=[60, 100], color="#ef4444")
            ]
        )
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        margin=dict(l=30, r=30, t=50, b=20),
        height=260
    )
    return fig

def plot_net_flow_plotly(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    days_back: int = 90
) -> go.Figure:
    """
    Generates interactive Plotly visualization for Net Flow Imbalance (Transfers - Discharges),
    displaying recent historical net flow, forward multi-horizon predictions with 80% and 95%
    confidence intervals, and the zero equilibrium boundary.
    """
    if historical_df.empty or forecast_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No net flow forecast data available")
        return fig

    recent_hist = historical_df.dropna(subset=['date']).tail(days_back).copy()
    last_date = recent_hist['date'].max()
    last_val = float(recent_hist['net_flow'].iloc[-1])
    
    fc_dates = [last_date]
    fc_points = [last_val]
    fc_l80 = [last_val]
    fc_u80 = [last_val]
    fc_l95 = [last_val]
    fc_u95 = [last_val]
    
    for _, row in forecast_df.sort_values('horizon_days').iterrows():
        h = int(row['horizon_days'])
        fc_dates.append(last_date + pd.Timedelta(days=h))
        fc_points.append(float(row['point_forecast']))
        fc_l80.append(float(row.get('lower_80', row['point_forecast'])))
        fc_u80.append(float(row.get('upper_80', row['point_forecast'])))
        fc_l95.append(float(row.get('lower_95', row['point_forecast'])))
        fc_u95.append(float(row.get('upper_95', row['point_forecast'])))
        
    fig = go.Figure()
    
    # 95% Confidence Band
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1],
        y=fc_u95 + fc_l95[::-1],
        fill='toself',
        fillcolor='rgba(147, 51, 234, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name='95% Prediction Interval'
    ))
    
    # 80% Confidence Band
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1],
        y=fc_u80 + fc_l80[::-1],
        fill='toself',
        fillcolor='rgba(147, 51, 234, 0.25)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name='80% Prediction Interval'
    ))
    
    # Recent Historical Net Flow
    fig.add_trace(go.Scatter(
        x=recent_hist['date'],
        y=recent_hist['net_flow'],
        mode='lines',
        line=dict(color='#a855f7', width=2),
        name='Historical Net Flow'
    ))
    
    # Forecast Points
    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=fc_points,
        mode='lines+markers',
        line=dict(color='#ec4899', width=3, dash='dash'),
        marker=dict(size=8, color='#ec4899', symbol='diamond'),
        name='Predicted Net Flow (7d, 14d, 30d)'
    ))
    
    # Net Zero Equilibrium Line
    min_date = recent_hist['date'].min() if not recent_hist.empty else last_date
    fig.add_trace(go.Scatter(
        x=[min_date, fc_dates[-1]],
        y=[0.0, 0.0],
        mode='lines',
        line=dict(color='#94a3b8', width=1.5, dash='dash'),
        name='Equilibrium (Net Zero Flow)'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Daily Net Flow Imbalance & Multi-Horizon Projections (Transfers − Discharges)</b>",
            font=dict(size=16, color='#f8fafc')
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor='#334155'),
        yaxis=dict(title="Net Daily Flow (Children / Day)", showgrid=True, gridcolor='#334155'),
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        paper_bgcolor='rgba(15, 23, 42, 0.0)',
        font=dict(color='#cbd5e1'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(15, 23, 42, 0.8)'
        ),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    return fig
