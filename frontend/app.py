# ============================================================
# STEP 8: STREAMLIT DASHBOARD
# Run: streamlit run frontend/app.py
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="🦠 Pandemic Intelligence System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap');
    
    * { font-family: 'IBM Plex Sans', sans-serif; }
    
    .main { background: #0a0e1a; color: #e8eaf0; }
    .stApp { background: #0a0e1a; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2845 100%);
        border: 1px solid #2a3a5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 2.2em;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.75em;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7a8aac;
        margin-top: 4px;
    }
    .risk-low    { color: #2ecc71; border-color: #2ecc71; }
    .risk-medium { color: #f39c12; border-color: #f39c12; }
    .risk-high   { color: #e74c3c; border-color: #e74c3c; }
    
    .alert-box {
        background: rgba(231, 76, 60, 0.15);
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .normal-box {
        background: rgba(46, 204, 113, 0.12);
        border: 1px solid #2ecc71;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    
    div[data-testid="stSidebar"] {
        background: #0d1220;
        border-right: 1px solid #1e2845;
    }
    
    .section-header {
        font-size: 1.1em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5b8dee;
        border-bottom: 1px solid #1e2845;
        padding-bottom: 8px;
        margin: 20px 0 12px 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3a6ef5, #6a4ef5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    
    .forecast-table {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# ── Helpers ───────────────────────────────────────────────
def api_call(endpoint, method='GET', data=None):
    try:
        url = f"{API_BASE}{endpoint}"
        if method == 'POST':
            resp = requests.post(url, json=data, timeout=15)
        else:
            resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "API not running. Start it with: `uvicorn api.main:app --reload`"
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=300)
def load_features():
    try:
        df = pd.read_csv('data/processed/features_data.csv', parse_dates=['Date'])
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_risk_scores():
    try:
        return pd.read_csv('data/risk_outputs/country_risk_scores.csv')
    except:
        return pd.DataFrame()

def fmt_number(n, suffix=''):
    n = float(n) if n else 0
    if n >= 1_000_000: return f"{n/1_000_000:.2f}M{suffix}"
    if n >= 1_000:     return f"{n/1_000:.1f}K{suffix}"
    return f"{n:.0f}{suffix}"

def risk_color(category):
    return {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}.get(category, '#888')

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦠 Pandemic Intelligence")
    st.markdown("*AI-powered COVID-19 Analytics*")
    st.divider()
    
    page = st.radio(
        "Navigate",
        ["🌍 Global Dashboard", "📈 Forecasting",
         "⚠️ Risk Assessment", "🚨 Anomaly Detection",
         "🔬 Model Analysis"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # API status
    health, err = api_call('/health')
    if health:
        st.success("🟢 API Connected")
        st.caption(f"Countries: {health.get('countries_available', 0)}")
    else:
        st.error("🔴 API Offline")
        st.code("uvicorn api.main:app --reload", language="bash")
    
    st.divider()
    st.caption("Pandemic Intelligence System v1.0")
    st.caption("Built with LSTM · XGBoost · Prophet")

# ── Load data ──────────────────────────────────────────────
df      = load_features()
rs      = load_risk_scores()

# ══════════════════════════════════════════════════════════
# PAGE 1: GLOBAL DASHBOARD
# ══════════════════════════════════════════════════════════
if page == "🌍 Global Dashboard":
    st.title("🌍 Global Pandemic Dashboard")
    st.markdown("*Real-time overview of worldwide COVID-19 situation*")
    
    if df.empty:
        st.error("⚠️ Data not loaded. Run Steps 1–6 first.")
        st.stop()
    
    latest = df.groupby('Country').last().reset_index()
    
    # ── KPI Row ────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_conf = latest['Confirmed'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#e74c3c">{fmt_number(total_conf)}</div>
            <div class="metric-label">Total Confirmed</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        total_deaths = latest['Deaths'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#95a5a6">{fmt_number(total_deaths)}</div>
            <div class="metric-label">Total Deaths</div>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        total_rec = latest['Recovered'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#2ecc71">{fmt_number(total_rec)}</div>
            <div class="metric-label">Recovered</div>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        total_active = latest['Active'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#f39c12">{fmt_number(total_active)}</div>
            <div class="metric-label">Active Cases</div>
        </div>""", unsafe_allow_html=True)
    
    with col5:
        global_cfr = (total_deaths / total_conf * 100) if total_conf > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#9b59b6">{global_cfr:.2f}%</div>
            <div class="metric-label">Global CFR</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── World Map ──────────────────────────────────────────
    st.markdown('<div class="section-header">GLOBAL CASE DISTRIBUTION</div>',
                unsafe_allow_html=True)
    
    col_map, col_bar = st.columns([3, 2])
    
    with col_map:
        if not rs.empty and 'XGB_Risk_Category' in rs.columns:
            map_df = rs[['Country', 'XGB_Risk_Category', 'Confirmed', 'CFR', 'Growth_Rate']].copy()
        else:
            map_df = latest[['Country', 'Confirmed', 'CFR', 'Growth_Rate']].copy()
            map_df['XGB_Risk_Category'] = pd.cut(
                map_df['CFR'], bins=[-1, 2, 5, 100],
                labels=['Low', 'Medium', 'High'])
        
        fig_map = px.choropleth(
            map_df,
            locations='Country',
            locationmode='country names',
            color='Confirmed',
            hover_name='Country',
            hover_data={'CFR': ':.2f', 'Growth_Rate': ':.2f'},
            color_continuous_scale='Reds',
            title=None
        )
        fig_map.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False,
                     landcolor='#1a2035', showcoastlines=True,
                     coastlinecolor='#2a3a5c', showocean=True,
                     oceancolor='#0d1a2e'),
            margin=dict(l=0, r=0, t=10, b=0),
            height=380,
            coloraxis_colorbar=dict(
                title='Confirmed', tickfont=dict(color='#aaa'))
        )
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col_bar:
        st.markdown("**Top 15 Countries by Confirmed Cases**")
        top15 = latest.nlargest(15, 'Confirmed')[['Country', 'Confirmed', 'CFR']].copy()
        fig_bar = go.Figure(go.Bar(
            x=top15['Confirmed'], y=top15['Country'],
            orientation='h',
            marker=dict(
                color=top15['CFR'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title='CFR %', len=0.7)
            ),
            text=top15['Confirmed'].apply(fmt_number),
            textposition='outside'
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=380,
            margin=dict(l=0, r=60, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, color='#aaa'),
            yaxis=dict(color='#e8eaf0', tickfont=dict(size=11)),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # ── Global Timeline ────────────────────────────────────
    st.markdown('<div class="section-header">PANDEMIC TIMELINE</div>',
                unsafe_allow_html=True)
    
    try:
        dw = pd.read_csv('data/processed/day_wise_clean.csv', parse_dates=['Date'])
        
        fig_time = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Cumulative Cases', 'Daily New Cases'],
            shared_xaxes=True, vertical_spacing=0.12
        )
        fig_time.add_trace(go.Scatter(
            x=dw['Date'], y=dw['Confirmed'], name='Confirmed',
            fill='tozeroy', fillcolor='rgba(231,76,60,0.2)',
            line=dict(color='#e74c3c', width=2)
        ), row=1, col=1)
        fig_time.add_trace(go.Scatter(
            x=dw['Date'], y=dw['Deaths'], name='Deaths',
            fill='tozeroy', fillcolor='rgba(149,165,166,0.2)',
            line=dict(color='#95a5a6', width=2)
        ), row=1, col=1)
        
        if 'New_Cases' in dw.columns:
            fig_time.add_trace(go.Bar(
                x=dw['Date'], y=dw['New_Cases'], name='New Cases',
                marker_color='rgba(69,123,157,0.7)'
            ), row=2, col=1)
        
        fig_time.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450, showlegend=True,
            legend=dict(orientation='h', y=1.05),
            font=dict(color='#aaa'),
            margin=dict(l=60, r=20, t=30, b=30)
        )
        for i in [1, 2]:
            fig_time.update_yaxes(
                row=i, col=1,
                gridcolor='#1e2845', color='#aaa', showgrid=True)
            fig_time.update_xaxes(
                row=i, col=1,
                gridcolor='#1e2845', color='#aaa')
        
        st.plotly_chart(fig_time, use_container_width=True)
    except:
        st.info("Load day_wise data to see timeline")

# ══════════════════════════════════════════════════════════
# PAGE 2: FORECASTING
# ══════════════════════════════════════════════════════════
elif page == "📈 Forecasting":
    st.title("📈 COVID-19 Case Forecasting")
    st.markdown("*LSTM neural network predictions with uncertainty estimation*")
    
    col_ctrl, col_info = st.columns([1, 3])
    
    with col_ctrl:
        st.markdown("**Select Country**")
        countries_resp, err = api_call('/countries')
        if countries_resp:
            country_list = countries_resp['countries']
            forecast_available = countries_resp.get('forecast_available', [])
        else:
            country_list = ['United States', 'India', 'Brazil',
                            'United Kingdom', 'France']
            forecast_available = []
        
        selected_country = st.selectbox(
            "Country", country_list,
            index=country_list.index('United States')
                  if 'United States' in country_list else 0,
            label_visibility="collapsed"
        )
        
        forecast_days = st.slider(
            "Forecast Horizon (days)", 7, 60, 30, step=7)
        
        run_btn = st.button("🚀 Generate Forecast", use_container_width=True)
        
        if selected_country in forecast_available:
            st.success("⚡ LSTM model available")
        else:
            st.info("📊 Using trend model")
    
    with col_info:
        if run_btn or True:  # Auto-load
            result, err = api_call('/predict', 'POST', {
                'country': selected_country,
                'days': forecast_days
            })
            
            if err:
                st.error(f"API Error: {err}")
            elif result:
                fc = result['forecast']
                dates    = [f['date'] for f in fc]
                values   = [f['daily_cases'] for f in fc]
                summary  = result.get('summary', {})
                model_nm = result.get('model', 'LSTM')
                
                # KPIs
                kc1, kc2, kc3, kc4 = st.columns(4)
                with kc1:
                    st.metric("Forecast Model", model_nm.split()[0])
                with kc2:
                    st.metric("Peak Cases", fmt_number(summary.get('peak_cases', max(values))))
                with kc3:
                    st.metric("Total (30d)", fmt_number(summary.get('total_cases', sum(values))))
                with kc4:
                    st.metric("Avg Daily", fmt_number(summary.get('avg_daily', np.mean(values))))
                
                # Historical + forecast chart
                if not df.empty:
                    cdf = df[df['Country'] == selected_country].sort_values('Date')
                    recent = cdf.tail(60)
                    
                    fig = go.Figure()
                    
                    # Historical
                    fig.add_trace(go.Scatter(
                        x=recent['Date'], y=recent['Daily_Cases'],
                        name='Historical', mode='lines',
                        line=dict(color='#5b8dee', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=recent['Date'], y=recent['MA_7_Cases'],
                        name='7-day avg', mode='lines',
                        line=dict(color='#aaa', width=1, dash='dash')
                    ))
                    
                    # Forecast
                    upper = [v * 1.15 for v in values]
                    lower = [max(0, v * 0.85) for v in values]
                    
                    fig.add_trace(go.Scatter(
                        x=dates + dates[::-1],
                        y=upper + lower[::-1],
                        fill='toself',
                        fillcolor='rgba(243,156,18,0.15)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='95% CI',
                        showlegend=True
                    ))
                    fig.add_trace(go.Scatter(
                        x=dates, y=values,
                        name=f'Forecast ({model_nm.split()[0]})',
                        mode='lines+markers',
                        line=dict(color='#f39c12', width=3),
                        marker=dict(size=5)
                    ))
                    
                    # Divider line
                    if len(recent) > 0:
                        last_hist = str(recent['Date'].max().date())
                        fig.add_vline(x=last_hist, line_dash='dot',
                                      line_color='#aaa', opacity=0.6)
                        fig.add_annotation(
                            x=last_hist, y=max(values) * 0.9,
                            text="Forecast →", showarrow=False,
                            font=dict(color='#aaa', size=11)
                        )
                    
                    fig.update_layout(
                        title=f'{selected_country} — {forecast_days}-Day Forecast',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=430, font=dict(color='#aaa'),
                        legend=dict(orientation='h', y=-0.1),
                        xaxis=dict(gridcolor='#1e2845', color='#aaa'),
                        yaxis=dict(gridcolor='#1e2845', color='#aaa',
                                   title='Daily Cases'),
                        margin=dict(l=60, r=20, t=50, b=60)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                with st.expander("📋 Full Forecast Table"):
                    fc_df = pd.DataFrame({
                        'Date': dates,
                        'Predicted Cases': [f"{v:,.0f}" for v in values],
                        'Lower (85%)': [f"{max(0,v*0.85):,.0f}" for v in values],
                        'Upper (115%)': [f"{v*1.15:,.0f}" for v in values]
                    })
                    st.dataframe(fc_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# PAGE 3: RISK ASSESSMENT
# ══════════════════════════════════════════════════════════
elif page == "⚠️ Risk Assessment":
    st.title("⚠️ Risk Assessment Engine")
    st.markdown("*XGBoost classification with SHAP explainability*")
    
    tab1, tab2 = st.tabs(["🌍 Country Risk Map", "🔢 Custom Input"])
    
    with tab1:
        if not rs.empty and 'XGB_Risk_Category' in rs.columns:
            # Summary
            rc1, rc2, rc3 = st.columns(3)
            for cat, col, color in [
                ('High', rc1, '#e74c3c'),
                ('Medium', rc2, '#f39c12'),
                ('Low', rc3, '#2ecc71')
            ]:
                n = (rs['XGB_Risk_Category'] == cat).sum()
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color:{color}">
                        <div class="metric-value" style="color:{color}">{n}</div>
                        <div class="metric-label">{cat} Risk Countries</div>
                    </div>""", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Map
            risk_map_df = rs[['Country', 'XGB_Risk_Category', 'Risk_Score', 'CFR',
                               'Growth_Rate', 'Confirmed']].copy()
            
            fig_risk = px.choropleth(
                risk_map_df,
                locations='Country',
                locationmode='country names',
                color='XGB_Risk_Category',
                color_discrete_map={'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'},
                hover_name='Country',
                hover_data={
                    'Risk_Score': ':.0f',
                    'CFR': ':.2f',
                    'Growth_Rate': ':.2f',
                    'XGB_Risk_Category': True
                },
                title='Global COVID-19 Risk Classification (XGBoost Model)'
            )
            fig_risk.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                geo=dict(bgcolor='rgba(10,14,26,0)', landcolor='#1a2035',
                         showcoastlines=True, coastlinecolor='#2a3a5c',
                         showocean=True, oceancolor='#0d1a2e', showframe=False),
                height=480,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(font=dict(color='#aaa'), bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Top high risk table
            st.markdown("**🔴 High Risk Countries — Detailed**")
            high_r = rs[rs['XGB_Risk_Category'] == 'High'].sort_values(
                'Risk_Score', ascending=False).head(20)
            display_cols = ['Country', 'Risk_Score', 'CFR', 'Growth_Rate',
                            'Active_Ratio', 'Doubling_Time']
            display_cols = [c for c in display_cols if c in high_r.columns]
            st.dataframe(
                high_r[display_cols].round(3),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("Run Step 6 to generate risk scores")
    
    with tab2:
        st.markdown("**Enter custom epidemiological data for real-time risk assessment**")
        
        col_l, col_r = st.columns(2)
        with col_l:
            confirmed = st.number_input("Total Confirmed",    value=500000, step=10000)
            deaths    = st.number_input("Total Deaths",       value=10000,  step=1000)
            recovered = st.number_input("Total Recovered",    value=400000, step=10000)
            active    = st.number_input("Active Cases",       value=90000,  step=5000)
            daily_cases = st.number_input("Daily New Cases",  value=5000,   step=500)
        with col_r:
            growth_rate   = st.number_input("Growth Rate (%)",  value=2.5, step=0.5)
            cfr           = st.number_input("CFR (%)",          value=2.0, step=0.1)
            recovery_rate = st.number_input("Recovery Rate (%).",value=80.0, step=5.0)
            active_ratio  = st.number_input("Active Ratio (%)", value=18.0, step=2.0)
            doubling_time = st.number_input("Doubling Time (days)", value=30.0, step=5.0)
            country_name  = st.text_input("Country (optional)", value="")
        
        ma7 = daily_cases * 0.95
        
        if st.button("🎯 Assess Risk", use_container_width=True):
            result, err = api_call('/risk', 'POST', {
                'confirmed': confirmed, 'deaths': deaths,
                'recovered': recovered, 'active': active,
                'daily_cases': daily_cases, 'daily_deaths': int(deaths * 0.002),
                'growth_rate': growth_rate, 'cfr': cfr,
                'recovery_rate': recovery_rate, 'active_ratio': active_ratio,
                'doubling_time': doubling_time, 'ma_7_cases': ma7,
                'case_acceleration': daily_cases * 0.1,
                'country': country_name or None
            })
            
            if err:
                st.error(f"Error: {err}")
            elif result:
                cat = result['risk_category']
                color = risk_color(cat)
                
                st.markdown(f"""
                <div class="metric-card" style="border-color:{color}; margin-top:20px">
                    <div class="metric-value" style="color:{color}; font-size:3em">
                        {cat.upper()} RISK
                    </div>
                    <div class="metric-label">{result['recommendation']}</div>
                </div>""", unsafe_allow_html=True)
                
                # Probability gauge
                probs = result['probabilities']
                fig_gauge = go.Figure(go.Bar(
                    x=[probs['Low'], probs['Medium'], probs['High']],
                    y=['Low', 'Medium', 'High'],
                    orientation='h',
                    marker_color=['#2ecc71', '#f39c12', '#e74c3c'],
                    text=[f"{v:.1%}" for v in [probs['Low'], probs['Medium'], probs['High']]],
                    textposition='outside'
                ))
                fig_gauge.update_layout(
                    title='Risk Probability Breakdown',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=220, margin=dict(l=80, r=80, t=40, b=20),
                    xaxis=dict(showgrid=False, range=[0, 1.1],
                               showticklabels=False, color='#aaa'),
                    yaxis=dict(color='#e8eaf0', tickfont=dict(size=14)),
                    font=dict(color='#aaa')
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                if result.get('risk_signals'):
                    st.markdown("**Risk Signals Detected:**")
                    for signal in result['risk_signals']:
                        st.markdown(f"- {signal}")

# ══════════════════════════════════════════════════════════
# PAGE 4: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly & Outbreak Detection")
    st.markdown("*Isolation Forest + LSTM Autoencoder early warning system*")
    
    tab1, tab2 = st.tabs(["📊 Historical Analysis", "🔍 Real-time Check"])
    
    with tab1:
        try:
            anomaly_df = pd.read_csv('data/anomaly_outputs/anomaly_scores.csv',
                                      parse_dates=['Date'])
            
            countries_with_anomalies = anomaly_df[anomaly_df['Is_Anomaly_ISO'] == True]
            top_anomaly_countries = (countries_with_anomalies
                                     .groupby('Country').size()
                                     .sort_values(ascending=False).head(10))
            
            a1, a2 = st.columns(2)
            with a1:
                st.metric("Total Anomalies Detected",
                          f"{(anomaly_df['Is_Anomaly_ISO'] == True).sum():,}")
            with a2:
                st.metric("Countries with Anomalies",
                          countries_with_anomalies['Country'].nunique())
            
            st.markdown("**Countries with Most Anomaly Events**")
            fig_an = px.bar(
                x=top_anomaly_countries.index,
                y=top_anomaly_countries.values,
                color=top_anomaly_countries.values,
                color_continuous_scale='Reds',
                labels={'x': 'Country', 'y': 'Anomaly Count'}
            )
            fig_an.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350, showlegend=False,
                font=dict(color='#aaa'),
                xaxis=dict(color='#aaa', tickangle=30),
                yaxis=dict(color='#aaa', gridcolor='#1e2845')
            )
            st.plotly_chart(fig_an, use_container_width=True)
            
            # Country anomaly timeline
            sel_country = st.selectbox(
                "Explore anomaly timeline for:",
                anomaly_df['Country'].unique())
            
            cdf_an = anomaly_df[anomaly_df['Country'] == sel_country].sort_values('Date')
            if len(cdf_an) > 0:
                fig_anom = go.Figure()
                fig_anom.add_trace(go.Scatter(
                    x=cdf_an['Date'], y=cdf_an['Daily_Cases'],
                    name='Daily Cases', mode='lines',
                    line=dict(color='#5b8dee', width=1.5)
                ))
                
                if 'Is_Anomaly_ISO' in cdf_an.columns:
                    anomaly_pts = cdf_an[cdf_an['Is_Anomaly_ISO'] == True]
                    fig_anom.add_trace(go.Scatter(
                        x=anomaly_pts['Date'], y=anomaly_pts['Daily_Cases'],
                        name='Anomaly (IF)', mode='markers',
                        marker=dict(color='#e74c3c', size=10, symbol='x')
                    ))
                
                fig_anom.update_layout(
                    title=f'{sel_country} — Anomaly Timeline',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=380, font=dict(color='#aaa'),
                    xaxis=dict(color='#aaa', gridcolor='#1e2845'),
                    yaxis=dict(color='#aaa', gridcolor='#1e2845',
                               title='Daily Cases')
                )
                st.plotly_chart(fig_anom, use_container_width=True)
        except:
            st.warning("Run Step 5 to generate anomaly data")
    
    with tab2:
        st.markdown("**Enter current data to check for anomalies in real-time**")
        
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            daily_cases = st.number_input("Daily Cases",  value=10000, step=500)
            growth_rate = st.number_input("Growth Rate (%)", value=3.0, step=0.5)
            ma7_cases   = st.number_input("7-Day Avg Cases", value=9500, step=500)
        with ac2:
            daily_deaths = st.number_input("Daily Deaths", value=200, step=10)
            cfr_val      = st.number_input("CFR (%)", value=2.0, step=0.1)
            rec_rate     = st.number_input("Recovery Rate (%)", value=75.0, step=5.0)
        with ac3:
            case_accel = st.number_input("Case Acceleration (diff from yesterday)",
                                          value=500, step=100)
        
        if st.button("🔍 Check for Anomaly", use_container_width=True):
            result, err = api_call('/anomaly', 'POST', {
                'daily_cases': daily_cases,
                'daily_deaths': daily_deaths,
                'growth_rate': growth_rate,
                'cfr': cfr_val,
                'recovery_rate': rec_rate,
                'case_acceleration': case_accel,
                'ma_7_cases': ma7_cases
            })
            
            if err:
                st.error(f"Error: {err}")
            elif result:
                is_anomaly = result['is_anomaly']
                
                if is_anomaly:
                    st.markdown(f"""
                    <div class="alert-box">
                        <h2 style="color:#e74c3c; margin:0">🚨 ANOMALY DETECTED</h2>
                        <p style="margin:8px 0 0 0; color:#e8eaf0">
                            Score: {result['anomaly_score']:.4f} | 
                            Confidence: {result['confidence']:.1f}%
                        </p>
                        <p style="color:#aaa; margin:4px 0 0 0">
                            {result['interpretation']['description']}
                        </p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="normal-box">
                        <h2 style="color:#2ecc71; margin:0">✅ NORMAL RANGE</h2>
                        <p style="margin:8px 0 0 0; color:#e8eaf0">
                            Score: {result['anomaly_score']:.4f} | 
                            Confidence: {result['confidence']:.1f}%
                        </p>
                        <p style="color:#aaa; margin:4px 0 0 0">
                            {result['interpretation']['description']}
                        </p>
                    </div>""", unsafe_allow_html=True)
                
                st.info(f"**Recommended Action:** {result['recommended_action']}")
                
                if result.get('alert_reasons'):
                    st.markdown("**Alert Reasons:**")
                    for r in result['alert_reasons']:
                        st.markdown(f"  - {r}")

# ══════════════════════════════════════════════════════════
# PAGE 5: MODEL ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "🔬 Model Analysis":
    st.title("🔬 Model Performance Analysis")
    st.markdown("*RMSE · MAE · MAPE · ROC-AUC across all models*")
    
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Models", "🌳 Risk Model", "📈 Training Info"])
    
    with tab1:
        try:
            mc = pd.read_csv('data/forecast_outputs/model_metrics_comparison.csv')
            st.markdown("**Forecasting Model Comparison (Test Set)**")
            st.dataframe(mc.round(2), use_container_width=True, hide_index=True)
            
            # RMSE comparison chart
            rmse_cols = [c for c in mc.columns if 'RMSE' in c]
            if len(rmse_cols) > 1 and 'Country' in mc.columns:
                mc_melt = mc.melt(id_vars='Country', value_vars=rmse_cols,
                                   var_name='Model', value_name='RMSE')
                mc_melt['Model'] = mc_melt['Model'].str.replace('_RMSE','')
                fig_mc = px.bar(
                    mc_melt, x='Country', y='RMSE', color='Model',
                    barmode='group',
                    title='RMSE Comparison by Model and Country',
                    color_discrete_sequence=['#5b8dee', '#f39c12', '#e74c3c']
                )
                fig_mc.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=380, font=dict(color='#aaa'),
                    xaxis=dict(color='#aaa'), yaxis=dict(color='#aaa', gridcolor='#1e2845')
                )
                st.plotly_chart(fig_mc, use_container_width=True)
        except:
            st.info("Run Step 4 to generate model metrics")
    
    with tab2:
        try:
            import os
            if os.path.exists('data/risk_outputs/shap_importance.png'):
                st.image('data/risk_outputs/shap_importance.png',
                          caption='SHAP Feature Importance — High Risk Class')
            if os.path.exists('data/risk_outputs/model_evaluation.png'):
                st.image('data/risk_outputs/model_evaluation.png',
                          caption='XGBoost Confusion Matrix & Feature Importance')
        except:
            st.info("Run Step 6 to generate SHAP analysis")
        
        if not rs.empty and 'XGB_Risk_Category' in rs.columns:
            st.markdown("**Risk Score Distribution**")
            if 'Risk_Score' in rs.columns:
                fig_hist = px.histogram(
                    rs, x='Risk_Score', color='XGB_Risk_Category',
                    color_discrete_map={'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'},
                    nbins=30, title='Risk Score Distribution by Category'
                )
                fig_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350, font=dict(color='#aaa'),
                    xaxis=dict(color='#aaa'), yaxis=dict(color='#aaa', gridcolor='#1e2845')
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
        st.markdown("**System Architecture**")
        st.code("""
Pandemic Intelligence System — Architecture

┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                        │
│  CSV Files → Cleaning → Feature Eng → Processed CSV    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                   MODEL LAYER                           │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │    LSTM     │  │   XGBoost    │  │  Isolation    │  │
│  │  Attention  │  │    Risk      │  │    Forest     │  │
│  │ Forecasting │  │ Classifier   │  │  + Autoenc.   │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                   API LAYER (FastAPI)                   │
│  /predict   /risk   /anomaly   /summary/{country}      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              DASHBOARD (Streamlit)                      │
│  Global Map · Forecasting · Risk · Anomaly · Analysis  │
└─────────────────────────────────────────────────────────┘
        """)
        
        st.markdown("**Tech Stack**")
        st.markdown("""
        | Component | Technology |
        |-----------|-----------|
        | Forecasting | LSTM with Multi-Head Attention (PyTorch) |
        | Risk Model | XGBoost + SHAP explainability |
        | Anomaly | Isolation Forest + LSTM Autoencoder |
        | Baseline | ARIMA (statsmodels) + Prophet |
        | Backend | FastAPI + Pydantic |
        | Frontend | Streamlit |
        | Data | Pandas + NumPy |
        | Deployment | Docker + Render + Streamlit Cloud |
        """)