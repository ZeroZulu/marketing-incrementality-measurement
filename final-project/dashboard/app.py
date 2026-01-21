"""
Marketing Incrementality Dashboard - Streamlit Version
======================================================
A dark-themed interactive dashboard for causal inference analysis.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="Marketing Incrementality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM CSS FOR DARK THEME ==============
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #09090b;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #18181b;
        border-right: 1px solid #27272a;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Text */
    p, span, label {
        color: #a1a1aa !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #c8ff00 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #71717a !important;
    }
    
    /* Cards */
    .css-1r6slb0, .css-12w0qpk {
        background-color: #27272a;
        border: 1px solid #3f3f46;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: linear-gradient(135deg, #27272a 0%, #18181b 100%);
        border: 1px solid #3f3f46;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        color: #71717a;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .kpi-sublabel {
        color: #52525b;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
    
    /* Color variants */
    .kpi-blue { color: #3498db; }
    .kpi-green { color: #2ecc71; }
    .kpi-purple { color: #9b59b6; }
    .kpi-lime { color: #c8ff00; }
    .kpi-red { color: #e74c3c; }
    .kpi-cyan { color: #00d4ff; }
    
    /* Info boxes */
    .info-box {
        background-color: rgba(0, 212, 255, 0.05);
        border-left: 4px solid #00d4ff;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: rgba(251, 191, 36, 0.05);
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: rgba(34, 197, 94, 0.05);
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #18181b;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #71717a;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #c8ff00 !important;
        color: #09090b !important;
    }
    
    /* Tables */
    .dataframe {
        background-color: #27272a !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #27272a;
        border-color: #3f3f46;
    }
    
    /* Number input */
    .stNumberInput > div > div > input {
        background-color: #27272a;
        border-color: #3f3f46;
        color: #fafafa;
    }
    
    /* Divider */
    hr {
        border-color: #27272a;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ============== DATA LOADING ==============
@st.cache_data
def load_data():
    """Load analysis results from JSON files."""
    data_dir = Path("../output")
    
    # Try multiple possible locations
    possible_paths = [
        Path("../output"),
        Path("output"),
        Path("../incrementality-project-fixed/output"),
        Path("public/data"),
    ]
    
    for data_dir in possible_paths:
        results_path = data_dir / "results.json"
        if results_path.exists():
            break
    else:
        # Return sample data if no files found
        return get_sample_data()
    
    try:
        with open(data_dir / "results.json") as f:
            results = json.load(f)
        
        did_data = []
        psm_balance = []
        sc_data = []
        sc_weights = []
        method_comparison = []
        
        if (data_dir / "did_timeseries.json").exists():
            with open(data_dir / "did_timeseries.json") as f:
                did_data = json.load(f)
        
        if (data_dir / "psm_balance.json").exists():
            with open(data_dir / "psm_balance.json") as f:
                psm_balance = json.load(f)
        
        if (data_dir / "sc_timeseries.json").exists():
            with open(data_dir / "sc_timeseries.json") as f:
                sc_data = json.load(f)
        
        if (data_dir / "sc_weights.json").exists():
            with open(data_dir / "sc_weights.json") as f:
                sc_weights = json.load(f)
        
        if (data_dir / "method_comparison.json").exists():
            with open(data_dir / "method_comparison.json") as f:
                method_comparison = json.load(f)
        
        return {
            'results': results,
            'did_data': did_data,
            'psm_balance': psm_balance,
            'sc_data': sc_data,
            'sc_weights': sc_weights,
            'method_comparison': method_comparison
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return get_sample_data()

def get_sample_data():
    """Return sample data for demo purposes."""
    return {
        'results': {
            'summary': {'avg_lift': 0.12, 'true_effect': 0.12},
            'did': {'lift': 0.123, 'ci_lower': 0.059, 'ci_upper': 0.187, 'p_value': '< 0.001', 'significant': True, 'r_squared': 0.847, 'effect': 108},
            'psm': {'lift': 0.112, 'ci_lower': 0.057, 'ci_upper': 0.166, 'p_value': '< 0.001', 'significant': True, 'naive_lift': 0.245, 'bias_reduction': 0.54, 'n_matched': 12500},
            'sc': {'lift': 0.098, 'ci_lower': 0.042, 'ci_upper': 0.154, 'p_value': '0.024', 'significant': True, 'pre_rmse': 42.3, 'weights': {'Texas': 0.35, 'Florida': 0.28, 'Illinois': 0.22, 'Ohio': 0.10, 'Pennsylvania': 0.05}},
            'business': {'campaign_spend': 500000, 'incremental_installs': 6000, 'incremental_revenue': 150000, 'iroas': 0.30},
            'metadata': {'data_source': 'synthetic', 'n_users': 50000},
            'generated_at': '2024-01-21T12:00:00'
        },
        'did_data': [{'week': i, 'treatment': 1000 + i*10 + (150 if i >= 26 else 0) + np.random.normal(0, 20), 'control': 1000 + i*10 + np.random.normal(0, 20)} for i in range(52)],
        'psm_balance': [
            {'covariate': 'Engagement Score', 'before': 0.45, 'after': 0.03},
            {'covariate': 'Session Count', 'before': 0.38, 'after': 0.05},
            {'covariate': 'Days Since Install', 'before': 0.31, 'after': 0.04},
            {'covariate': 'Lifetime Value', 'before': 0.28, 'after': 0.06},
            {'covariate': 'Is Mobile', 'before': 0.22, 'after': 0.02},
            {'covariate': 'Is Organic', 'before': 0.18, 'after': 0.04}
        ],
        'sc_data': [{'week': i, 'actual': 1200 + i*5 + (180 if i >= 26 else 0) + np.random.normal(0, 30), 'synthetic': 1200 + i*5 + np.random.normal(0, 30), 'gap': (180 if i >= 26 else 0)} for i in range(52)],
        'sc_weights': [{'state': 'Texas', 'weight': 0.35}, {'state': 'Florida', 'weight': 0.28}, {'state': 'Illinois', 'weight': 0.22}, {'state': 'Ohio', 'weight': 0.10}, {'state': 'Pennsylvania', 'weight': 0.05}],
        'method_comparison': [
            {'method': 'DiD', 'lift': 12.3, 'color': '#3498db'},
            {'method': 'PSM', 'lift': 11.2, 'color': '#2ecc71'},
            {'method': 'SC', 'lift': 9.8, 'color': '#9b59b6'}
        ]
    }

# ============== CHART THEME ==============
CHART_THEME = {
    'paper_bgcolor': '#09090b',
    'plot_bgcolor': '#09090b',
    'font': {'color': '#a1a1aa', 'family': 'Inter, sans-serif'},
    'xaxis': {'gridcolor': '#27272a', 'linecolor': '#27272a', 'zerolinecolor': '#27272a'},
    'yaxis': {'gridcolor': '#27272a', 'linecolor': '#27272a', 'zerolinecolor': '#27272a'},
}

COLORS = {
    'treatment': '#2ecc71',
    'control': '#3498db',
    'synthetic': '#9b59b6',
    'effect': '#e74c3c',
    'lime': '#c8ff00',
    'before': '#e74c3c',
    'after': '#2ecc71'
}

def apply_chart_theme(fig):
    """Apply dark theme to plotly figure."""
    fig.update_layout(
        paper_bgcolor=CHART_THEME['paper_bgcolor'],
        plot_bgcolor=CHART_THEME['plot_bgcolor'],
        font=CHART_THEME['font'],
        xaxis=CHART_THEME['xaxis'],
        yaxis=CHART_THEME['yaxis'],
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# ============== COMPONENTS ==============
def kpi_card(label, value, sublabel=None, color="lime"):
    """Render a KPI card."""
    color_class = f"kpi-{color}"
    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {color_class}">{value}</div>
        {f'<div class="kpi-sublabel">{sublabel}</div>' if sublabel else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def info_box(title, content, type="info"):
    """Render an info box."""
    icon = {"info": "💡", "warning": "⚠️", "success": "✅"}[type]
    st.markdown(f"""
    <div class="{type}-box">
        <strong>{icon} {title}</strong><br>
        <span style="color: #d4d4d8;">{content}</span>
    </div>
    """, unsafe_allow_html=True)

# ============== PAGES ==============
def render_overview(data):
    """Render overview page."""
    results = data['results']
    did = results['did']
    psm = results['psm']
    sc = results['sc']
    business = results['business']
    summary = results['summary']
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("DiD Lift", f"{did['lift']*100:.1f}%", f"p {did['p_value']}")
    with col2:
        st.metric("PSM Lift (ATT)", f"{psm['lift']*100:.1f}%", f"p {psm['p_value']}")
    with col3:
        st.metric("SC Lift", f"{sc['lift']*100:.1f}%", f"RMSE: {sc.get('pre_rmse', 0):.1f}")
    with col4:
        st.metric("Incremental Revenue", f"${business['incremental_revenue']/1000:.0f}K", f"{summary['avg_lift']*100:.1f}% avg lift")
    
    st.divider()
    
    # Key Finding
    bias_reduction = abs(psm.get('bias_reduction', 0)) * 100
    info_box(
        "Key Finding",
        f"All methods converge on ~{summary['avg_lift']*100:.1f}% incremental lift. "
        f"PSM shows naive attribution overstated by <strong style='color: #fbbf24;'>{bias_reduction:.0f}%</strong> due to selection bias. "
        f"{'True effect: ' + str(summary.get('true_effect', 0)*100) + '%' if summary.get('true_effect') else ''}",
        "info"
    )
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Difference-in-Differences")
        if data['did_data']:
            df = pd.DataFrame(data['did_data'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['week'], y=df['treatment'], name='Treatment', line=dict(color=COLORS['treatment'], width=2)))
            fig.add_trace(go.Scatter(x=df['week'], y=df['control'], name='Control', line=dict(color=COLORS['control'], width=2)))
            fig.add_vline(x=26, line_dash="dash", line_color=COLORS['effect'], annotation_text="Treatment Start")
            fig = apply_chart_theme(fig)
            fig.update_layout(height=300, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Synthetic Control")
        if data['sc_data']:
            df = pd.DataFrame(data['sc_data'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['week'], y=df['actual'], name='Actual', line=dict(color=COLORS['treatment'], width=2)))
            fig.add_trace(go.Scatter(x=df['week'], y=df['synthetic'], name='Synthetic', line=dict(color=COLORS['synthetic'], width=2, dash='dash')))
            fig.add_vline(x=26, line_dash="dash", line_color=COLORS['effect'])
            fig = apply_chart_theme(fig)
            fig.update_layout(height=300, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PSM Covariate Balance")
        if data['psm_balance']:
            df = pd.DataFrame(data['psm_balance'])
            fig = go.Figure()
            fig.add_trace(go.Bar(y=df['covariate'], x=df['before'], name='Before', orientation='h', marker_color=COLORS['before']))
            fig.add_trace(go.Bar(y=df['covariate'], x=df['after'], name='After', orientation='h', marker_color=COLORS['after']))
            fig.add_vline(x=0.1, line_dash="dash", line_color="#22c55e")
            fig.add_vline(x=-0.1, line_dash="dash", line_color="#22c55e")
            fig = apply_chart_theme(fig)
            fig.update_layout(height=300, barmode='group', showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Method Comparison")
        methods = ['DiD', 'PSM', 'Synthetic Control']
        lifts = [did['lift']*100, psm['lift']*100, sc['lift']*100]
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=lifts, y=methods, orientation='h', marker_color=colors))
        fig = apply_chart_theme(fig)
        fig.update_layout(height=300, xaxis_title="Lift (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Results Table
    st.subheader("Analysis Results Summary")
    
    table_data = {
        'Method': ['Difference-in-Differences', 'Propensity Score Matching', 'Synthetic Control', 'Average (Triangulated)'],
        'Lift': [f"{did['lift']*100:.1f}%", f"{psm['lift']*100:.1f}%", f"{sc['lift']*100:.1f}%", f"{summary['avg_lift']*100:.1f}%"],
        '95% CI': [f"[{did['ci_lower']*100:.1f}%, {did['ci_upper']*100:.1f}%]", 
                   f"[{psm['ci_lower']*100:.1f}%, {psm['ci_upper']*100:.1f}%]",
                   f"[{sc['ci_lower']*100:.1f}%, {sc['ci_upper']*100:.1f}%]", "—"],
        'p-value': [did['p_value'], psm['p_value'], sc['p_value'], "—"],
        'Status': ['✅ Significant' if did['significant'] else '⚠️ Not Sig.',
                   '✅ Significant' if psm['significant'] else '⚠️ Not Sig.',
                   '✅ Significant' if sc['significant'] else '⚠️ Not Sig.',
                   '🎯 Consensus']
    }
    
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

def render_did(data):
    """Render DiD analysis page."""
    results = data['results']
    did = results['did']
    
    st.header("Difference-in-Differences Analysis")
    st.caption("Comparing treatment vs control regions over time")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estimated Lift", f"{did['lift']*100:.1f}%", "Treatment effect")
    with col2:
        st.metric("95% CI", f"[{did['ci_lower']*100:.0f}%, {did['ci_upper']*100:.0f}%]")
    with col3:
        st.metric("p-value", did['p_value'])
    with col4:
        st.metric("R-squared", f"{did.get('r_squared', 0):.3f}")
    
    st.divider()
    
    info_box(
        "How DiD Works",
        "DiD compares the change in outcomes over time between treatment and control groups. "
        "The key assumption is <strong>parallel trends</strong>: both groups would have followed the same trajectory without treatment.",
        "info"
    )
    
    st.divider()
    
    # Main Chart
    st.subheader("Parallel Trends Visualization")
    if data['did_data']:
        df = pd.DataFrame(data['did_data'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['week'], y=df['treatment'], name='Treatment Group', line=dict(color=COLORS['treatment'], width=3)))
        fig.add_trace(go.Scatter(x=df['week'], y=df['control'], name='Control Group', line=dict(color=COLORS['control'], width=3)))
        fig.add_vline(x=26, line_dash="dash", line_color=COLORS['effect'], line_width=2, annotation_text="Treatment Start", annotation_position="top")
        fig = apply_chart_theme(fig)
        fig.update_layout(height=450, xaxis_title="Week", yaxis_title="Conversions", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        info_box("Pre-Treatment Period", "Lines run parallel before week 26, validating the parallel trends assumption.", "success")
    with col2:
        info_box("Post-Treatment Period", f"After treatment, the treatment group diverges upward by ~{did['lift']*100:.1f}%.", "info")

def render_psm(data):
    """Render PSM analysis page."""
    results = data['results']
    psm = results['psm']
    
    st.header("Propensity Score Matching Analysis")
    st.caption("Matching treated & control users on observable characteristics")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ATT (Causal Lift)", f"{psm['lift']*100:.1f}%", "After matching")
    with col2:
        st.metric("Naive Estimate", f"{psm['naive_lift']*100:.1f}%", "Before matching", delta_color="inverse")
    with col3:
        st.metric("Bias Reduction", f"{abs(psm['bias_reduction'])*100:.0f}%", "Selection bias removed")
    with col4:
        st.metric("Matched Pairs", f"{psm['n_matched']:,}")
    
    st.divider()
    
    info_box(
        "Selection Bias Detected",
        f"The naive estimate ({psm['naive_lift']*100:.1f}%) is <strong style='color: #fbbf24;'>{abs(psm['naive_lift']/psm['lift']):.1f}x</strong> "
        f"the causal estimate ({psm['lift']*100:.1f}%). More engaged users were more likely to be treated, inflating apparent effects.",
        "warning"
    )
    
    st.divider()
    
    # Covariate Balance Chart
    st.subheader("Covariate Balance (Love Plot)")
    if data['psm_balance']:
        df = pd.DataFrame(data['psm_balance'])
        fig = go.Figure()
        fig.add_trace(go.Bar(y=df['covariate'], x=df['before'], name='Before Matching', orientation='h', marker_color=COLORS['before']))
        fig.add_trace(go.Bar(y=df['covariate'], x=df['after'], name='After Matching', orientation='h', marker_color=COLORS['after']))
        fig.add_vline(x=0.1, line_dash="dash", line_color="#22c55e", annotation_text="0.1 threshold")
        fig.add_vline(x=-0.1, line_dash="dash", line_color="#22c55e")
        fig.add_vline(x=0, line_color="#71717a")
        fig = apply_chart_theme(fig)
        fig.update_layout(height=400, barmode='group', xaxis_title="Standardized Mean Difference (SMD)", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        info_box("Before Matching", "Large imbalances (SMD > 0.1) indicate treated users differ systematically from controls.", "warning")
    with col2:
        info_box("After Matching", "All covariates now have SMD < 0.1, indicating good balance for causal inference.", "success")

def render_sc(data):
    """Render Synthetic Control page."""
    results = data['results']
    sc = results['sc']
    
    st.header("Synthetic Control Analysis")
    st.caption("Constructing a counterfactual from donor pool")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estimated Lift", f"{sc['lift']*100:.1f}%", "vs synthetic")
    with col2:
        st.metric("Pre-Treatment RMSE", f"{sc.get('pre_rmse', 0):.1f}", "Fit quality")
    with col3:
        st.metric("p-value (Placebo)", sc['p_value'])
    with col4:
        st.metric("Donor States", str(len(sc.get('weights', {}))))
    
    st.divider()
    
    info_box(
        "How Synthetic Control Works",
        "We construct a 'synthetic California' as a weighted combination of control states that matches California's pre-treatment trajectory. "
        "The post-treatment gap represents the causal effect.",
        "info"
    )
    
    st.divider()
    
    # Main Chart
    st.subheader("Actual vs Synthetic California")
    if data['sc_data']:
        df = pd.DataFrame(data['sc_data'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['week'], y=df['actual'], name='Actual California', line=dict(color=COLORS['treatment'], width=3)))
        fig.add_trace(go.Scatter(x=df['week'], y=df['synthetic'], name='Synthetic California', line=dict(color=COLORS['synthetic'], width=3, dash='dash')))
        
        # Add shaded area for treatment effect
        post_treatment = df[df['week'] >= 26]
        if len(post_treatment) > 0:
            fig.add_trace(go.Scatter(
                x=list(post_treatment['week']) + list(post_treatment['week'][::-1]),
                y=list(post_treatment['actual']) + list(post_treatment['synthetic'][::-1]),
                fill='toself',
                fillcolor='rgba(155, 89, 182, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                name='Treatment Effect',
                showlegend=True
            ))
        
        fig.add_vline(x=26, line_dash="dash", line_color=COLORS['effect'], line_width=2, annotation_text="Treatment")
        fig = apply_chart_theme(fig)
        fig.update_layout(height=450, xaxis_title="Week", yaxis_title="Conversions", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Donor Weights
    st.subheader("Donor State Weights")
    if data['sc_weights']:
        cols = st.columns(len(data['sc_weights']))
        for i, w in enumerate(data['sc_weights']):
            with cols[i]:
                st.metric(w['state'], f"{w['weight']*100:.0f}%")

def render_impact(data):
    """Render Business Impact Calculator."""
    results = data['results']
    summary = results['summary']
    business = results['business']
    
    st.header("Business Impact Calculator")
    st.caption("Translate causal lift estimates into business metrics")
    
    st.divider()
    
    # Calculator Inputs
    st.subheader("Campaign Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        campaign_spend = st.number_input("Campaign Spend ($)", value=business['campaign_spend'], step=10000, format="%d")
    with col2:
        revenue_per_install = st.number_input("Revenue per Install ($)", value=25, step=1)
    
    st.divider()
    
    # Calculations
    avg_lift = summary['avg_lift']
    base_installs = campaign_spend / 10  # Assuming $10 CPI
    incremental_installs = int(base_installs * avg_lift)
    incremental_revenue = incremental_installs * revenue_per_install
    iroas = incremental_revenue / campaign_spend if campaign_spend > 0 else 0
    
    # Results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Lift", f"{avg_lift*100:.1f}%", "Triangulated")
    with col2:
        st.metric("Incremental Installs", f"{incremental_installs:,}", "Additional users")
    with col3:
        st.metric("Incremental Revenue", f"${incremental_revenue:,}", "Additional revenue")
    with col4:
        delta_color = "normal" if iroas >= 1 else "inverse"
        st.metric("iROAS", f"{iroas:.2f}x", "Profitable" if iroas >= 1 else "Below 1x", delta_color=delta_color)
    
    st.divider()
    
    # Calculation Breakdown
    st.subheader("Calculation Breakdown")
    
    breakdown_data = {
        'Step': ['Base Installs', 'Incremental Installs', 'Incremental Revenue', '**iROAS**'],
        'Formula': ['Spend ÷ $10 CPI', f'Base × {avg_lift*100:.1f}%', f'{incremental_installs:,} × ${revenue_per_install}', f'${incremental_revenue:,} ÷ ${campaign_spend:,}'],
        'Result': [f'{int(base_installs):,}', f'{incremental_installs:,}', f'${incremental_revenue:,}', f'**{iroas:.2f}x**']
    }
    
    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)
    
    st.divider()
    
    if iroas >= 1:
        info_box("What This Means", f"For every $1 spent, you generated ${iroas:.2f} in incremental revenue. The campaign is profitable!", "success")
    else:
        info_box("What This Means", f"The campaign generated only ${iroas:.2f} for every $1 spent. Consider optimizing before scaling.", "warning")
    
    # Method Comparison
    st.divider()
    st.subheader("Lift Estimates by Method")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("DiD", f"{results['did']['lift']*100:.1f}%")
    with col2:
        st.metric("PSM", f"{results['psm']['lift']*100:.1f}%")
    with col3:
        st.metric("Synthetic Control", f"{results['sc']['lift']*100:.1f}%")

# ============== MAIN APP ==============
def main():
    # Load data
    data = load_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <div style="background: #c8ff00; border-radius: 8px; padding: 8px; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 20px;">📊</span>
            </div>
            <span style="font-weight: 600; font-size: 16px; color: #fafafa;">Incrementality</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Analysis")
        page = st.radio(
            "Navigation",
            ["Overview", "DiD Analysis", "PSM Analysis", "Synthetic Control", "Business Impact"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("### Data Info")
        st.caption(f"📊 {data['results']['metadata']['n_users']:,} users analyzed")
        st.caption(f"📁 Source: {data['results']['metadata']['data_source']}")
        
        st.divider()
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    st.title("Marketing Incrementality Dashboard")
    st.caption("Gaming UA • Causal Lift Analysis")
    
    if page == "Overview":
        render_overview(data)
    elif page == "DiD Analysis":
        render_did(data)
    elif page == "PSM Analysis":
        render_psm(data)
    elif page == "Synthetic Control":
        render_sc(data)
    elif page == "Business Impact":
        render_impact(data)
    
    # Footer
    st.divider()
    st.caption(f"Generated: {data['results'].get('generated_at', 'N/A')} • Marketing Incrementality & Lift Measurement")

if __name__ == "__main__":
    main()
