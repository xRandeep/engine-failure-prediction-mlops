import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
from datetime import datetime

# --- CONFIGURATION ---
HF_USERNAME = os.getenv("HF_USERNAME", "iStillWaters")
MODEL_REPO_NAME = os.getenv("MODEL_REPO_NAME", "auto_predictive_maintenance_model")
MODEL_REPO_ID = f"{HF_USERNAME}/{MODEL_REPO_NAME}"

MODEL_FILENAME = "best_engine_model.pkl"
SCALER_FILENAME = "scaler.joblib"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Engine Health Monitor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_artifacts():
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        scaler_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=SCALER_FILENAME)
        return joblib.load(model_path), joblib.load(scaler_path), None
    except Exception as e:
        return None, None, str(e)

# --- INITIALIZE SESSION STATE ---
if 'rpm' not in st.session_state:
    st.session_state.rpm = 750
if 'fuel_p' not in st.session_state:
    st.session_state.fuel_p = 6.2
if 'oil_p' not in st.session_state:
    st.session_state.oil_p = 3.16
if 'coolant_temp' not in st.session_state:
    st.session_state.coolant_temp = 80.0
if 'coolant_p' not in st.session_state:
    st.session_state.coolant_p = 2.16
if 'oil_temp' not in st.session_state:
    st.session_state.oil_temp = 80.0

# Callback functions to sync values
def update_rpm_from_num():
    st.session_state.rpm = st.session_state.rpm_num

def update_rpm_from_sld():
    st.session_state.rpm = st.session_state.rpm_sld

def update_fuel_from_num():
    st.session_state.fuel_p = st.session_state.fuel_num

def update_fuel_from_sld():
    st.session_state.fuel_p = st.session_state.fuel_sld

def update_oil_p_from_num():
    st.session_state.oil_p = st.session_state.oil_p_num

def update_oil_p_from_sld():
    st.session_state.oil_p = st.session_state.oil_p_sld

def update_coolant_t_from_num():
    st.session_state.coolant_temp = st.session_state.coolant_t_num

def update_coolant_t_from_sld():
    st.session_state.coolant_temp = st.session_state.coolant_t_sld

def update_coolant_p_from_num():
    st.session_state.coolant_p = st.session_state.coolant_p_num

def update_coolant_p_from_sld():
    st.session_state.coolant_p = st.session_state.coolant_p_sld

def update_oil_t_from_num():
    st.session_state.oil_temp = st.session_state.oil_t_num

def update_oil_t_from_sld():
    st.session_state.oil_temp = st.session_state.oil_t_sld

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 100% !important;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    hr {display: none !important;}
    
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        gap: 0.2rem !important;
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #00d4ff, #0090ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.8rem;
        text-align: center;
        color: #8b95a5;
        margin-bottom: 0.5rem;
    }
    
    .section-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #00d4ff;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    .param-card {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.6), rgba(10, 14, 39, 0.8));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 6px;
        padding: 0.5rem;
        margin-bottom: 0.4rem;
    }
    
    .param-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.3rem;
    }
    
    .param-icon {
        font-size: 1.1rem;
        margin-right: 0.4rem;
    }
    
    .param-name {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        color: #8b95a5;
        text-transform: uppercase;
    }
    
    .param-value-display {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        text-align: center;
        margin: 0.2rem 0;
    }
    
    .stNumberInput input {
        height: 2rem !important;
        font-size: 0.8rem !important;
        padding: 0.2rem 0.5rem !important;
    }
    
    [data-testid="column"] {
        padding: 0.2rem !important;
    }
    
    .engine-status-container {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.6), rgba(10, 14, 39, 0.8));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .engine-icon-display {
        font-size: 5rem;
        margin: 0.3rem 0;
        filter: drop-shadow(0 0 15px currentColor);
    }
    
    .probability-text {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.625rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.0625rem;
        font-weight: 700;
        border-radius: 15px;
        border: 2px solid;
        text-transform: uppercase;
        margin: 0.3rem 0;
    }
    
    .alert-box {
        padding: 0.4rem 0.6rem;
        border-radius: 4px;
        margin: 0.2rem 0;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9375rem;
        border-left: 3px solid;
    }
    
    .alert-warning {
        background: rgba(255, 170, 0, 0.1);
        border-color: #ffaa00;
        color: #ffaa00;
    }
    
    .alert-critical {
        background: rgba(255, 51, 102, 0.1);
        border-color: #ff3366;
        color: #ff3366;
    }
    
    .rec-item {
        padding: 0.4rem 0.6rem;
        margin: 0.2rem 0;
        background: rgba(0, 212, 255, 0.05);
        border-left: 2px solid #00d4ff;
        border-radius: 3px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9375rem;
        color: #e0e6ed;
    }
    
    .feature-bar {
        height: 24px;
        background: linear-gradient(90deg, var(--color), transparent);
        border-radius: 3px;
        margin: 0.2rem 0;
        padding: 0.1rem 0.4rem;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.875rem;
        color: white;
    }
    
    .detail-box {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.4), rgba(10, 14, 39, 0.6));
        border-left: 2px solid #00d4ff;
        border-radius: 4px;
        padding: 0.5rem;
        margin: 0.3rem 0;
    }
    
    .detail-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.0625rem;
        font-weight: 700;
        color: #00d4ff;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
    }
    
    .detail-content {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9375rem;
        color: #b8c5d6;
        line-height: 1.4;
    }
    
    .stButton > button {
        padding: 0.5rem 1rem !important;
        font-size: 0.9rem !important;
        margin: 0.3rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def create_gauge(value, max_value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 11, 'color': color}},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.5},
            'bgcolor': "rgba(26, 31, 58, 0.3)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, max_value * 0.6], 'color': 'rgba(0, 255, 136, 0.1)'},
                {'range': [max_value * 0.6, max_value * 0.8], 'color': 'rgba(255, 170, 0, 0.1)'},
                {'range': [max_value * 0.8, max_value], 'color': 'rgba(255, 51, 102, 0.1)'}
            ]
        }
    ))
    
    fig.update_layout(
        height=70,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Rajdhani'}
    )
    
    return fig

def validate_parameter(name, value, thresholds):
    if value > thresholds.get('critical_high', float('inf')):
        return 'critical', f"🔴 {name}: Critically high ({value})"
    elif value < thresholds.get('critical_low', float('-inf')):
        return 'critical', f"🔴 {name}: Critically low ({value})"
    elif value > thresholds.get('warning_high', float('inf')):
        return 'warning', f"⚠️ {name}: High ({value})"
    elif value < thresholds.get('warning_low', float('-inf')):
        return 'warning', f"⚠️ {name}: Low ({value})"
    return 'normal', None

def get_status_info(probability):
    if probability < 0.25:
        return 'healthy', '🟢', 'HEALTHY', '#00ff88'
    elif probability < 0.50:
        return 'caution', '🟡', 'CAUTION', '#ffaa00'
    elif probability < 0.75:
        return 'warning', '🟠', 'WARNING', '#ff9500'
    else:
        return 'critical', '🔴', 'CRITICAL', '#ff3366'

def calculate_feature_importance(params):
    return {
        'Engine RPM': params['rpm'] / 2500,
        'Coolant Temperature': params['coolant_temp'] / 200,
        'Oil Pressure (Risk)': max(0, 1 - params['oil_pressure'] / 10),
        'Fuel Pressure': params['fuel_pressure'] / 25
    }

def get_recommendations(params, probability, warnings, criticals):
    recs = []
    
    if criticals:
        recs.append("🚨 IMMEDIATE ACTION: Critical parameters detected")
        recs.extend(criticals[:2])
    
    if params['coolant_temp'] > 100:
        recs.append("🔧 Check cooling system - radiator and thermostat")
    
    if params['oil_pressure'] < 2.0:
        recs.append("🔧 Inspect oil pump and filter")
    
    if params['fuel_pressure'] < 5.0:
        recs.append("🔧 Examine fuel system components")
    
    if params['rpm'] > 2000:
        recs.append("⚙️ Reduce engine load immediately")
    
    if probability > 0.75:
        recs.append("📅 Emergency maintenance required")
    elif probability > 0.5:
        recs.append("📅 Schedule maintenance within 24 hours")
    elif probability > 0.25:
        recs.append("📋 Monitor closely")
    else:
        recs.append("✅ Continue normal operations")
    
    return recs[:6] if recs else ["✅ All systems normal"]

# --- MAIN APP ---
def main():
    model, scaler, error = load_artifacts()
    
    if model is None:
        st.error(f"⚠️ Model Loading Error: {error}")
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-title">⚙️ Engine Health Monitor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Predictive Maintenance System</p>', unsafe_allow_html=True)
    
    # --- HORIZONTAL LAYOUT: LEFT (INPUTS) | RIGHT (ANALYSIS) ---
    left_col, right_col = st.columns([1, 1.5])
    
    # ========== LEFT COLUMN: PARAMETER INPUTS (2x3 GRID) ==========
    with left_col:
        st.markdown('<div class="section-title">📊 Engine Parameters</div>', unsafe_allow_html=True)
        
        # Parameter thresholds
        thresholds = {
            'rpm': {'warning_high': 2000, 'critical_high': 2300},
            'fuel_pressure': {'warning_low': 5.0, 'critical_low': 4.0},
            'oil_pressure': {'warning_low': 2.0, 'critical_low': 1.5},
            'coolant_temp': {'warning_high': 100, 'critical_high': 120},
            'coolant_pressure': {'warning_low': 1.5, 'critical_low': 1.0},
            'oil_temp': {'warning_high': 110, 'critical_high': 130}
        }
        
        # ROW 1: RPM and FUEL
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown('<div class="param-card">', unsafe_allow_html=True)
            st.markdown('<div class="param-header"><span class="param-icon">⚡</span><span class="param-name">Engine RPM</span></div>', unsafe_allow_html=True)
            
            input_col1, input_col2 = st.columns([1, 3])
            with input_col1:
                st.number_input("", 0, 2500, value=st.session_state.rpm, step=50, key="rpm_num", label_visibility="collapsed", on_change=update_rpm_from_num)
            with input_col2:
                st.slider("", 0, 2500, value=st.session_state.rpm, step=50, key="rpm_sld", label_visibility="collapsed", on_change=update_rpm_from_sld)
            
            st.markdown(f'<div class="param-value-display" style="color: #00d4ff;">{st.session_state.rpm}</div>', unsafe_allow_html=True)
            st.plotly_chart(create_gauge(st.session_state.rpm, 2500, "#00d4ff"), use_container_width=True, config={'displayModeBar': False}, key="g_rpm")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with row1_col2:
            st.markdown('<div class="param-card">', unsafe_allow_html=True)
            st.markdown('<div class="param-header"><span class="param-icon">⛽</span><span class="param-name">Fuel Pressure (Bar)</span></div>', unsafe_allow_html=True)
            
            input_col1, input_col2 = st.columns([1, 3])
            with input_col1:
                st.number_input("", 0.0, 25.0, value=st.session_state.fuel_p, step=0.1, key="fuel_num", label_visibility="collapsed", on_change=update_fuel_from_num)
            with input_col2:
                st.slider("", 0.0, 25.0, value=st.session_state.fuel_p, step=0.1, key="fuel_sld", label_visibility="collapsed", on_change=update_fuel_from_sld)
            
            st.markdown(f'<div class="param-value-display" style="color: #ff6b35;">{st.session_state.fuel_p:.1f}</div>', unsafe_allow_html=True)
            st.plotly_chart(create_gauge(st.session_state.fuel_p, 25, "#ff6b35"), use_container_width=True, config={'displayModeBar': False}, key="g_fuel")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ROW 2: OIL PRESSURE and COOLANT TEMP
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.markdown('<div class="param-card">', unsafe_allow_html=True)
            st.markdown('<div class="param-header"><span class="param-icon">🛢️</span><span class="param-name">Oil Pressure (Bar)</span></div>', unsafe_allow_html=True)
            
            input_col1, input_col2 = st.columns([1, 3])
            with input_col1:
                st.number_input("", 0.0, 10.0, value=st.session_state.oil_p, step=0.1, key="oil_p_num", label_visibility="collapsed", on_change=update_oil_p_from_num)
            with input_col2:
                st.slider("", 0.0, 10.0, value=st.session_state.oil_p, step=0.1, key="oil_p_sld", label_visibility="collapsed", on_change=update_oil_p_from_sld)
            
            st.markdown(f'<div class="param-value-display" style="color: #ffaa00;">{st.session_state.oil_p:.2f}</div>', unsafe_allow_html=True)
            st.plotly_chart(create_gauge(st.session_state.oil_p, 10, "#ffaa00"), use_container_width=True, config={'displayModeBar': False}, key="g_oil_p")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with row2_col2:
            st.markdown('<div class="param-card">', unsafe_allow_html=True)
            st.markdown('<div class="param-header"><span class="param-icon">🌡️</span><span class="param-name">Coolant Temp (°C)</span></div>', unsafe_allow_html=True)
            
            input_col1, input_col2 = st.columns([1, 3])
            with input_col1:
                st.number_input("", 0.0, 200.0, value=st.session_state.coolant_temp, step=1.0, key="coolant_t_num", label_visibility="collapsed", on_change=update_coolant_t_from_num)
            with input_col2:
                st.slider("", 0.0, 200.0, value=st.session_state.coolant_temp, step=1.0, key="coolant_t_sld", label_visibility="collapsed", on_change=update_coolant_t_from_sld)
            
            st.markdown(f'<div class="param-value-display" style="color: #ff3366;">{st.session_state.coolant_temp:.1f}</div>', unsafe_allow_html=True)
            st.plotly_chart(create_gauge(st.session_state.coolant_temp, 200, "#ff3366"), use_container_width=True, config={'displayModeBar': False}, key="g_coolant_t")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ROW 3: COOLANT PRESSURE and OIL TEMP
        row3_col1, row3_col2 = st.columns(2)
        
        with row3_col1:
            st.markdown('<div class="param-card">', unsafe_allow_html=True)
            st.markdown('<div class="param-header"><span class="param-icon">💧</span><span class="param-name">Coolant Pressure (Bar)</span></div>', unsafe_allow_html=True)
            
            input_col1, input_col2 = st.columns([1, 3])
            with input_col1:
                st.number_input("", 0.0, 10.0, value=st.session_state.coolant_p, step=0.1, key="coolant_p_num", label_visibility="collapsed", on_change=update_coolant_p_from_num)
            with input_col2:
                st.slider("", 0.0, 10.0, value=st.session_state.coolant_p, step=0.1, key="coolant_p_sld", label_visibility="collapsed", on_change=update_coolant_p_from_sld)
            
            st.markdown(f'<div class="param-value-display" style="color: #00ff88;">{st.session_state.coolant_p:.2f}</div>', unsafe_allow_html=True)
            st.plotly_chart(create_gauge(st.session_state.coolant_p, 10, "#00ff88"), use_container_width=True, config={'displayModeBar': False}, key="g_coolant_p")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with row3_col2:
            st.markdown('<div class="param-card">', unsafe_allow_html=True)
            st.markdown('<div class="param-header"><span class="param-icon">🔥</span><span class="param-name">Oil Temp (°C)</span></div>', unsafe_allow_html=True)
            
            input_col1, input_col2 = st.columns([1, 3])
            with input_col1:
                st.number_input("", 0.0, 150.0, value=st.session_state.oil_temp, step=1.0, key="oil_t_num", label_visibility="collapsed", on_change=update_oil_t_from_num)
            with input_col2:
                st.slider("", 0.0, 150.0, value=st.session_state.oil_temp, step=1.0, key="oil_t_sld", label_visibility="collapsed", on_change=update_oil_t_from_sld)
            
            st.markdown(f'<div class="param-value-display" style="color: #a855f7;">{st.session_state.oil_temp:.1f}</div>', unsafe_allow_html=True)
            st.plotly_chart(create_gauge(st.session_state.oil_temp, 150, "#a855f7"), use_container_width=True, config={'displayModeBar': False}, key="g_oil_t")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== RIGHT COLUMN: ANALYSIS ==========
    with right_col:
        st.markdown('<div class="section-title">🔍 Analysis & Results</div>', unsafe_allow_html=True)
        
        # Real-time validation
        warnings = []
        criticals = []
        
        for param_name, value, threshold_key in [
            ('RPM', st.session_state.rpm, 'rpm'),
            ('Fuel Pressure', st.session_state.fuel_p, 'fuel_pressure'),
            ('Oil Pressure', st.session_state.oil_p, 'oil_pressure'),
            ('Coolant Temp', st.session_state.coolant_temp, 'coolant_temp'),
            ('Coolant Pressure', st.session_state.coolant_p, 'coolant_pressure'),
            ('Oil Temp', st.session_state.oil_temp, 'oil_temp')
        ]:
            status, msg = validate_parameter(param_name, value, thresholds.get(threshold_key, {}))
            if status == 'critical' and msg:
                criticals.append(msg)
            elif status == 'warning' and msg:
                warnings.append(msg)
        
        # Display alerts
        if criticals or warnings:
            if criticals:
                for alert in criticals:
                    st.markdown(f'<div class="alert-box alert-critical">{alert}</div>', unsafe_allow_html=True)
            
            if warnings:
                for alert in warnings:
                    st.markdown(f'<div class="alert-box alert-warning">{alert}</div>', unsafe_allow_html=True)
        
        # Analyze button
        if st.button("🔍 ANALYZE ENGINE STATUS", type="primary", use_container_width=True):
            input_df = pd.DataFrame({
                'Engine rpm': [st.session_state.rpm],
                'Lub oil pressure': [st.session_state.oil_p],
                'Fuel pressure': [st.session_state.fuel_p],
                'Coolant pressure': [st.session_state.coolant_p],
                'lub oil temp': [st.session_state.oil_temp],
                'Coolant temp': [st.session_state.coolant_temp]
            })
            
            try:
                scaled = scaler.transform(input_df)
                pred = model.predict(scaled)[0]
                prob = model.predict_proba(scaled)[0][1] if hasattr(model, 'predict_proba') else 0.0
                
                status, emoji, status_text, color = get_status_info(prob)
                
                # Engine Status
                st.markdown(f"""
                <div class="engine-status-container">
                    <div class="engine-icon-display" style="color: {color};">{emoji}</div>
                    <div class="probability-text" style="color: {color};">{prob*100:.1f}% Failure Risk</div>
                    <div class="status-badge" style="border-color: {color}; color: {color};">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature Importance
                st.markdown('<div class="section-title">⚡ Risk Factors</div>', unsafe_allow_html=True)
                
                params_dict = {
                    'rpm': st.session_state.rpm,
                    'coolant_temp': st.session_state.coolant_temp,
                    'oil_pressure': st.session_state.oil_p,
                    'fuel_pressure': st.session_state.fuel_p
                }
                
                importance = calculate_feature_importance(params_dict)
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                for feature, score in sorted_features:
                    if score > 0.3:
                        bar_color = '#ff3366' if score > 0.7 else '#ffaa00' if score > 0.5 else '#00d4ff'
                        st.markdown(f"""
                        <div class="feature-bar" style="--color: {bar_color}; width: {score*100}%;">
                            <strong>{feature}:</strong> {score*100:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown('<div class="section-title">📋 Recommendations</div>', unsafe_allow_html=True)
                
                recommendations = get_recommendations(params_dict, prob, warnings, criticals)
                
                for rec in recommendations:
                    st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)
                
                # Analysis Details
                st.markdown('<div class="section-title">📊 Details</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="detail-box">
                    <div class="detail-title">Prediction Summary</div>
                    <div class="detail-content">
                        <strong>Probability:</strong> {prob*100:.2f}% | 
                        <strong>Classification:</strong> {'FAILURE RISK' if pred == 1 else 'OPERATIONAL'} | 
                        <strong>Status:</strong> {status_text}<br>
                        <strong>Time:</strong> {datetime.now().strftime('%H:%M:%S')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="detail-box">
                    <div class="detail-title">Parameter Check</div>
                    <div class="detail-content">
                        <strong>RPM:</strong> {st.session_state.rpm} {'⚠️' if st.session_state.rpm > 2000 else '✓'} | 
                        <strong>Coolant:</strong> {st.session_state.coolant_temp:.1f}°C {'⚠️' if st.session_state.coolant_temp > 100 else '✓'}<br>
                        <strong>Oil P:</strong> {st.session_state.oil_p:.2f} Bar {'⚠️' if st.session_state.oil_p < 2.0 else '✓'} | 
                        <strong>Fuel P:</strong> {st.session_state.fuel_p:.1f} Bar {'⚠️' if st.session_state.fuel_p < 5.0 else '✓'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
        
        else:
            st.markdown("""
            <div class="engine-status-container">
                <div class="engine-icon-display" style="color: #4a5568;">⚙️</div>
                <div class="probability-text" style="color: #8b95a5;">Awaiting Analysis</div>
                <p style="font-family: 'Rajdhani', sans-serif; color: #8b95a5; font-size: 1.0rem; margin: 0.5rem 0;">
                    Configure parameters and click "ANALYZE"
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
