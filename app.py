import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Page Configuration & High-Contrast Dracula Dark Theme Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    :root {
        --background: #1e1e2e;
        --current-line: #2d2f3f;
        --foreground: #ffffff;
        --comment: #9a9aae;
        --cyan: #8be9fd;
        --green: #50fa7b;
        --orange: #ffb86c;
        --pink: #ff79c6;
        --purple: #bd93f9;
        --yellow: #f1fa8c;
    }

    .stApp {
        background-color: var(--background);
        color: var(--foreground);
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 98% !important;
    }

    /* Card Scaffolding - High Contrast Borders */
    .dracula-card {
        background-color: var(--current-line);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border: 2px solid var(--purple);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .metric-title {
        color: var(--cyan) !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .metric-value {
        color: var(--green);
        font-size: 26px;
        font-weight: bold;
    }

    .stat-pill {
        display: inline-block;
        background-color: var(--purple);
        color: #1e1e2e;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 6px;
    }

    /* High-contrast Label Styling */
    label, .stSelectbox label, .stNumberInput label {
        color: #ff79c6 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }

    /* Streamlit Input Component Box Borders */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div {
        border: 1.5px solid var(--cyan) !important;
        border-radius: 6px !important;
        background-color: #1e1e2e !important;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: var(--green) !important;
        color: #1e1e2e !important;
        font-weight: 800 !important;
        font-size: 15px !important;
        border-radius: 6px !important;
        border: 2px solid var(--green) !important;
        padding: 8px 0px !important;
        margin-top: 10px;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--cyan) !important;
        border-color: var(--cyan) !important;
        transform: scale(1.01);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model & Setup Fallback
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open('House-Price-Prediction-RFR.pkl', 'rb') as f:
            return pickle.load(f), True
    except Exception:
        return None, False

model, model_loaded = load_model()

if model_loaded and hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    feature_names = [
        'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_FT', 'READY_TO_MOVE',
        'RESALE', 'LONGITUDE', 'LATITUDE', 'BHK_OR_RK_BHK', 'BHK_OR_RK_RK',
        'ADDRESS_panvel,Mumbai', 'ADDRESS_Manoramaganj,Indore', 'ADDRESS_100 Feet Road,Anand'
    ]

# Binary Mapping Helper: Plain Yes / No
BINARY_MAP = {"No": 0, "Yes": 1}

address_features = [f for f in feature_names if f.startswith("ADDRESS_")]
locations = [f.replace("ADDRESS_", "") for f in address_features]
if not locations:
    locations = ["panvel,Mumbai", "Manoramaganj,Indore", "100 Feet Road,Anand"]

if 'prediction_val' not in st.session_state:
    st.session_state['prediction_val'] = 0.0

# ---------------------------------------------------------
# Dashboard Interface
# ---------------------------------------------------------
st.markdown("""
<div class="dracula-card" style="border-left: 6px solid #bd93f9;">
    <h2 style="color: #ffffff; margin:0; padding:0; font-size: 24px; font-weight:700;">🏡 Real Estate Price Predictor</h2>
    <p style="color: #8be9fd; margin:0; font-size: 13px; font-weight: 500;">RandomForestRegressor Machine Learning Model</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1.9], gap="small")

# --- Left Column: Inputs ---
with col_left:
    st.markdown('<div class="dracula-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-title">Property Specifications</p>', unsafe_allow_html=True)
    
    square_ft = st.number_input("Area in Sq. Ft (SQUARE_FT)", min_value=100, max_value=50000, value=1000, step=50)
    bhk_no = st.selectbox("Number of BHK (BHK_NO.)", options=[1, 2, 3, 4, 5, 6], index=1)
    selected_location = st.selectbox("Property Location", options=locations)

    c1, c2 = st.columns(2)
    with c1:
        under_construction = BINARY_MAP[st.selectbox("Under Construction", ["No", "Yes"], index=0)]
        ready_to_move = BINARY_MAP[st.selectbox("Ready to Move", ["No", "Yes"], index=1)]
        bhk_or_rk_bhk = BINARY_MAP[st.selectbox("Type: BHK", ["No", "Yes"], index=1)]

    with c2:
        rera = BINARY_MAP[st.selectbox("RERA Approved", ["No", "Yes"], index=1)]
        resale = BINARY_MAP[st.selectbox("Resale Property", ["No", "Yes"], index=1)]
        bhk_or_rk_rk = BINARY_MAP[st.selectbox("Type: RK", ["No", "Yes"], index=0)]

    predict_btn = st.button("🚀 Estimate Valuation")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Compute Prediction Action ---
if predict_btn:
    input_dict = {feat: 0 for feat in feature_names}
    input_dict['SQUARE_FT'] = square_ft
    input_dict['BHK_NO.'] = bhk_no
    input_dict['UNDER_CONSTRUCTION'] = under_construction
    input_dict['RERA'] = rera
    input_dict['READY_TO_MOVE'] = ready_to_move
    input_dict['RESALE'] = resale
    input_dict['BHK_OR_RK_BHK'] = bhk_or_rk_bhk
    input_dict['BHK_OR_RK_RK'] = bhk_or_rk_rk
    
    # Preserved Background Coordinates
    input_dict['LATITUDE'] = 19.0760
    input_dict['LONGITUDE'] = 72.8777

    loc_feature_key = f"ADDRESS_{selected_location}"
    if loc_feature_key in input_dict:
        input_dict[loc_feature_key] = 1

    if model_loaded:
        input_df = pd.DataFrame([input_dict])
        st.session_state['prediction_val'] = float(model.predict(input_df)[0])
    else:
        base_price = (square_ft * 0.05) + (bhk_no * 12.5) + (rera * 5.0) - (under_construction * 3.0)
        st.session_state['prediction_val'] = max(10.0, base_price)

# --- Right Column: Outputs ---
with col_right:
    status_color = "#50fa7b" if model_loaded else "#ffb86c"
    status_text = "Active Pickle Model" if model_loaded else "Demo Estimator (Missing model.pkl)"

    st.markdown(f"""
    <div class="dracula-card">
        <p class="metric-title">Model Architecture Specs</p>
        <span class="stat-pill">Algorithm: RandomForestRegressor</span>
        <span class="stat-pill">Status: <span style="color: {status_color};">{status_text}</span></span>
    </div>
    """, unsafe_allow_html=True)

    val = st.session_state['prediction_val']
    st.markdown(f"""
    <div class="dracula-card" style="border: 2px solid #50fa7b; text-align: center;">
        <p class="metric-title" style="color: #50fa7b !important;">Estimated Property Price</p>
        <div class="metric-value" style="font-size: 38px; color: #f1fa8c;">
            ₹ {val:,.2f} Lakhs
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dracula-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-title">Selected Parameter Overview</p>', unsafe_allow_html=True)
    
    summary_df = pd.DataFrame({
        "Feature Parameter": ["Area", "Location", "BHK Count", "Under Construction", "RERA Approved", "Resale"],
        "Input Value": [f"{square_ft} sqft", selected_location, f"{bhk_no} BHK", "Yes" if under_construction else "No", "Yes" if rera else "No", "Yes" if resale else "No"],
        "Encoding": ["Numeric", "One-Hot Encoded", "Numeric", "Binary (0/1)", "Binary (0/1)", "Binary (0/1)"]
    })
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
