import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Page Configuration & Dracula Dark Theme Setup
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
        --background: #282a36;
        --current-line: #44475a;
        --foreground: #f8f8f2;
        --comment: #6272a4;
        --cyan: #8be9fd;
        --green: #50fa7b;
        --orange: #ffb86c;
        --pink: #ff79c6;
        --purple: #bd93f9;
        --red: #ff5555;
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

    .dracula-card {
        background-color: var(--current-line);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border: 1px solid var(--purple);
    }

    .metric-title {
        color: var(--cyan);
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    .metric-value {
        color: var(--green);
        font-size: 24px;
        font-weight: bold;
    }

    .stat-pill {
        display: inline-block;
        background-color: var(--purple);
        color: var(--background);
        font-weight: bold;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 5px;
    }

    label, .stSelectbox label, .stNumberInput label {
        color: var(--pink) !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    .stButton>button {
        width: 100%;
        background-color: var(--green) !important;
        color: var(--background) !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 8px 0px !important;
        margin-top: 10px;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--cyan) !important;
        transform: scale(1.01);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model & Prepare Features
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
    feature_names = list(model.feature_names_in_)
except Exception:
    feature_names = [
        'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_FT', 'READY_TO_MOVE',
        'RESALE', 'LONGITUDE', 'LATITUDE', 'BHK_OR_RK_BHK', 'BHK_OR_RK_RK',
        'ADDRESS_, panvel,Mumbai', 'ADDRESS_,Manoramaganj,Indore', 'ADDRESS_100 Feet Road,Anand'
    ]
    model = None

# Binary Mapping Helper: UI Label -> Integer Value
BINARY_MAP = {"No (0)": 0, "Yes (1)": 1}

address_features = [f for f in feature_names if f.startswith("ADDRESS_")]
locations = [f.replace("ADDRESS_", "") for f in address_features]

# ---------------------------------------------------------
# Dashboard Interface
# ---------------------------------------------------------
st.markdown("""
<div class="dracula-card" style="border-left: 5px solid #bd93f9;">
    <h2 style="color: #f8f8f2; margin:0; padding:0; font-size: 22px;">🏡 Real Estate Price Predictor</h2>
    <p style="color: #6272a4; margin:0; font-size: 12px;">RandomForestRegressor Machine Learning Model</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1.9], gap="small")

# --- Left Column: Streamlined Inputs ---
with col_left:
    st.markdown('<div class="dracula-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-title" style="color:#ff79c6;">Property Specifications</p>', unsafe_allow_html=True)
    
    square_ft = st.number_input("Area in Sq. Ft (SQUARE_FT)", min_value=100, max_value=50000, value=1000, step=50)
    bhk_no = st.selectbox("Number of BHK (BHK_NO.)", options=[1, 2, 3, 4, 5, 6], index=1)
    selected_location = st.selectbox("Property Location", options=locations if locations else ["Default Location"])

    c1, c2 = st.columns(2)
    with c1:
        under_construction = BINARY_MAP[st.selectbox("Under Construction", ["No (0)", "Yes (1)"], index=0)]
        ready_to_move = BINARY_MAP[st.selectbox("Ready to Move", ["No (0)", "Yes (1)"], index=1)]
        bhk_or_rk_bhk = BINARY_MAP[st.selectbox("Type: BHK", ["No (0)", "Yes (1)"], index=1)]

    with c2:
        rera = BINARY_MAP[st.selectbox("RERA Approved", ["No (0)", "Yes (1)"], index=1)]
        resale = BINARY_MAP[st.selectbox("Resale Property", ["No (0)", "Yes (1)"], index=1)]
        bhk_or_rk_rk = BINARY_MAP[st.selectbox("Type: RK", ["No (0)", "Yes (1)"], index=0)]

    predict_btn = st.button("🚀 Estimate Valuation")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Right Column: Predictions & Model Factors ---
with col_right:
    st.markdown(f"""
    <div class="dracula-card">
        <p class="metric-title">Model Architecture Specs</p>
        <span class="stat-pill">Algorithm: RandomForestRegressor</span>
        <span class="stat-pill">Estimators: 40 Trees</span>
        <span class="stat-pill">Features: {len(feature_names)} Active</span>
    </div>
    """, unsafe_allow_html=True)

    prediction_val = 0.0
    if predict_btn and model is not None:
        input_dict = {feat: 0 for feat in feature_names}
        
        # Binary & Numerical Inputs
        input_dict['SQUARE_FT'] = square_ft
        input_dict['BHK_NO.'] = bhk_no
        input_dict['UNDER_CONSTRUCTION'] = under_construction
        input_dict['RERA'] = rera
        input_dict['READY_TO_MOVE'] = ready_to_move
        input_dict['RESALE'] = resale
        input_dict['BHK_OR_RK_BHK'] = bhk_or_rk_bhk
        input_dict['BHK_OR_RK_RK'] = bhk_or_rk_rk

        # Lat/Long preserved silently in background
        input_dict['LATITUDE'] = 19.0760
        input_dict['LONGITUDE'] = 72.8777

        # Location Hot-Encoding
        loc_feature_key = f"ADDRESS_{selected_location}"
        if loc_feature_key in input_dict:
            input_dict[loc_feature_key] = 1

        input_df = pd.DataFrame([input_dict])
        prediction_val = model.predict(input_df)[0]

    st.markdown(f"""
    <div class="dracula-card" style="border: 1px solid #50fa7b; text-align: center;">
        <p class="metric-title" style="color: #50fa7b;">Estimated Property Price</p>
        <div class="metric-value" style="font-size: 32px; color: #f1fa8c;">
            ₹ {prediction_val:,.2f} Lakhs
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dracula-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-title">Selected Parameter Overview</p>', unsafe_allow_html=True)
    
    summary_df = pd.DataFrame({
        "Feature Parameter": ["Area", "Location", "BHK Count", "Under Construction", "RERA Approved", "Resale"],
        "Input Value": [f"{square_ft} sqft", selected_location, f"{bhk_no} BHK", "Yes (1)" if under_construction else "No (0)", "Yes (1)" if rera else "No (0)", "Yes (1)" if resale else "No (0)"],
        "Encoding": ["Numeric", "One-Hot Encoded", "Numeric", "Binary (0/1)", "Binary (0/1)", "Binary (0/1)"]
    })
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
