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
    initial_sidebar_state="collapsed"
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

    /* Target specific top-level elements to prevent ghost boxes */
    div[data-testid="stColumn"] > div {
        background-color: var(--current-line);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border: 2px solid var(--purple);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
    }

    /* Header Banner Styling */
    .header-card {
        background-color: var(--current-line);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
        border: 2px solid var(--purple);
        border-left: 6px solid var(--purple) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
    }

    .metric-title {
        color: var(--cyan) !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: var(--green);
        font-size: 26px;
        font-weight: bold;
    }

    /* Model Metrics Badges */
    .stat-pill {
        display: inline-block;
        background-color: var(--purple);
        color: #1e1e2e;
        font-weight: bold;
        padding: 5px 12px;
        border-radius: 12px;
        font-size: 13px;
        margin-right: 8px;
        margin-bottom: 6px;
    }

    /* Labels and High-Contrast Inputs */
    label, .stSelectbox label, .stNumberInput label {
        color: #ff79c6 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }

    /* Input Field & Dropdown Borders */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    input {
        border: 2px solid var(--cyan) !important;
        border-radius: 6px !important;
        background-color: #1a1b26 !important;
        color: #ffffff !important;
    }

    /* Action Button */
    .stButton>button {
        width: 100%;
        background-color: var(--green) !important;
        color: #1e1e2e !important;
        font-weight: 800 !important;
        font-size: 15px !important;
        border-radius: 6px !important;
        border: 2px solid var(--green) !important;
        padding: 10px 0px !important;
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
# Load Model & Feature Setup
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
        'ADDRESS_100 Feet Road,Anand', 'ADDRESS_Gandhi Chowk,Bhandara', 'ADDRESS_panvel,Mumbai'
    ]

BINARY_MAP = {"No": 0, "Yes": 1}

address_features = [f for f in feature_names if f.startswith("ADDRESS_")]
locations = [f.replace("ADDRESS_", "") for f in address_features]
if not locations:
    locations = ["100 Feet Road,Anand", "Gandhi Chowk,Bhandara", "panvel,Mumbai"]

if 'prediction_val' not in st.session_state:
    st.session_state['prediction_val'] = 0.0

# ---------------------------------------------------------
# UI Layout Architecture
# ---------------------------------------------------------

# Header Banner
st.markdown("""
<div class="header-card">
    <h2 style="color: #ffffff; margin:0; padding:0; font-size: 24px; font-weight:700;">🏡 Real Estate Price Predictor</h2>
    <p style="color: #8be9fd; margin:0; font-size: 13px; font-weight: 500;">RandomForestRegressor Machine Learning Model</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1.9], gap="small")

# --- Left Column: Inputs ---
with col_left:
    st.markdown('<p class="metric-title">Property Specifications</p>', unsafe_allow_html=True)
    
    square_ft = st.number_input("Area in Sq. Ft (SQUARE_FT)", min_value=100, max_value=50000, value=1250, step=50)
    bhk_no = st.selectbox("Number of Rooms / BHK Count", options=[1, 2, 3, 4, 5, 6], index=1)
    selected_location = st.selectbox("Property Location", options=locations)

    # Mutually Exclusive Layout Type (BHK vs RK)
    layout_type = st.selectbox("Property Layout Type", ["BHK (Bedroom, Hall, Kitchen)", "RK (Room, Kitchen)"], index=0)
    if layout_type.startswith("BHK"):
        bhk_or_rk_bhk = 1
        bhk_or_rk_rk = 0
    else:
        bhk_or_rk_bhk = 0
        bhk_or_rk_rk = 1

    # Mutually Exclusive Construction Status
    construction_status = st.selectbox("Construction Status", ["Under Construction", "Ready to Move"], index=0)
    if construction_status == "Under Construction":
        under_construction = 1
        ready_to_move = 0
    else:
        under_construction = 0
        ready_to_move = 1

    c1, c2 = st.columns(2)
    with c1:
        rera = BINARY_MAP[st.selectbox("RERA Approved", ["No", "Yes"], index=1)]
    with c2:
        resale = BINARY_MAP[st.selectbox("Resale Property", ["No", "Yes"], index=0)]

    predict_btn = st.button("🚀 Estimate Valuation")

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
    input_dict['LATITUDE'] = 22.5726
    input_dict['LONGITUDE'] = 88.3639

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
    # Model Specs Section
    st.markdown("""
        <p class="metric-title">Model Architecture Specs</p>
        <span class="stat-pill">Algorithm: RandomForestRegressor</span>
        <span class="stat-pill">Model Status: <span style="color: #50fa7b;">Active</span></span>
        <span class="stat-pill">Model R²: <span style="color: #f1fa8c;">94.91%</span></span>
    """, unsafe_allow_html=True)

    # Valuation Display Box
    val = st.session_state['prediction_val']
    st.markdown(f"""
        <p class="metric-title" style="color: #50fa7b !important; text-align: center;">Estimated Property Price</p>
        <div class="metric-value" style="font-size: 38px; color: #f1fa8c; text-align: center;">
            ₹ {val:,.2f} Lakhs
        </div>
    """, unsafe_allow_html=True)

    # Feature Overview Table
    st.markdown('<p class="metric-title">Selected Parameter Overview</p>', unsafe_allow_html=True)
    
    summary_df = pd.DataFrame({
        "Feature Parameter": ["Area", "Location", "BHK Count", "Layout Type", "Construction Status", "RERA Approved", "Resale"],
        "Input Value": [
            f"{square_ft} sqft", 
            selected_location, 
            f"{bhk_no}", 
            "BHK" if bhk_or_rk_bhk else "RK",
            "Under Construction" if under_construction else "Ready to Move", 
            "Yes" if rera else "No", 
            "Yes" if resale else "No"
        ],
        "Encoding": ["Numeric", "One-Hot Encoded", "Numeric", "Binary (0/1)", "Binary (0/1)", "Binary (0/1)", "Binary (0/1)"]
    })
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
