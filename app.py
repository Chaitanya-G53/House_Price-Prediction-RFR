import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Page Configuration & Dracula Dark Theme Injection
# ---------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dracula Dark Theme fitting standard screens
st.markdown("""
    <style>
    /* Dracula Color Palette Settings */
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

    /* Main Container Padding Minimization for Screenshots */
    .stApp {
        background-color: var(--background);
        color: var(--foreground);
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 98% !important;
    }

    /* Card Scaffolding */
    .dracula-card {
        background-color: var(--current-line);
        border-radius: 8px;
        padding: 12px 18px;
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
        font-size: 22px;
        font-weight: bold;
    }

    /* Model Stats Pills */
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

    /* Input Labels and Text Contrast */
    label, .stSelectbox label, .stNumberInput label {
        color: var(--pink) !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    /* Customizing Streamlit Button */
    .stButton>button {
        width: 100%;
        background-color: var(--green) !important;
        color: var(--background) !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 6px 0px !important;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--cyan) !important;
        transform: scale(1.01);
    }

    /* Hide standard Streamlit header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Pickle Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # Replace 'model.pkl' with your actual .pkl file path
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    feature_names = list(model.feature_names_in_)
except Exception as e:
    # Fallback/Demo features extracted from your pkl string
    feature_names = [
        'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_FT', 'READY_TO_MOVE',
        'RESALE', 'LONGITUDE', 'LATITUDE', 'BHK_OR_RK_BHK', 'BHK_OR_RK_RK',
        'ADDRESS_, panvel,Mumbai', 'ADDRESS_,Manoramaganj,Indore', 'ADDRESS_100 Feet Road,Anand'
    ]
    model = None

# Extract non-address numeric features and address categorical features
address_features = [f for f in feature_names if f.startswith("ADDRESS_")]
base_features = [f for f in feature_names if not f.startswith("ADDRESS_")]

# Extract clean location names for dropdown selection
locations = [f.replace("ADDRESS_", "") for f in address_features]

# ---------------------------------------------------------
# UI Layout Architecture (Single-Screen Dashboard)
# ---------------------------------------------------------

# Top Header Banner
st.markdown("""
<div class="dracula-card" style="border-left: 5px solid #bd93f9;">
    <h2 style="color: #f8f8f2; margin:0; padding:0; font-size: 24px;">🏡 House Price Prediction Dashboard</h2>
    <p style="color: #6272a4; margin:0; font-size: 13px;">RandomForestRegressor Machine Learning Deployment</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1.9], gap="small")

# --- Left Column: Inputs & Controls ---
with col_left:
    st.markdown('<div class="dracula-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-title" style="color:#ff79c6;">Input Parameters</p>', unsafe_allow_html=True)
    
    square_ft = st.number_input("Square Feet (SQUARE_FT)", min_value=100, max_value=50000, value=1000, step=50)
    bhk_no = st.selectbox("Number of BHK (BHK_NO.)", options=[1, 2, 3, 4, 5, 6], index=1)
    
    c1, c2 = st.columns(2)
    with c1:
        latitude = st.number_input("Latitude", value=19.0760, format="%.4f")
        under_construction = st.selectbox("Under Construction", [0, 1], index=0)
        ready_to_move = st.selectbox("Ready to Move", [0, 1], index=1)
        bhk_or_rk_bhk = st.selectbox("BHK Type (BHK)", [0, 1], index=1)
    with c2:
        longitude = st.number_input("Longitude", value=72.8777, format="%.4f")
        rera = st.selectbox("RERA Approved", [0, 1], index=1)
        resale = st.selectbox("Resale", [0, 1], index=1)
        bhk_or_rk_rk = st.selectbox("RK Type (RK)", [0, 1], index=0)

    selected_location = st.selectbox("Select Property Location", options=locations if locations else ["Default Location"])
    predict_btn = st.button("🚀 Predict Estimated Price")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Right Column: Predictions & Model Factors ---
with col_right:
    # Model Specs Section
    st.markdown(f"""
    <div class="dracula-card">
        <p class="metric-title">Model Specifications & Architecture</p>
        <span class="stat-pill">Algorithm: RandomForestRegressor</span>
        <span class="stat-pill">Estimators: 40 Trees</span>
        <span class="stat-pill">Criterion: Squared Error</span>
        <span class="stat-pill">Max Features: Auto/Sqrt</span>
        <div style="margin-top: 8px; color: #f8f8f2; font-size: 12px;">
            <b>Total Features Trained:</b> {len(feature_names)} features | <b>Sklearn Version:</b> 1.6.1
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Compute Prediction
    prediction_val = 0.0
    if predict_btn and model is not None:
        input_dict = {feat: 0 for feat in feature_names}
        input_dict['SQUARE_FT'] = square_ft
        input_dict['BHK_NO.'] = bhk_no
        input_dict['LATITUDE'] = latitude
        input_dict['LONGITUDE'] = longitude
        input_dict['UNDER_CONSTRUCTION'] = under_construction
        input_dict['RERA'] = rera
        input_dict['READY_TO_MOVE'] = ready_to_move
        input_dict['RESALE'] = resale
        input_dict['BHK_OR_RK_BHK'] = bhk_or_rk_bhk
        input_dict['BHK_OR_RK_RK'] = bhk_or_rk_rk

        loc_feature_key = f"ADDRESS_{selected_location}"
        if loc_feature_key in input_dict:
            input_dict[loc_feature_key] = 1

        input_df = pd.DataFrame([input_dict])
        prediction_val = model.predict(input_df)[0]

    # Prediction Output Card
    st.markdown(f"""
    <div class="dracula-card" style="border: 1px solid #50fa7b; text-align: center;">
        <p class="metric-title" style="color: #50fa7b;">Predicted House Valuation</p>
        <div class="metric-value" style="font-size: 32px; color: #f1fa8c;">
            ₹ {prediction_val:,.2f} Lakhs
        </div>
        <p style="color: #6272a4; font-size: 11px; margin: 0;">Estimated using active ensemble decision pathways</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Feature Summary Table
    st.markdown('<div class="dracula-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-title">Active Feature Impact Vector</p>', unsafe_allow_html=True)
    
    summary_df = pd.DataFrame({
        "Feature Parameter": ["Area (Sq. Ft)", "Location", "BHK Count", "RERA Status", "Coordinates"],
        "Input Value": [f"{square_ft} sqft", selected_location, f"{bhk_no} BHK", "Approved" if rera else "Non-RERA", f"({latitude}, {longitude})"],
        "Data Type": ["Continuous", "Categorical (OHE)", "Discrete", "Binary", "Geospatial"]
    })
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
