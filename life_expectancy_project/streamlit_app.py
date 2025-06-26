import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# MLflow and DagsHub
try:
    import mlflow
    import mlflow.sklearn
    import dagshub
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    st.warning("MLflow not installed. Some features may be limited.")

# PyCaret imports
try:
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    st.warning("PyCaret not installed. AutoML features will be limited.")

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Custom CSS
st.set_page_config(
    page_title="Life Expectancy Prediction 🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🫀"
)

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Arial', sans-serif; }
    .sidebar .sidebar-content { background: linear-gradient(180deg, #2C3E50, #3498DB); color: white; }
    .stButton > button { background: linear-gradient(45deg, #FF6B6B, #4ECDC4); color: white; border: none; border-radius: 25px; padding: 0.6rem 1.5rem; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.37); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px 0 rgba(31, 38, 135, 0.37); }
    .metric-container { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 1rem; margin: 0.5rem 0; border: 1px solid rgba(255, 255, 255, 0.2); }
    .main-header { text-align: center; padding: 2rem 0; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.2); }
    .stSuccess, .stError, .stWarning { border-radius: 10px; border: none; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; font-size: 3rem; margin-bottom: 0;">Life Expectancy Prediction 🫀</h1>
    <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">
        Machine Learning Pipeline for WHO Life Expectancy Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# Authentication
def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        with st.sidebar:
            st.header("🔒 Authentication")
            password = st.text_input("Enter Password", type="password", key="auth_password")
            if st.button("🔑 Login", key="login_btn"):
                if password == "ds4everyone":
                    st.session_state.authenticated = True
                    st.success("✅ Access Granted!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect Password")
        st.info("🔐 Please authenticate to access the application")
        st.stop()

check_authentication()

# Initialize DagsHub for MLFlow with error handling
if MLFLOW_AVAILABLE:
    try:
        dagshub.init(repo_owner='Tianjun-li-123', repo_name='DS4E-LIFE-EXP', mlflow=True)
        st.sidebar.success("✅ DagsHub initialized for MLflow")
    except Exception as e:
        st.sidebar.error(f"❌ DagsHub initialization failed: {str(e)}")

# Session State Initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'pycaret_setup_done' not in st.session_state:
    st.session_state.pycaret_setup_done = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'model_comparison' not in st.session_state:
    st.session_state.model_comparison = None

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
pages = [
    "🏠 Home",
    "📊 Data Loading",
    "📈 Visualization",
    "🤖 Classical ML",
    "⚡ PyCaret AutoML",
    "🔬 Explainability",
    "📋 MLflow Tracking"
]
selected_page = st.sidebar.selectbox("Select Page", pages, key="page_selector")

# Utility Functions
def get_dataset_info(df):
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    return info

# Home Page Content
if selected_page == "🏠 Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ## Welcome to the Life Expectancy Prediction App! 🎉
        This app provides a complete ML pipeline for predicting life expectancy using the WHO dataset:
        - Start by loading and preprocessing your data on the **📊 Data Loading** page.
        - Explore insights with **📈 Visualization**.
        - Train models using **🤖 Classical ML** or **⚡ PyCaret AutoML**.
        - Analyze feature importance with **🔬 Explainability**.
        - Track experiments via **📋 MLflow Tracking**.
        """)
        st.markdown("---")
        if st.session_state.df is not None:
            info = get_dataset_info(st.session_state.df)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("📊 Rows", f"{info['shape'][0]:,}")
            with col_b:
                st.metric("📋 Columns", f"{info['shape'][1]:,}")
            with col_c:
                st.metric("🤖 Models Trained", len(st.session_state.trained_models))
            with st.expander("📋 Dataset Details"):
                st.json(info)
        else:
            st.warning("⚠️ Please load data on the **📊 Data Loading** page first to enable full functionality.")

