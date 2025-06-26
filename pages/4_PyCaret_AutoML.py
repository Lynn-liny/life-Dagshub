import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings
import os # Import os for path handling

warnings.filterwarnings('ignore')

# --- Module Availability Checks ---
# These blocks check if necessary libraries (MLflow, PyCaret, SHAP) are installed.
# If a library is not found, a warning is displayed, and related features are flagged as unavailable.
try:
    import mlflow
    import mlflow.sklearn
    import dagshub # Re-import dagshub for DagsHub integration
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    st.warning("MLflow not installed. Some features may be limited.")

try:
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    st.warning("PyCaret not installed. AutoML features will be limited.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --- Streamlit Page Configuration ---
# Sets up the basic configuration for the Streamlit application page.
# This applies globally to all pages in the multi-page app.
st.set_page_config(
    page_title="Life Expectancy Prediction 🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🫀"
)

# --- Custom CSS Styling ---
# Applies custom CSS to enhance the visual appeal of the Streamlit app.
# This styling will apply across all pages.
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

# --- Main Header Content ---
# Displays the main title and a brief description of the application.
# This will appear on the 'Home' page.
st.markdown("""
<div class="main-header">
    <h1 style="color: white; font-size: 3rem; margin-bottom: 0;">Life Expectancy Prediction 🫀</h1>
    <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">
        Machine Learning Pipeline for WHO Life Expectancy Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# --- User Authentication Function ---
# This function handles user login and applies across all pages of the app.
# If not authenticated, it stops the app until the correct password is entered.
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
        st.stop() # Stops rendering further until authentication is successful

# Call the authentication function to secure the app
check_authentication()

# --- MLflow Tracking Initialization ---
# Configures MLflow to track experiments.
if MLFLOW_AVAILABLE:
    try:
        # Initializing DagsHub for MLflow tracking.
        # This will set the MLflow tracking URI automatically to your DagsHub repo.
        dagshub.init(repo_owner='Tianjun-li-123', repo_name='DS4E-LIFE-EXP', mlflow=True)
        st.sidebar.success("✅ DagsHub initialized for MLflow")
    except Exception as e:
        st.sidebar.error(f"❌ DagsHub initialization failed: {str(e)}")
        # Provide guidance if token might be missing or permission issue
        if "AuthenticationError" in str(e) or "403" in str(e) or "401" in str(e):
             st.sidebar.warning(
                 "💡 DagsHub authentication failed. Please ensure your `DAGSHUB_USER_TOKEN` "
                 "environment variable is set with a valid DagsHub token that has write access to the repository. "
                 "You can generate one at `dagshub.com/user/settings/tokens`."
             )
        elif "unsupported endpoint" in str(e):
            st.sidebar.warning(
                "💡 DagsHub MLflow endpoint might not support all direct model logging features. "
                "The application attempts to use a workaround by logging models as `.joblib` artifacts. "
                "If issues persist, please refer to DagsHub documentation or contact their support."
            )
        else:
            st.sidebar.warning(f"💡 DagsHub initialization encountered an unexpected error. Details: {e}")


# --- Session State Initialization ---
# Initializes all necessary session state variables. These variables will persist
# across all pages of the multi-page Streamlit application.
if 'df' not in st.session_state:
    st.session_state.df = None # Stores the main DataFrame
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {} # Stores trained ML models
if 'pycaret_setup_done' not in st.session_state:
    st.session_state.pycaret_setup_done = False # Flag for PyCaret setup completion
if 'best_model' not in st.session_state:
    st.session_state.best_model = None # Stores the best model from AutoML
if 'model_comparison' not in st.session_state:
    st.session_state.model_comparison = None # Stores results of model comparison
# Add PyCaret specific session state variables that 5_Explainability.py relies on
if 'pycaret_selected_features' not in st.session_state:
    st.session_state.pycaret_selected_features = []
if 'automl_target_select' not in st.session_state:
    st.session_state.automl_target_select = None


# --- Initial Data Loading for Application ---
# This block attempts to load the dataset automatically once the application starts.
# The loaded DataFrame will be available in `st.session_state.df` for all pages.
DATA_FILE = "Life Expectancy Data.csv" # Path to your data file, assuming it's in the root project directory

if st.session_state.df is None: # Only load if DataFrame is not already in session state
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            st.session_state.df = df
            st.session_state.last_loaded = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # st.success(f"✅ Data loaded initially for application from {DATA_FILE}") # Optional: show success message
        else:
            # Warning if the data file is not found at the specified path
            st.warning(f"⚠️ Data file not found at '{DATA_FILE}'. "
                       "Please ensure it exists in your project's root directory. "
                       "You can also load it explicitly on the 'Data Loading' page.")
    except Exception as e:
        # Error message if any issue occurs during data loading
        st.error(f"❌ Error loading data initially: {str(e)}")


# --- Utility Function: Get Dataset Information ---
# A helper function to gather and return various statistics about the DataFrame.
# This can be imported by other pages if needed, or re-defined locally.
def get_dataset_info(df):
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    return info

# --- Home Page Content Display ---
# This section defines the content visible on the main (home) page of the application.
# Streamlit will automatically manage navigation to other pages placed in the 'pages/' directory.
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
        st.warning("⚠️ Data is not loaded. Please ensure the data file exists or use the **📊 Data Loading** page to load it.")
