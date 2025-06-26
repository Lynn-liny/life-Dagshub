import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

# --- Module Availability Checks (Copied from streamlit_app.py for standalone functionality) ---
# These blocks ensure necessary libraries are available when this script runs independently.
try:
    import mlflow
    import mlflow.sklearn
    import dagshub
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    st.warning("MLflow not installed. Some MLflow-related features may be limited in this standalone app.")

try:
    # PyCaret is not directly used on this page, but included for consistency if needed later
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    # st.warning("PyCaret not installed. AutoML features will be limited.") # No need for warning on this page

try:
    # SHAP is not directly used on this page, but included for consistency if needed later
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --- Streamlit Page Configuration (Essential for Standalone App) ---
# Sets up the basic configuration for this Streamlit application page.
# This ensures it has a title, layout, and icon when run directly.
st.set_page_config(
    page_title="Data Loading & Management 📊", # Specific title for this page
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊" # Specific icon for this page
)

# --- Custom CSS Styling (Copied from streamlit_app.py for consistent look) ---
# Applies custom CSS to enhance the visual appeal, maintaining consistency with the main app.
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

# --- User Authentication Function (Copied for Standalone App) ---
# This function is crucial for securing this page when run independently.
def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        with st.sidebar:
            st.header("🔒 Authentication")
            password = st.text_input("Enter Password", type="password", key="auth_password_data_load") # Unique key
            if st.button("🔑 Login", key="login_btn_data_load"): # Unique key
                if password == "ds4everyone":
                    st.session_state.authenticated = True
                    st.success("✅ Access Granted!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect Password")
        st.info("🔐 Please authenticate to access this application")
        st.stop()

# Call the authentication function to secure this page
check_authentication()

# --- DagsHub and MLflow Initialization (Copied for Standalone App) ---
# Initializes DagsHub/MLflow for tracking experiments if this app is run independently.
if MLFLOW_AVAILABLE:
    try:
        dagshub.init(repo_owner='Tianjun-li-123', repo_name='DS4E-LIFE-EXP', mlflow=True)
        st.sidebar.success("✅ DagsHub initialized for MLflow")
    except Exception as e:
        st.sidebar.error(f"❌ DagsHub initialization failed: {str(e)}")

# --- Session State Initialization (Specific to this app) ---
# Initializes session state variables used within this specific data loading application.
# Note: When running independent apps (Option 3), 'df' in this app's session state
# is separate from 'df' in streamlit_app.py's session state.
if 'df' not in st.session_state:
    st.session_state.df = None
if 'last_loaded' not in st.session_state: # Ensure last_loaded is initialized
    st.session_state.last_loaded = 'N/A'
# Add other session state variables if this page relies on them from other parts of the "full" app
# Example: if 'trained_models' or 'pycaret_setup_done' are needed for context here, initialize them.
# For a pure data loading page, they might not be strictly necessary, but good to consider.
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'pycaret_setup_done' not in st.session_state:
    st.session_state.pycaret_setup_done = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'model_comparison' not in st.session_state:
    st.session_state.model_comparison = None


# --- Utility Functions (Local Fallback for 'data_preprocessing.py') ---
# This section attempts to import a cleaning function from data_preprocessing.py.
# If the file or function is not found, a basic fallback implementation is provided.
try:
    # Assuming data_preprocessing.py is in the same directory
    from data_preprocessing import clean_missing
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    st.warning("⚠️ 'data_preprocessing.py' module not found or 'clean_missing' not available. "
               "Using basic NA handling. Ensure 'data_preprocessing.py' "
               "exists with 'clean_missing' function for full functionality.")

    # Fallback implementation of clean_missing if the module is not found
    def clean_missing(df, numeric_strategy="median"):
        """
        A basic fallback function to handle missing numeric values.
        It's recommended to implement a more robust version in data_preprocessing.py.
        """
        df_clean = df.copy()
        for column in df_clean.columns:
            if df_clean[column].dtype in [np.float64, np.int64]:
                if df_clean[column].isnull().any(): # Only process if there are NaNs
                    if numeric_strategy == "median":
                        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
                    elif numeric_strategy == "mean":
                        df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
                    elif numeric_strategy == "drop":
                        df_clean = df_clean.dropna(subset=[column])
                    elif numeric_strategy == "knn":
                        st.warning("KNN imputation not implemented in fallback. Using median instead.")
                        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
            # For non-numeric columns, consider adding a strategy like mode imputation if necessary.
        return df_clean

# --- Page Header ---
st.header("📊 Data Loading & Management")
st.markdown("---") # Add a separator for better visual structure

# --- Automatic Data Loading ---
# This section attempts to load the dataset automatically from the specified path.
# It checks if the DataFrame is already in session state to avoid re-loading on every rerun,
# but will reload if the script is run freshly.
# Assuming 'Life Expectancy Data.csv' is in the same directory as this script.
DATA_FILE = "Life Expectancy Data.csv" 
st.info(f"Attempting to load data from: `{os.path.abspath(DATA_FILE)}`")

# Only attempt to load if 'df' is not already in session state (for this independent app's session)
if st.session_state.df is None:
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            st.session_state.df = df
            st.session_state.last_loaded = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"✅ Automatically loaded {df.shape[0]} rows and {df.shape[1]} columns from `{DATA_FILE}`")
        else:
            st.error(f"❌ File **`{DATA_FILE}`** not found. Please ensure it exists in the same directory as `1_Data_Loading.py`.")
            st.session_state.df = None 
    except Exception as e:
        st.error(f"❌ Error loading **`{DATA_FILE}`**: {str(e)}")
        st.session_state.df = None 

# --- Data Information Display ---
# This section displays key statistics about the loaded dataset if available.
if st.session_state.df is not None:
    st.subheader("Dataset Overview")
    info = {
        'shape': st.session_state.df.shape,
        'columns': st.session_state.df.columns.tolist(),
        'missing_values_count': st.session_state.df.isnull().sum().to_dict(),
        'memory_usage': f"{st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'numeric_columns': st.session_state.df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': st.session_state.df.select_dtypes(exclude=[np.number]).columns.tolist(),
        'last_loaded': getattr(st.session_state, 'last_loaded', 'N/A')
    }
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("📊 Rows", f"{info['shape'][0]:,}")
        st.metric("📋 Columns", f"{info['shape'][1]:,}")
        st.metric("💾 Memory Usage", info['memory_usage'])
    with col_b:
        st.metric("🔢 Numeric Columns", len(info['numeric_columns']))
        st.metric("📝 Categorical Columns", len(info['categorical_columns']))
        st.metric("❌ Total Missing Values", sum(info['missing_values_count'].values()))
    
    with st.expander("📋 Detailed Info", expanded=False):
        st.json(info)
else:
    st.info("🔍 Data not loaded. Please ensure the data file is present and accessible.")

# --- Data Preview and Download ---
# Allows users to view a snippet of the data and download the current DataFrame.
if st.session_state.df is not None:
    st.subheader("📋 Data Preview")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_rows = st.slider("Rows to display", 5, 50, 10, key="rows_slider")
    with col2:
        show_info = st.checkbox("Show column info", value=True, key="show_info_checkbox")
    with col3:
        # Prepare CSV data for download button
        csv_data = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="💾 Download Current Data",
            data=csv_data,
            file_name=f"life_expectancy_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_csv_actual_button"
        )
    
    st.dataframe(st.session_state.df.head(show_rows), use_container_width=True)

    if show_info:
        st.subheader("📊 Column Information (Data Types and Missing Values)")
        info_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Data Type': st.session_state.df.dtypes,
            'Non-Null Count': st.session_state.df.count(),
            'Missing Values': st.session_state.df.isnull().sum(),
            'Missing %': (st.session_state.df.isnull().sum() / len(st.session_state.df) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)

    # --- Data Preprocessing Section ---
    # Provides options to handle missing values in the dataset.
    st.subheader("🛠️ Preprocess Data (Handle Missing Values)")
    methods = {
        "Drop rows with any NA": "drop",
        "Fill numeric with median": "median",
        "Fill numeric with mean": "mean",
        "KNN imputation (k=5)": "knn" # Note: KNN is only fully supported if data_preprocessing.py is robustly implemented
    }
    choice = st.selectbox("Choose a strategy for missing values", list(methods.keys()), key="preprocess_strategy")
    
    if st.button("Apply Preprocessing", key="apply_preprocess_button"):
        if st.session_state.df is not None:
            df_to_preprocess = st.session_state.df.copy() # Work on a copy to avoid unintended side effects
            try:
                if methods[choice] == "drop":
                    initial_rows = df_to_preprocess.shape[0]
                    df_processed = df_to_preprocess.dropna().reset_index(drop=True)
                    st.success(f"✅ Dropped {initial_rows - df_processed.shape[0]} rows with missing values.")
                else:
                    df_processed = clean_missing(df_to_preprocess, numeric_strategy=methods[choice])
                    st.success(f"✅ Preprocessing with '{choice}' applied successfully!")
                
                st.session_state.df = df_processed # Update the session state DataFrame with processed data
                
                # Display remaining NA counts after preprocessing
                remaining_nas = st.session_state.df.isnull().sum().sum()
                if remaining_nas > 0:
                    st.warning(f"⚠️ Some missing values may still remain. Total remaining NA counts: {remaining_nas}")
                    with st.expander("Show remaining NA counts per column"):
                        st.dataframe(st.session_state.df.isnull().sum()[st.session_state.df.isnull().sum() > 0].to_frame('Missing Count'))
                else:
                    st.success("🎉 All missing values handled!")

            except Exception as e:
                st.error(f"❌ Preprocessing error: {str(e)}")
        else:
            st.error("❌ No data loaded to preprocess.")

# --- Debugging Confirmation ---
# This line helps confirm that the script has executed.
st.write("Debug: 1_Data_Loading.py (standalone) execution complete.")
