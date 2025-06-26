import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Attempt to import utils, fall back to basic implementation if missing
try:
    from utils.data_preprocessing import clean_missing
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    st.warning("⚠️ 'utils' module not found. Using basic NA handling. Install or add utils/data_preprocessing.py for full functionality.")
    def clean_missing(df, numeric_strategy="median"):
        df_clean = df.copy()
        for column in df_clean.columns:
            if df_clean[column].dtype in [np.float64, np.int64]:
                if numeric_strategy == "median":
                    df_clean[column] = df_clean[column].fillna(df_clean[column].median())
                elif numeric_strategy == "mean":
                    df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
                elif numeric_strategy == "drop":
                    df_clean = df_clean.dropna(subset=[column])
                elif numeric_strategy == "knn":
                    st.warning("KNN imputation not implemented without utils. Using median instead.")
                    df_clean[column] = df_clean[column].fillna(df_clean[column].median())
        return df_clean

st.header("📊 Data Loading & Management")

# Automatically load data from a fixed location
DATA_FILE = "Life Expectancy Data.csv"  # Adjust path if in a subdirectory (e.g., "data/Life Expectancy Data.csv")
st.write(f"Debug: Checking file {DATA_FILE}")
if st.session_state.df is None:
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            st.session_state.df = df
            st.session_state.last_loaded = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"Debug: Loaded dataframe with shape {df.shape}")
            st.success(f"✅ Automatically loaded {df.shape[0]} rows and {df.shape[1]} columns from {DATA_FILE}")
        else:
            st.write(f"Debug: File {DATA_FILE} not found")
            st.error(f"❌ File {DATA_FILE} not found. Please ensure it exists in the project directory.")
    except Exception as e:
        st.write(f"Debug: Error loading file - {str(e)}")
        st.error(f"❌ Error loading {DATA_FILE}: {str(e)}")

# Data Info (always available after loading)
if st.session_state.df is not None:
    info = {
        'shape': st.session_state.df.shape,
        'columns': st.session_state.df.columns.tolist(),
        'missing_values': st.session_state.df.isnull().sum().to_dict(),
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
        st.metric("❌ Missing Values", sum(info['missing_values'].values()))
    with st.expander("📋 Detailed Info", expanded=False):
        st.json(info)
else:
    st.write("Debug: No dataframe loaded")
    st.info("🔍 Data not loaded. Check file availability.")

# Data Preview and Download
if st.session_state.df is not None:
    st.subheader("📋 Data Preview")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_rows = st.slider("Rows to display", 5, 50, 10, key="rows_slider")
    with col2:
        show_info = st.checkbox("Show column info", value=True, key="show_info_checkbox")
    with col3:
        if st.button("💾 Download Current Data", key="download_button"):
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"life_expectancy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv"
            )
    st.dataframe(st.session_state.df.head(show_rows), use_container_width=True)

    if show_info:
        st.subheader("📊 Column Information")
        info_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Data Type': st.session_state.df.dtypes,
            'Non-Null Count': st.session_state.df.count(),
            'Missing Values': st.session_state.df.isnull().sum(),
            'Missing %': (st.session_state.df.isnull().sum() / len(st.session_state.df) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)

    # Preprocessing
    st.subheader("🛠️ Preprocess Data")
    methods = {
        "Drop rows with any NA": "drop",
        "Fill numeric with median": "median",
        "Fill numeric with mean": "mean",
        "KNN imputation (k=5)": "knn"
    }
    choice = st.selectbox("Choose a strategy", list(methods.keys()), key="preprocess_strategy")
    if st.button("Apply Preprocessing", key="apply_preprocess_button"):
        if st.session_state.df is not None:
            df = st.session_state.df.copy()  # Avoid direct modification
            try:
                if methods[choice] == "drop":
                    df = df.dropna().reset_index(drop=True)
                else:
                    df = clean_missing(df, numeric_strategy=methods[choice])
                st.session_state.df = df
                st.success("✅ Preprocessing applied successfully!")
                st.write("Remaining NA counts:", df.isna().sum())
            except Exception as e:
                st.write(f"Debug: Preprocessing error - {str(e)}")
                st.error(f"❌ Preprocessing error: {str(e)}")
        else:
            st.error("❌ No data loaded to preprocess.")

# Debug
st.write("Data Loading page loaded")  # Confirm page is active