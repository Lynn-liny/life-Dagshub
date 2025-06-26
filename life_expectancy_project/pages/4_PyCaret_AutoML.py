import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn

# Attempt to import PyCaret and utils, fall back if missing
try:
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, create_model as reg_create
    from pycaret.regression import tune_model as reg_tune, finalize_model as reg_finalize, predict_model as reg_predict, pull as reg_pull, plot_model as reg_plot
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    st.warning("⚠️ PyCaret not installed. AutoML features are disabled. Install pycaret to proceed.")
    def reg_setup(*args, **kwargs): return None
    def reg_compare(*args, **kwargs): return None
    def reg_create(*args, **kwargs): return None
    def reg_tune(*args, **kwargs): return None
    def reg_finalize(*args, **kwargs): return None
    def reg_predict(*args, **kwargs): return None
    def reg_pull(*args, **kwargs): return None
    def reg_plot(*args, **kwargs): return None

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
                df_clean[column] = df_clean[column].fillna(df_clean[column].median())
        return df_clean

st.header("⚡ PyCaret AutoML")

# Check for data
if st.session_state.df is None:
    st.warning("⚠️ Please load data from the 📊 Data Loading page first")
    st.stop()

df = st.session_state.df
df = clean_missing(df, numeric_strategy="median")

# AutoML Configuration
st.subheader("⚙️ AutoML Configuration")
col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox("🎯 Select target variable:", df.columns, index=0 if not df.columns else (df.columns.get_loc("Life expectancy") if "Life expectancy" in df.columns else 0), key="target_select")
    available_features = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect("📊 Select features:", available_features, default=[col for col in available_features if col not in ["Country"]][:5], key="features_select")
with col2:
    train_size = st.slider("🔄 Training set size:", 0.5, 0.9, 0.8, 0.05, key="train_size_slider")
    sample_size = st.slider("📊 Sample size (for performance):", 500, min(5000, len(df)), 1000, key="sample_size_slider")

if len(df) > sample_size:
    df_sample = df.sample(n=sample_size, random_state=42)
    st.info(f"📊 Using {sample_size} samples for faster processing")
else:
    df_sample = df.copy()

with st.expander("🔧 Advanced Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        cross_validation = st.checkbox("🔄 Cross Validation", value=True, key="cross_validation_check")
        normalize = st.checkbox("📏 Normalize Features", value=True, key="normalize_check")
    with col2:
        remove_outliers = st.checkbox("🚫 Remove Outliers", value=False, key="remove_outliers_check")
        feature_selection = st.checkbox("🎯 Feature Selection", value=False, key="feature_selection_check")

if st.button("🚀 Setup PyCaret Environment", key="setup_button"):
    with st.spinner("Setting up PyCaret environment..."):
        try:
            if not PYCARET_AVAILABLE:
                raise ImportError("PyCaret not installed.")
            st.session_state.pycaret_exp = reg_setup(
                data=df_sample,
                target=target_col,
                train_size=train_size,
                session_id=42,
                normalize=normalize,
                remove_outliers=remove_outliers,
                feature_selection=feature_selection,
                silent=True,
                preprocess=False  # Let clean_missing handle initial preprocessing
            )
            st.session_state.pycaret_setup_done = True
            st.session_state.pycaret_problem_type = "regression"
            st.success("✅ PyCaret environment setup complete!")
        except Exception as e:
            st.error(f"❌ Error setting up PyCaret: {str(e)}. Ensure PyCaret is installed and data is valid.")

if st.session_state.pycaret_setup_done:
    st.subheader("📊 Model Comparison")
    if st.button("🔄 Compare Models", key="compare_button"):
        with st.spinner("Comparing multiple models..."):
            try:
                if not PYCARET_AVAILABLE:
                    raise ImportError("PyCaret not available for comparison.")
                comparison_df = reg_compare(include=['lr', 'rf', 'et', 'dt', 'huber'], sort='R2', n_select=5)
                st.session_state.model_comparison = reg_pull()
                st.success("✅ Model comparison complete!")
            except Exception as e:
                st.error(f"❌ Error comparing models: {str(e)}")

    if st.session_state.model_comparison is not None:
        st.subheader("📈 Model Comparison Results")
        st.dataframe(st.session_state.model_comparison[["Model", "MAE", "R2"]], use_container_width=True)
        
        best_model_name = st.selectbox("🏆 Select model for tuning:", ['lr', 'rf', 'et', 'dt', 'huber'], key="best_model_select")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Create Model", key="create_button"):
                with st.spinner("Creating model..."):
                    try:
                        if not PYCARET_AVAILABLE:
                            raise ImportError("PyCaret not available for model creation.")
                        model = reg_create(best_model_name)
                        st.session_state.pycaret_model = model
                        st.success("✅ Model created successfully!")
                    except Exception as e:
                        st.error(f"❌ Error creating model: {str(e)}")
        with col2:
            if st.button("⚡ Tune Hyperparameters", key="tune_button"):
                if 'pycaret_model' in st.session_state:
                    with st.spinner("Tuning hyperparameters..."):
                        try:
                            tuned_model = reg_tune(st.session_state.pycaret_model, optimize='R2', n_iter=10)
                            st.session_state.tuned_model = tuned_model
                            st.success("✅ Hyperparameter tuning complete!")
                        except Exception as e:
                            st.error(f"❌ Error tuning model: {str(e)}")
                else:
                    st.warning("⚠️ Please create a model first")
        
        if st.button("🏁 Finalize Best Model", key="finalize_button"):
            if 'tuned_model' in st.session_state:
                model_to_finalize = st.session_state.tuned_model
            elif 'pycaret_model' in st.session_state:
                model_to_finalize = st.session_state.pycaret_model
            else:
                st.warning("⚠️ Please create a model first")
                model_to_finalize = None
            if model_to_finalize is not None:
                with st.spinner("Finalizing model..."):
                    try:
                        final_model = reg_finalize(model_to_finalize)
                        st.session_state.best_model = final_model
                        with mlflow.start_run(run_name=f"PyCaret_{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                            mlflow.log_param("model_type", best_model_name)
                            mlflow.log_metric("R2", st.session_state.model_comparison.loc[st.session_state.model_comparison['Model'] == best_model_name, "R2"].iloc[0])
                            mlflow.log_metric("MAE", st.session_state.model_comparison.loc[st.session_state.model_comparison['Model'] == best_model_name, "MAE"].iloc[0])
                            mlflow.sklearn.log_model(final_model, "pycaret_model")
                        st.success("✅ Model finalized and logged to MLFlow!")
                    except Exception as e:
                        st.error(f"❌ Error finalizing model: {str(e)}")