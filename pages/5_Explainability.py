import streamlit as st
import pandas as pd
import shap # Library for model explainability
import matplotlib.pyplot as plt # Used for plotting SHAP visualizations
import numpy as np # Used for data type checks (e.g., np.number)
import os # For path operations if needed (not directly used here, but common)

# --- Page Header ---
st.header("🔬 Model Explainability with SHAP")
st.markdown("---") # Visual separator

# --- SHAP Availability Check ---
# Checks if the SHAP library is installed. If not, explainability features
# are disabled, and a warning is displayed.
try:
    shap_available = True
    shap.initjs() # Initializes JavaScript for interactive SHAP plots (important for some plot types)
except ImportError:
    shap_available = False
    st.error("❌ SHAP is not installed. Explainability features are disabled. Please install `shap` (`pip install shap`) to use this page.")
    st.stop() # Stop execution if SHAP is not available

# --- Data and Model Availability Checks ---
# These checks ensure that a DataFrame (`df`) is loaded and that at least one
# machine learning model has been trained and stored in session state.
if st.session_state.df is None:
    st.warning("⚠️ Please load data from the **📊 Data Loading** page first to perform SHAP analysis.")
    st.stop() # Stop execution if no data is loaded

# Check if any models are trained (either classical or PyCaret best model)
if not st.session_state.trained_models and ('best_model' not in st.session_state or st.session_state.best_model is None):
    st.warning("⚠️ No trained models available for explanation. Please train a model first on the "
               "**🤖 Classical ML** or **⚡ PyCaret AutoML** page.")
    st.stop() # Stop if no models are found

# --- Model Selection ---
# Allows the user to select which trained model they want to explain.
available_models = list(st.session_state.trained_models.keys()) # Models from Classical ML page
if 'best_model' in st.session_state and st.session_state.best_model is not None:
    available_models.append("PyCaret Best Model") # Add the finalized PyCaret model

if not available_models: # Fallback if somehow no models are available after initial checks
    st.warning("No models found to explain. Please train a model.")
    st.stop()

selected_model_name = st.selectbox("🤖 Select model to explain:", available_models, key="model_select")

# --- Prepare Model and Data for SHAP ---
# This section retrieves the selected model and the appropriate input data (X_shap)
# for SHAP analysis. It handles differences between classical ML models and PyCaret pipelines.
model = None
features_for_shap = [] # Features corresponding to the X_shap DataFrame
X_shap = pd.DataFrame() # DataFrame that will be used for SHAP explanation

if selected_model_name != "PyCaret Best Model":
    # --- Case: Classical ML Model ---
    # Retrieve the model and its associated test data/features from session state.
    # For classical models, X_test should already be preprocessed (scaled, encoded)
    # as per the `preprocess_data` function in `data_preprocessing.py`.
    model_data = st.session_state.trained_models[selected_model_name]
    model = model_data['model']
    features_for_shap = model_data['features']
    
    # Ensure X_test is a DataFrame with correct column names for SHAP.
    # If X_test is originally a numpy array, convert it to a DataFrame.
    if isinstance(model_data['X_test'], np.ndarray):
        X_shap = pd.DataFrame(model_data['X_test'], columns=features_for_shap)
    else: # Already a DataFrame
        X_shap = model_data['X_test']
    
    st.info(f"Using {selected_model_name} (Classical ML) for SHAP. Data should be preprocessed (scaled/encoded).")

else:
    # --- Case: PyCaret Best Model ---
    # PyCaret models are often wrapped in pipelines that handle preprocessing internally.
    # Therefore, SHAP should receive the *raw*, unscaled features.
    model = st.session_state.best_model

    # Retrieve features and target used during PyCaret setup from session state.
    if ('pycaret_selected_features' in st.session_state and 
        st.session_state.pycaret_selected_features and # Ensure list is not empty
        'automl_target_select' in st.session_state and 
        st.session_state.automl_target_select is not None):

        features_for_shap = st.session_state.pycaret_selected_features
        target_col = st.session_state.automl_target_select
        
        # Get a copy of the original cleaned DataFrame (from data loading page)
        # and filter it to include only the features selected for PyCaret.
        df_source_for_shap = st.session_state.df.copy()

        # IMPORTANT: Apply the same initial NA cleaning that was done before PyCaret setup (in 4_Pycaret_AutoML.py)
        # This prevents SHAP or the model's pipeline from crashing on NaNs in the sampled data.
        try:
            from data_preprocessing import clean_missing # Import clean_missing here if not already
            df_source_for_shap = clean_missing(df_source_for_shap, numeric_strategy="median")
        except ImportError:
            st.warning("Could not import `clean_missing` for PyCaret SHAP data preparation. "
                       "Ensure `data_preprocessing.py` is available. Missing values might cause issues.")
            # Basic fallback if clean_missing fails for some reason
            for col in df_source_for_shap.columns:
                if df_source_for_shap[col].dtype in [np.float64, np.int64]:
                    df_source_for_shap[col] = df_source_for_shap[col].fillna(df_source_for_shap[col].median())
                elif df_source_for_shap[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_source_for_shap[col]):
                    df_source_for_shap[col] = df_source_for_shap[col].fillna(df_source_for_shap[col].mode()[0])

        # Filter the DataFrame to include only the features used for PyCaret training
        # and exclude the target column from X_shap.
        valid_features_in_df = [f for f in features_for_shap if f in df_source_for_shap.columns]
        X_shap = df_source_for_shap[valid_features_in_df]

        # Use a small sample of the data for SHAP to avoid performance issues.
        # This sampling is applied to the raw features before SHAP calculation.
        if len(X_shap) > 1000: # Limit sample for SHAP computation
            X_shap = X_shap.sample(n=1000, random_state=42)
            
        st.info(f"Using {selected_model_name} (PyCaret) for SHAP. Data should be raw (unscaled/unencoded for PyCaret's pipeline).")

    else:
        st.error("❌ PyCaret features or target information not found in session state. "
                 "Please ensure PyCaret setup was completed on the '⚡ PyCaret AutoML' page and features/target were saved.")
        st.stop() # Stop if PyCaret info is missing


# --- SHAP Analysis and Visualization ---
st.subheader("🔬 SHAP Analysis Results")

if shap_available and model is not None and not X_shap.empty:
    try:
        # Limit to a manageable sample size for SHAP computation efficiency.
        # This slider allows users to control the number of instances for SHAP calculation.
        # Ensure the max value of the slider is correctly bounded by the available data.
        max_shap_sample_size = min(500, len(X_shap)) # Limit max to 500 or available data length
        sample_size = st.slider("🔧 Sample size for SHAP explanation (fewer instances = faster computation):", 
                                50, max(50, max_shap_sample_size), # Min 50, but not more than available
                                100, # Default value
                                key="shap_sample_size")
        
        # Ensure the sample is taken consistently, handling cases where X_shap might be smaller than sample_size
        X_shap_sampled = X_shap.sample(n=min(sample_size, len(X_shap)), random_state=42)
        
        if X_shap_sampled.empty:
            st.warning("SHAP sample is empty. Cannot generate plots.")
            st.stop()


        with st.spinner("Calculating SHAP values. This may take a while for larger samples/models..."):
            # Create a SHAP explainer for the selected model.
            # shap.Explainer automatically detects the best explainer (e.g., TreeExplainer for tree models).
            explainer = shap.Explainer(model, X_shap_sampled) # Pass the sampled data
            
            # Compute SHAP values for the sampled data.
            shap_values = explainer(X_shap_sampled) # Compute on the sampled data


        # --- Global Feature Importance (Bar Plot) ---
        # Shows the overall impact of each feature on the model's predictions.
        st.subheader("🌍 Global Feature Importance (Mean Absolute SHAP Value)")
        fig, ax = plt.subplots(figsize=(10, 6))
        # `shap_values` object can be passed directly. `plot_type="bar"` shows mean absolute SHAP values.
        shap.summary_plot(shap_values, X_shap_sampled, plot_type="bar", ax=ax, show=False)
        plt.tight_layout() # Adjust layout to prevent labels from being cut off
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free memory

        # --- Feature Impact Summary (Dot Plot / Beeswarm Plot) ---
        # Provides a more detailed view, showing the distribution of SHAP values
        # for each feature, and how feature values (color) influence impact.
        st.subheader("📊 Feature Impact Summary (SHAP Values per Feature)")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Default `plot_type` is "dot" (beeswarm plot), showing individual data points' SHAP values.
        shap.summary_plot(shap_values, X_shap_sampled, ax=ax, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Close the figure

        # --- Individual Prediction Explanation (Waterfall Plot) ---
        # Explains a single prediction by breaking down the contribution of each feature.
        st.subheader("🔍 Individual Prediction Explanation (Waterfall Plot)")
        
        # Allow user to select an instance from the *sampled* data for detailed explanation.
        # Ensure the slider range is based on the actual size of the sampled data for SHAP.
        max_instance_idx = max(0, len(X_shap_sampled) - 1)
        instance_idx = st.slider("Select instance index (from SHAP sample):", 
                                 0, max_instance_idx, 
                                 0, # Default to the first instance
                                 key="instance_slider")
        
        if len(shap_values) > instance_idx: # Ensure the index is valid
            fig, ax = plt.subplots(figsize=(10, 6))
            # Waterfall plot for the selected instance.
            # `shap_values[instance_idx]` provides the SHAP values for that specific data point.
            shap.plots.waterfall(shap_values[instance_idx], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig) # Close the figure
        else:
            st.warning("Selected instance index is out of bounds for the SHAP sample.")

    except Exception as e:
        st.error(f"❌ Error generating SHAP explanations: {str(e)}. "
                 "This can happen if the model type is not fully supported by SHAP, "
                 "if there are data issues (e.g., NaNs after sampling), or if model is not compatible with explainer.")
        st.write(f"Debug: selected_model_name={selected_model_name}, "
                 f"model_type={type(model)}, "
                 f"features_for_shap={features_for_shap}, "
                 f"X_shap_sampled_shape={X_shap_sampled.shape if not X_shap_sampled.empty else 'Empty'}")
        st.exception(e) # Display full traceback for debugging
else:
    st.info("🔍 SHAP analysis cannot be performed. Ensure SHAP is installed, data is loaded, and a model is trained.")

# --- Debugging Confirmation ---
st.write("Debug: 5_Explainability.py page loaded and executed.")
