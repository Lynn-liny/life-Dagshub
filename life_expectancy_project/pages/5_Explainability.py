import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.header("🔬 Model Explainability with SHAP")

# Check for SHAP availability
try:
    shap_available = True
    shap.initjs()  # Enable JavaScript for interactive plots if supported
except ImportError:
    shap_available = False
    st.warning("⚠️ SHAP is not installed. Explainability features are disabled. Install shap to proceed.")
    st.stop()

# Check for data and models
if st.session_state.df is None:
    st.warning("⚠️ Please load data from the 📊 Data Loading page first")
    st.stop()

if not st.session_state.trained_models and 'best_model' not in st.session_state:
    st.warning("⚠️ No trained models available. Please train a model first (Classical ML or PyCaret).")
    st.stop()

# Model selection
available_models = list(st.session_state.trained_models.keys())
if 'best_model' in st.session_state:
    available_models.append("PyCaret Best Model")
selected_model = st.selectbox("🤖 Select model to explain:", available_models, key="model_select")

# Prepare model and data
if selected_model != "PyCaret Best Model":
    model_data = st.session_state.trained_models[selected_model]
    model = model_data['model']
    features = model_data['features']
    X_test = pd.DataFrame(model_data['X_test'], columns=features)  # Ensure proper DataFrame
else:
    model = st.session_state.best_model
    features = st.session_state.model_comparison['features'] if 'features' in st.session_state.model_comparison else st.session_state.df.columns.drop(st.session_state.df.columns[0], errors='ignore').tolist()  # Fallback to first column as target
    X_test = st.session_state.df[features].iloc[-100:].copy()  # Use last 100 rows

st.subheader("🔬 SHAP Analysis")
if shap_available:
    try:
        with st.spinner("Creating SHAP explainer..."):
            # Limit to a manageable sample size
            sample_size = st.slider("🔧 Sample size for SHAP:", 50, min(500, len(X_test)), 100, key="shap_sample_size")
            explainer = shap.Explainer(model, X_test.sample(n=sample_size, random_state=42))
            shap_values = explainer(X_test.sample(n=sample_size, random_state=42))

        # Global Feature Importance
        st.subheader("🌍 Global Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test.sample(n=sample_size, random_state=42), plot_type="bar", ax=ax, show=False)
        st.pyplot(fig)

        # Feature Impact Summary
        st.subheader("📊 Feature Impact Summary")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test.sample(n=sample_size, random_state=42), ax=ax, show=False)
        st.pyplot(fig)

        # Individual Prediction Explanation
        st.subheader("🔍 Individual Prediction Explanation")
        instance_idx = st.slider("Select instance:", 0, min(sample_size - 1, len(X_test) - 1), 0, key="instance_slider")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[instance_idx], ax=ax, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error generating SHAP explanations: {str(e)}. Check model compatibility or data integrity.")
        st.write(f"Debug: selected_model={selected_model}, features={features}, X_test_shape={X_test.shape if 'X_test' in locals() else 'N/A'}")
else:
    st.info("🔍 SHAP is not available. Install the shap library to enable explainability.")

# Cleanup
plt.close('all')  # Prevent memory leaks from multiple plots