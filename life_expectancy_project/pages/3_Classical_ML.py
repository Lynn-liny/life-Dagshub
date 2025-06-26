import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Attempt to import utils, fall back to basic implementation if missing
try:
    from utils.data_preprocessing import preprocess_data
    from utils.model_training import train_and_evaluate_model
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    st.warning("⚠️ 'utils' module not found. Basic training disabled. Install or add utils/data_preprocessing.py and utils/model_training.py for full functionality.")
    def preprocess_data(df, features, target):
        st.error("Preprocessing not available without utils. Please ensure utils module is present.")
        return None, None, None, None
    def train_and_evaluate_model(model_type, X, y, params, test_size):
        st.error("Model training not available without utils. Please ensure utils module is present.")
        return None, None, None, None, None, None, None, None

st.header("🤖 Classical Machine Learning")

# Check for data
if st.session_state.df is None:
    st.warning("⚠️ Please load data from the 📊 Data Loading page first")
    st.stop()

df = st.session_state.df

# Model Configuration
st.subheader("⚙️ Model Configuration")
col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox("🎯 Select target variable:", df.columns, index=0 if not df.columns else (df.columns.get_loc("Life expectancy") if "Life expectancy" in df.columns else 0), key="target_select")
    available_features = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect("📊 Select features:", available_features, default=[col for col in available_features if col not in ["Country"]][:5], key="features_select")
with col2:
    selected_model = st.selectbox("🤖 Select model:", ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"], key="model_select")
    test_size = st.slider("🔄 Test set size:", 0.1, 0.5, 0.2, 0.05, key="test_size_slider")
    params = {}
    if selected_model in ["Decision Tree", "Random Forest", "XGBoost"]:
        if selected_model == "Decision Tree":
            params["max_depth"] = st.slider("Max Depth", 1, 20, 5, key="dt_max_depth")
        if selected_model in ["Random Forest", "XGBoost"]:
            params["n_estimators"] = st.slider("Number of Estimators", 10, 500, 100, key=f"{selected_model}_n_estimators")
        if selected_model in ["Decision Tree", "Random Forest"]:
            params["max_depth"] = st.slider("Max Depth", 1, 20, 5, key=f"{selected_model}_max_depth")
        if selected_model == "XGBoost":
            params["learning_rate"] = st.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01, key="xgb_learning_rate")

if not selected_features:
    st.warning("⚠️ Please select at least one feature")
    st.stop()

if st.button("🚀 Train Model", key="train_button"):
    with st.spinner("Training model..."):
        try:
            X, y, scaler, le = preprocess_data(df, selected_features, target_col)
            if X is None or y is None:
                raise ValueError("Preprocessing failed due to missing utils module.")
            model, X_train, X_test, y_train, y_test, mse, mae, r2 = train_and_evaluate_model(selected_model, X, y, params, test_size)
            if model is None:
                raise ValueError("Model training failed due to missing utils module.")
            st.session_state.trained_models[selected_model] = {
                'model': model,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'predictions': model.predict(X_test),
                'features': selected_features,
                'target': target_col,
                'scaler': scaler,
                'le': le,
                'problem_type': 'Regression'
            }
            st.success("✅ Model trained successfully!")
            
            st.subheader("📊 Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MSE", f"{mse:.4f}", delta_color="inverse")
            with col2:
                st.metric("MAE", f"{mae:.4f}", delta_color="inverse")
            with col3:
                st.metric("R² Score", f"{r2:.4f}")
            
            fig = px.scatter(x=y_test, y=st.session_state.trained_models[selected_model]['predictions'], 
                             labels={'x': 'Actual Life Expectancy', 'y': 'Predicted Life Expectancy'},
                             title='Actual vs Predicted Values')
            fig.add_shape(type="line", x0=min(y_test), y0=min(y_test), 
                          x1=max(y_test), y1=max(y_test), line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
            
            if hasattr(model, 'feature_importances_'):
                st.subheader("📊 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🔍 Feature importance not available for this model.")
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}. Check data, features, or utils module.")

# Make a Prediction
st.subheader("🔮 Make a Prediction")
if selected_model in st.session_state.trained_models:
    model_data = st.session_state.trained_models[selected_model]
    feature_inputs = {}
    cols = st.columns(min(3, len(model_data['features'])))
    for i, feature in enumerate(model_data['features']):
        with cols[i % len(cols)]:
            if feature in df.select_dtypes(exclude=[np.number]).columns.tolist():
                unique_values = df[feature].dropna().unique()
                feature_inputs[feature] = st.selectbox(feature, [""] + list(unique_values), key=f"input_{feature}")
            else:
                feature_inputs[feature] = st.number_input(feature, value=float(df[feature].mean()), key=f"input_{feature}")
    if st.button("Predict", key="predict_button"):
        if any(v == "" for v in feature_inputs.values()):
            st.error("❌ Please fill all input fields.")
        else:
            input_data = pd.DataFrame([feature_inputs])
            if any(col in df.select_dtypes(exclude=[np.number]).columns for col in input_data.columns):
                input_data = input_data.replace("", np.nan).dropna()
                if "Status" in input_data.columns and model_data['le']:
                    input_data["Status"] = model_data['le'].transform(input_data["Status"])
            input_data = model_data['scaler'].transform(input_data)
            try:
                prediction = model_data['model'].predict(input_data)
                st.markdown(f"**Predicted {model_data['target']}: {prediction[0]:.2f}**")
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
else:
    st.info("🔍 Train a model first to enable predictions.")