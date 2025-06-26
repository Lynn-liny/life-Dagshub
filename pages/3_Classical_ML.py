import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Utility Module Imports ---
# This block attempts to import necessary functions from your utility files.
# 'data_preprocessing.py' handles cleaning and transforming the dataset.
# 'model_training.py' contains the logic for training and evaluating ML models.
# A fallback implementation is provided if these modules cannot be imported,
# disabling core functionality and displaying warnings/errors to the user.
try:
    from data_preprocessing import preprocess_data
    from model_training import train_and_evaluate_model
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    st.error("❌ Core utility modules ('data_preprocessing.py', 'model_training.py') not found or have errors. "
             "Please ensure they are in the correct directory and free of syntax issues. "
             "Classical ML features are currently disabled.")
    # Fallback functions to prevent script crashes if imports fail
    def preprocess_data(df, features, target):
        st.warning("Preprocessing function unavailable. Cannot proceed with training.")
        return None, None, None, None # Return Nones to indicate failure
    def train_and_evaluate_model(model_type, X, y, params, test_size):
        st.warning("Model training function unavailable. Cannot proceed with training.")
        return None, None, None, None, None, None, None, None # Return Nones to indicate failure

# --- Page Header ---
st.header("🤖 Classical Machine Learning Models")
st.markdown("---") # Visual separator

# --- Data Availability Check ---
# This check ensures that the main DataFrame ('df') has been loaded into
# Streamlit's session state by the 'streamlit_app.py' or '1_Data_Loading.py' page.
# If data is not found, a warning is displayed, and the page execution stops.
if st.session_state.df is None:
    st.warning("⚠️ Please load data from the **📊 Data Loading** page first to train models.")
    st.stop() # Stops execution of this page if data is missing

# Retrieve the DataFrame from session state for use in this page
df = st.session_state.df

# --- Model Configuration ---
# This section allows the user to configure the machine learning task,
# including selecting the target variable, features, model type, and test set size.
st.subheader("⚙️ Model Configuration")
col1, col2 = st.columns(2) # Uses two columns for a more organized layout

with col1:
    # Select the target variable for prediction.
    # Attempts to pre-select "Life expectancy" if it exists in the DataFrame.
    target_col = st.selectbox(
        "🎯 Select target variable:", 
        df.columns, 
        index=df.columns.get_loc("Life expectancy ") if "Life expectancy " in df.columns else 0, # Note the space in "Life expectancy "
        key="target_select"
    )
    
    # Filter out the target column from the list of available features.
    available_features = [col for col in df.columns if col != target_col]
    
    # Allow multi-selection of features. Default to a subset (excluding 'Country', first 5).
    selected_features = st.multiselect(
        "📊 Select features:", 
        available_features, 
        default=[col for col in available_features if col not in ["Country"]][:5], # Pre-select some common features
        key="features_select"
    )

with col2:
    # Select the machine learning model to train.
    selected_model = st.selectbox(
        "🤖 Select model:", 
        ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"], 
        key="model_select"
    )
    
    # Slider to determine the proportion of data to be used for testing.
    test_size = st.slider("🔄 Test set size:", 0.1, 0.5, 0.2, 0.05, key="test_size_slider")
    
    # Initialize an empty dictionary to store model-specific hyperparameters.
    params = {}
    
    # --- Hyperparameter Inputs (Dynamic based on selected model) ---
    # Conditional display of sliders for model-specific hyperparameters.
    if selected_model == "Decision Tree":
        params["max_depth"] = st.slider("Max Depth", 1, 20, 5, key="dt_max_depth")
    elif selected_model == "Random Forest":
        params["n_estimators"] = st.slider("Number of Estimators", 10, 500, 100, key="rf_n_estimators")
        params["max_depth"] = st.slider("Max Depth", 1, 20, 5, key="rf_max_depth")
    elif selected_model == "XGBoost":
        params["n_estimators"] = st.slider("Number of Estimators", 10, 500, 100, key="xgb_n_estimators")
        params["learning_rate"] = st.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01, key="xgb_learning_rate")

# --- Validation for Feature Selection ---
# Ensures that the user has selected at least one feature before proceeding.
if not selected_features:
    st.warning("⚠️ Please select at least one feature to train the model.")
    st.stop() # Stops execution if no features are selected

# --- Model Training Button ---
# When this button is clicked, the selected model is trained and evaluated.
if st.button("🚀 Train Model", key="train_button"):
    if not UTILS_AVAILABLE:
        st.error("Cannot train model: Utility functions are not available due to import errors.")
    else:
        with st.spinner("Training model... This might take a moment."): # Show a spinner during training
            try:
                # 1. Preprocess Data: Call the preprocess_data function from 'data_preprocessing.py'.
                # This function handles missing values, label encoding ('Status'), dropping 'Country',
                # and standardizing numeric features.
                X, y, scaler, le = preprocess_data(df, selected_features, target_col)
                
                # Check if preprocessing returned valid data (in case of fallback/errors)
                if X is None or y is None:
                    st.error("Preprocessing failed. Please check your data and utility functions.")
                    st.stop()
                
                # 2. Train and Evaluate Model: Call the train_and_evaluate_model function from 'model_training.py'.
                # This trains the chosen model, evaluates it, and logs the results to MLflow.
                model, X_train, X_test, y_train, y_test, mse, mae, r2 = \
                    train_and_evaluate_model(selected_model, X, y, params, test_size)
                
                # Check if model training returned a valid model (in case of fallback/errors)
                if model is None:
                    st.error("Model training failed. Please check your model_training.py and selected parameters.")
                    st.stop()

                # 3. Store Trained Model in Session State:
                # Store all relevant artifacts (model, test/train data, features, target, scaler, label encoder)
                # in session state so they can be accessed by other parts of the app (e.g., Explainability, Prediction).
                st.session_state.trained_models[selected_model] = {
                    'model': model,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'predictions': model.predict(X_test), # Store predictions for quick access
                    'features': selected_features,
                    'target': target_col,
                    'scaler': scaler, # Store the fitted scaler
                    'le': le,       # Store the fitted LabelEncoder
                    'problem_type': 'Regression' # Indicate the problem type
                }
                st.success(f"✅ Model '{selected_model}' trained successfully!")
                
                # --- Model Performance Display ---
                # Displays the calculated evaluation metrics.
                st.subheader("📊 Model Performance")
                col_metrics_1, col_metrics_2, col_metrics_3 = st.columns(3)
                with col_metrics_1:
                    st.metric("MSE", f"{mse:.4f}", delta_color="inverse") # Lower MSE is better
                with col_metrics_2:
                    st.metric("MAE", f"{mae:.4f}", delta_color="inverse") # Lower MAE is better
                with col_metrics_3:
                    st.metric("R² Score", f"{r2:.4f}") # Higher R2 is better (closer to 1)
                
                # --- Actual vs. Predicted Plot ---
                # Visualizes the model's performance by plotting actual vs. predicted values.
                # A diagonal red dashed line represents perfect predictions.
                fig = px.scatter(
                    x=y_test, 
                    y=st.session_state.trained_models[selected_model]['predictions'], 
                    labels={'x': f'Actual {target_col}', 'y': f'Predicted {target_col}'}, # Dynamic labels
                    title=f'Actual vs Predicted {target_col} ({selected_model})' # Dynamic title
                )
                fig.add_shape(type="line", x0=min(y_test), y0=min(y_test), 
                              x1=max(y_test), y1=max(y_test), line=dict(color="red", dash="dash"))
                st.plotly_chart(fig, use_container_width=True)
                
                # --- Feature Importance Display ---
                # Displays feature importance for models that support it (e.g., Tree-based models).
                if hasattr(model, 'feature_importances_'):
                    st.subheader("📊 Feature Importance")
                    # Create a DataFrame for importance, sort, and plot as a bar chart.
                    importance_df = pd.DataFrame({
                        'Feature': selected_features, # Use selected_features for display
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True) # Sort ascending for horizontal bar chart
                    
                    fig = px.bar(
                        importance_df, 
                        x='Importance', 
                        y='Feature', 
                        orientation='h', 
                        title=f'Feature Importance for {selected_model}',
                        labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("🔍 Feature importance not available for this model type (e.g., Linear Regression).")
            
            except Exception as e:
                # Catch any errors during the training process and display them.
                st.error(f"❌ An error occurred during model training: {str(e)}. "
                         "Please check your data, feature selections, or utility functions (`data_preprocessing.py`, `model_training.py`).")
                st.exception(e) # Show full traceback in debug mode for more details

# --- Make a Prediction Section ---
# Allows users to input values for features and get a real-time prediction
# from the currently selected and trained model.
st.subheader("🔮 Make a Prediction on New Data")
if selected_model in st.session_state.trained_models:
    # Retrieve the trained model and associated data from session state
    model_data = st.session_state.trained_models[selected_model]
    
    # Create input fields for each feature used in the trained model
    feature_inputs = {}
    
    # Organize input fields into columns for better layout (max 3 columns)
    cols = st.columns(min(3, len(model_data['features'])))
    
    for i, feature in enumerate(model_data['features']):
        with cols[i % len(cols)]: # Cycle through columns
            # Differentiate input widgets based on feature data type
            if feature in df.select_dtypes(exclude=[np.number]).columns.tolist():
                # For categorical features, use a selectbox with unique values
                unique_values = df[feature].dropna().unique().tolist() # Get unique values from original df
                feature_inputs[feature] = st.selectbox(
                    f"Select {feature}:", 
                    [""] + unique_values, # Add empty option for initial state
                    key=f"input_{feature}"
                )
            else:
                # For numerical features, use a number_input with mean as default value
                feature_inputs[feature] = st.number_input(
                    f"Enter {feature}:", 
                    value=float(df[feature].mean() if pd.api.types.is_numeric_dtype(df[feature]) else 0.0), 
                    key=f"input_{feature}"
                )
    
    if st.button("✨ Predict Life Expectancy", key="predict_button"):
        # Validate that all input fields have been filled
        if any(v == "" or v is None for v in feature_inputs.values()):
            st.error("❌ Please fill all input fields before predicting.")
            st.stop()
        
        # Create a DataFrame from the user inputs
        input_df = pd.DataFrame([feature_inputs])
        
        # --- Preprocessing for Prediction ---
        # This section mimics the preprocessing steps applied during training
        # to ensure the new input data is in the same format as the training data.
        
        # 1. Handle 'Country' column if it was present and dropped in training
        # Ensure 'Country' is removed from input if it was not used in features
        if "Country" in input_df.columns and "Country" not in model_data['features']:
            input_df = input_df.drop(columns=["Country"])

        # 2. Apply Label Encoding for 'Status' if it was encoded during training
        # IMPORTANT: Ensure 'Status' is processed BEFORE numerical scaling.
        if "Status" in input_df.columns and model_data['le'] is not None:
            try:
                # Transform 'Status' using the *fitted* LabelEncoder from training
                input_df["Status"] = model_data['le'].transform(input_df["Status"])
            except ValueError as ve:
                st.error(f"❌ Error encoding 'Status' for prediction: {str(ve)}. "
                         "Please ensure the selected 'Status' value is one seen during training.")
                st.stop()

        # 3. Scale Numerical Features: Use the *fitted* StandardScaler from training
        # to transform the numerical input features.
        num_cols_for_scaling = [col for col in model_data['features'] if col in input_df.columns and pd.api.types.is_numeric_dtype(input_df[col])]
        
        # Create a new DataFrame with only the features to be scaled
        scaled_input_features = input_df[num_cols_for_scaling]
        
        # Transform the features using the trained scaler
        input_df[num_cols_for_scaling] = model_data['scaler'].transform(scaled_input_features)

        try:
            # Make the prediction using the trained model
            prediction = model_data['model'].predict(input_df[model_data['features']]) # Ensure only trained features are passed
            st.markdown(f"**Predicted {model_data['target']}: <span style='color:#4ECDC4; font-size: 1.5em;'>{prediction[0]:.2f}</span>**", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}. "
                     "Ensure input features match the model's expectations and data types.")
            st.exception(e) # Show full traceback for debug
else:
    st.info("🔍 Train a model first using the '🚀 Train Model' button to enable predictions.")

# --- Debugging Confirmation ---
st.write("Debug: 3_Classical_ML.py page loaded and executed.")
