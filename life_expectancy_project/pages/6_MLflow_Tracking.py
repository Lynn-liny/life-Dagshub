import streamlit as st
import mlflow
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Check for MLflow availability
try:
    mlflow_available = True
except ImportError:
    mlflow_available = False
    st.warning("⚠️ MLflow is not installed. Experiment tracking is disabled. Install mlflow to proceed.")
    st.stop()

st.header("📋 MLflow Experiment Tracking")

# MLflow Configuration
st.subheader("⚙️ MLflow Configuration")
col1, col2 = st.columns(2)
with col1:
    tracking_uri = st.text_input("🔗 Tracking URI:", "http://localhost:5000", key="tracking_uri_input")
    experiment_name = st.text_input("🧪 Experiment Name:", "DS4E-LIFE-EXP", key="experiment_name_input")
with col2:
    if st.button("🔧 Set MLflow Configuration", key="set_config_button"):
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            st.session_state.mlflow_config = {"uri": tracking_uri, "experiment": experiment_name}
            st.success("✅ MLflow configuration set!")
        except Exception as e:
            st.error(f"❌ Error setting MLflow: {str(e)}. Check URI and permissions.")

# Log Models to MLflow
st.subheader("📊 Log Models to MLflow")
if st.session_state.trained_models:
    model_to_log = st.selectbox("Select model to log:", list(st.session_state.trained_models.keys()), key="model_to_log_select")
    if st.button("📤 Log Model", key="log_button"):
        try:
            if 'mlflow_config' not in st.session_state:
                raise ValueError("Please set MLflow configuration first.")
            with mlflow.start_run(run_name=f"{model_to_log}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                model_data = st.session_state.trained_models[model_to_log]
                model = model_data['model']
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_param("model_type", model_to_log)
                mlflow.log_param("features", ",".join(model_data['features']))
                mlflow.log_param("target", model_data['target'])
                if 'predictions' in model_data:
                    y_test = model_data['y_test']
                    predictions = model_data['predictions']
                    mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
                    mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))
                    mlflow.log_metric("r2", r2_score(y_test, predictions))
                st.success("✅ Model logged to MLflow!")
        except Exception as e:
            st.error(f"❌ Error logging model: {str(e)}. Check model data or MLflow setup.")
else:
    st.info("🔍 No trained models available. Train a model first (Classical ML or PyCaret).")

# Recent Experiment Runs
st.subheader("📈 Recent Experiment Runs")
if 'mlflow_config' in st.session_state:
    if st.button("🔄 Refresh Runs", key="refresh_button"):
        try:
            experiment = mlflow.get_experiment_by_name(st.session_state.mlflow_config["experiment"])
            if experiment:
                runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
                if not runs.empty:
                    st.dataframe(runs[['run_id', 'status', 'start_time', 'params.model_type', 
                                    'metrics.mse', 'metrics.r2']], use_container_width=True)
                else:
                    st.info("📊 No runs found in this experiment. Start logging some models!")
            else:
                st.warning("⚠️ Experiment not found. Configure and set up MLflow first.")
        except Exception as e:
            st.error(f"❌ Error fetching runs: {str(e)}. Verify MLflow configuration.")
else:
    st.info("🔍 Set MLflow configuration to view runs.")