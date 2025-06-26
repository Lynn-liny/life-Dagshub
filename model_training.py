import pandas as pd
import numpy as np # Although not directly used in functions, often useful for data ops
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor # Make sure xgboost is installed: pip install xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow # Make sure mlflow is installed: pip install mlflow
import mlflow.sklearn

# --- Function: train_and_evaluate_model ---
def train_and_evaluate_model(model_name: str, X: pd.DataFrame, y: pd.Series, params: dict = None, test_size: float = 0.2):
    """
    Trains and evaluates a specified regression model, logs metrics and the model to MLflow.

    Args:
        model_name (str): The name of the model to train.
                          Supported options: "Linear Regression", "Decision Tree",
                          "Random Forest", "XGBoost".
        X (pd.DataFrame): The feature DataFrame.
        y (pd.Series): The target Series.
        params (dict, optional): A dictionary of hyperparameters for the model.
                                 Defaults to None, which uses default parameters or
                                 predefined basic parameters.
        test_size (float, optional): The proportion of the dataset to include in the
                                     test split. Defaults to 0.2 (20%).

    Returns:
        tuple: A tuple containing:
            - model (sklearn.base.BaseEstimator): The trained machine learning model.
            - X_train (pd.DataFrame): Training features.
            - X_test (pd.DataFrame): Testing features.
            - y_train (pd.Series): Training target.
            - y_test (pd.Series): Testing target.
            - mse (float): Mean Squared Error of the model's predictions.
            - mae (float): Mean Absolute Error of the model's predictions.
            - r2 (float): R-squared (coefficient of determination) of the predictions.
    """
    # Ensure params is a dictionary, even if None is passed
    if params is None:
        params = {}
    
    # 1. Data Splitting: Divide the dataset into training and testing sets.
    # This is crucial for evaluating the model's performance on unseen data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 2. Model Initialization: Select and initialize the regression model based on model_name.
    # Default parameters are set or overridden by the provided 'params' dictionary.
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        # Decision Tree Regressor: Uses max_depth to control tree complexity.
        model = DecisionTreeRegressor(max_depth=params.get("max_depth", 5), random_state=42)
    elif model_name == "Random Forest":
        # Random Forest Regressor: An ensemble method using multiple decision trees.
        # n_estimators is the number of trees, max_depth limits individual tree depth.
        model = RandomForestRegressor(n_estimators=params.get("n_estimators", 100),
                                      max_depth=params.get("max_depth", 5), random_state=42)
    elif model_name == "XGBoost": # Changed from 'else' to explicit check for clarity
        # XGBoost Regressor: A powerful gradient boosting algorithm.
        # objective specifies the learning task, n_estimators is number of boosting rounds,
        # learning_rate controls step size shrinkage.
        model = XGBRegressor(objective="reg:squarederror", # Standard objective for regression
                             n_estimators=params.get("n_estimators", 100),
                             learning_rate=params.get("learning_rate", 0.1),
                             random_state=42)
    else:
        # Handle unsupported model names
        raise ValueError(f"Unsupported model name: {model_name}. "
                         "Choose from 'Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost'.")

    # 3. Model Training: Fit the model to the training data.
    model.fit(X_train, y_train)
    
    # 4. Prediction: Make predictions on the test set.
    predictions = model.predict(X_test)
    
    # 5. Metric Calculation: Evaluate the model's performance using common regression metrics.
    mse = mean_squared_error(y_test, predictions) # Mean Squared Error
    mae = mean_absolute_error(y_test, predictions) # Mean Absolute Error
    r2 = r2_score(y_test, predictions) # R-squared
    
    # 6. MLflow Logging: Record experiment details, parameters, metrics, and the trained model.
    # Each call to train_and_evaluate_model will create a new MLflow run.
    with mlflow.start_run(run_name=f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("model_type", model_name) # Log the type of model
        
        # Log all provided hyperparameters
        for k, v in params.items():
            mlflow.log_param(k, v)
        
        # Log evaluation metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log the trained model. This allows for easy deployment and loading later.
        mlflow.sklearn.log_model(model, "model") # 'model' is the artifact path within the run
        
    return model, X_train, X_test, y_train, y_test, mse, mae, r2

# Example Usage (for testing purposes, not run when imported)
# if __name__ == "__main__":
#     # Create dummy data for demonstration
#     np.random.seed(42)
#     data = {
#         'feature1': np.random.rand(100),
#         'feature2': np.random.rand(100) * 10,
#         'target': 5 + 2 * np.random.rand(100) + 3 * np.random.rand(100) * 10 + np.random.randn(100)
#     }
#     dummy_df = pd.DataFrame(data)
#     
#     features = ['feature1', 'feature2']
#     target = 'target'
#     
#     # Dummy preprocessed X and y
#     X_dummy = dummy_df[features]
#     y_dummy = dummy_df[target]
#     
#     print("--- Training Linear Regression ---")
#     lr_model, _, _, _, _, lr_mse, lr_mae, lr_r2 = train_and_evaluate_model("Linear Regression", X_dummy, y_dummy)
#     print(f"Linear Regression - MSE: {lr_mse:.4f}, MAE: {lr_mae:.4f}, R2: {lr_r2:.4f}\n")
#     
#     print("--- Training Decision Tree ---")
#     dt_model, _, _, _, _, dt_mse, dt_mae, dt_r2 = train_and_evaluate_model("Decision Tree", X_dummy, y_dummy, params={"max_depth": 7})
#     print(f"Decision Tree - MSE: {dt_mse:.4f}, MAE: {dt_mae:.4f}, R2: {dt_r2:.4f}\n")
#     
#     print("--- Training Random Forest ---")
#     rf_model, _, _, _, _, rf_mse, rf_mae, rf_r2 = train_and_evaluate_model("Random Forest", X_dummy, y_dummy, params={"n_estimators": 50, "max_depth": 8})
#     print(f"Random Forest - MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}, R2: {rf_r2:.4f}\n")
#     
#     print("--- Training XGBoost ---")
#     xgb_model, _, _, _, _, xgb_mse, xgb_mae, xgb_r2 = train_and_evaluate_model("XGBoost", X_dummy, y_dummy, params={"n_estimators": 200, "learning_rate": 0.05})
#     print(f"XGBoost - MSE: {xgb_mse:.4f}, MAE: {xgb_mae:.4f}, R2: {xgb_r2:.4f}\n")
#     
#     print("MLflow runs should now be visible in your MLflow UI (e.g., by running 'mlflow ui' in your terminal).")

