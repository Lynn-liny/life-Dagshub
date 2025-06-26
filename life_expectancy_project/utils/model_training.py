import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

def train_and_evaluate_model(model_name, X, y, params=None, test_size=0.2):
    if params is None:
        params = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=params.get("max_depth", 5), random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=params.get("n_estimators", 100),
                                      max_depth=params.get("max_depth", 5), random_state=42)
    else:  # XGBoost
        model = XGBRegressor(objective="reg:squarederror",
                             n_estimators=params.get("n_estimators", 100),
                             learning_rate=params.get("learning_rate", 0.1),
                             random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    with mlflow.start_run(run_name=f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("model_type", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")
    return model, X_train, X_test, y_train, y_test, mse, mae, r2
