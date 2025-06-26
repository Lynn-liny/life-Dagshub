import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

# --- Function: clean_missing ---
def clean_missing(df: pd.DataFrame, numeric_strategy="median"):
    """
    Handles missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame with potential missing values.
        numeric_strategy (str): Strategy for imputing numeric missing values.
                                 Options: "median", "mean", "most_frequent", "knn", or "drop".
                                 Note: "drop" will drop rows with NA in any numeric column.
                                 If "knn" is chosen, it performs KNN imputation.
                                 "most_frequent" for numeric behaves like mean/median depending on data.
                                 
    Returns:
        pd.DataFrame: A DataFrame with missing values handled.
    """
    df_clean = df.copy() # Work on a copy to avoid modifying the original DataFrame
    
    # Separate numeric and categorical columns
    num_cols = df_clean.select_dtypes(include="number").columns
    cat_cols = df_clean.select_dtypes(exclude="number").columns
    
    # Handle numeric missing values based on the chosen strategy
    if numeric_strategy == "knn":
        # KNN Imputer: Fills missing values using k-Nearest Neighbors approach.
        # It's generally more sophisticated than simple mean/median imputation.
        if len(num_cols) > 0:
            knn = KNNImputer(n_neighbors=5) # Using 5 neighbors for imputation
            df_clean[num_cols] = knn.fit_transform(df_clean[num_cols])
    elif numeric_strategy == "drop":
        # Drop rows with any missing values in numeric columns
        if len(num_cols) > 0:
            df_clean = df_clean.dropna(subset=num_cols).reset_index(drop=True)
    else:
        # Simple Imputer: Fills missing values using mean, median, or most_frequent strategy.
        # This covers "median", "mean", and "most_frequent" for numeric types.
        if len(num_cols) > 0:
            imp = SimpleImputer(strategy=numeric_strategy)
            df_clean[num_cols] = imp.fit_transform(df_clean[num_cols])
    
    # Handle categorical missing values
    # For categorical columns, "most_frequent" is a common strategy.
    if len(cat_cols) > 0:
        cat_imp = SimpleImputer(strategy="most_frequent")
        df_clean[cat_cols] = cat_imp.fit_transform(df_clean[cat_cols])
        
    return df_clean

# --- Function: preprocess_data ---
def preprocess_data(df: pd.DataFrame, features: list, target: str):
    """
    Performs full data preprocessing steps including missing value handling,
    label encoding for 'Status', dropping 'Country', and standardizing numeric features.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): A list of column names to be used as features (X).
        target (str): The name of the target column (y).
        
    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): Processed feature DataFrame.
            - y (pd.Series): Processed target Series.
            - scaler (StandardScaler): Fitted StandardScaler object.
            - le (LabelEncoder): Fitted LabelEncoder object (for 'Status' column).
    """
    # 1. Handle missing values using the clean_missing function
    # Defaulting to median imputation for numerical features.
    df_clean = clean_missing(df, numeric_strategy="median")
    
    # Initialize LabelEncoder for categorical encoding
    le = LabelEncoder()
    
    # 2. Label Encoding for 'Status' column if present
    # 'Status' (e.g., Developed, Developing) is often a categorical feature.
    if "Status" in df_clean.columns:
        df_clean["Status"] = le.fit_transform(df_clean["Status"])
    
    # 3. Drop 'Country' column if present
    # 'Country' is typically a high-cardinality categorical feature that
    # might not be useful directly or could lead to overfitting without
    # proper handling (e.g., one-hot encoding for too many categories).
    if "Country" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Country"])
    
    # Initialize StandardScaler for feature scaling
    scaler = StandardScaler()
    
    # 4. Standardize Numeric Features
    # Select numeric columns, excluding the target variable if it's numeric.
    num_cols = df_clean.select_dtypes(include="number").columns.drop(target, errors="ignore")
    
    # Apply StandardScaler to numeric features. This scales features to have
    # zero mean and unit variance, which is important for many ML algorithms.
    if len(num_cols) > 0: # Ensure there are numeric columns to scale
        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
    
    # 5. Separate features (X) and target (y)
    # Ensure that the features list only contains columns that still exist in df_clean
    X = df_clean[features]
    y = df_clean[target]
    
    return X, y, scaler, le # Return scaler and le for inverse transformation or future use
