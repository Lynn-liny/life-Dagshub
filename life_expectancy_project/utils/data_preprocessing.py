import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

def clean_missing(df: pd.DataFrame, numeric_strategy="median"):
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include="number").columns
    cat_cols = df_clean.select_dtypes(exclude="number").columns
    if numeric_strategy == "knn":
        knn = KNNImputer(n_neighbors=5)
        df_clean[num_cols] = knn.fit_transform(df_clean[num_cols])
    else:
        imp = SimpleImputer(strategy=numeric_strategy)
        df_clean[num_cols] = imp.fit_transform(df_clean[num_cols])
    if len(cat_cols):
        cat_imp = SimpleImputer(strategy="most_frequent")
        df_clean[cat_cols] = cat_imp.fit_transform(df_clean[cat_cols])
    return df_clean

def preprocess_data(df, features, target):
    df_clean = clean_missing(df, numeric_strategy="median")
    le = LabelEncoder()
    if "Status" in df_clean.columns:
        df_clean["Status"] = le.fit_transform(df_clean["Status"])
    if "Country" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Country"])
    scaler = StandardScaler()
    num_cols = df_clean.select_dtypes(include="number").columns.drop(target, errors="ignore")
    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
    X = df_clean[features]
    y = df_clean[target]
    return X, y, scaler, le