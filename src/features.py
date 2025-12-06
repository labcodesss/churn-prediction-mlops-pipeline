# src/features.py
import numpy as np
import pandas as pd

NUMERIC_COLS = ["monthly_charges", "tenure_months", "total_charges"]  # example

def build_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    # keep only numeric example features for simplicity
    X = df[NUMERIC_COLS].fillna(0).astype(float)
    y = df["churn"]
    return X, y

