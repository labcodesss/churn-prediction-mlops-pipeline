# src/preprocess.py
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # example simple cleaning
    df = df.dropna(subset=["customer_id", "churn"])  # ensure labels exist
    # convert boolean-like labels if needed
    df["churn"] = df["churn"].astype(int)
    return df

