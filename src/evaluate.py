# src/evaluate.py
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.preprocess import load_data, basic_clean
from src.features import build_features
from src.model_utils import load_model

df = load_data("data/churn_sample.csv")
df = basic_clean(df)
X, y = build_features(df)

model = load_model("model/model.joblib")
y_prob = model.predict_proba(X)[:,1]
y_pred = (y_prob >= 0.3).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y, y_pred)),
    "precision": float(precision_score(y, y_pred, zero_division=0)),
    "recall": float(recall_score(y, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y, y_prob))
}

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics -> model/metrics.json")
print(metrics)
