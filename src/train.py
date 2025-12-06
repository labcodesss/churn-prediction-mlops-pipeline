# src/train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from src.preprocess import load_data, basic_clean
from src.features import build_features
from src.model_utils import save_model
import argparse

def main(data_path: str, model_path: str):
    df = load_data(data_path)
    df = basic_clean(df)
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

    save_model(clf, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to csv")
    parser.add_argument("--model-out", default="model/model.joblib", help="where to save model")
    args = parser.parse_args()
    main(args.data, args.model_out)

