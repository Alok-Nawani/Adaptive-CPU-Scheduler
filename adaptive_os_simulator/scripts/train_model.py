from __future__ import annotations

import argparse
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from backend.ml_model import FEATURE_NAMES, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train adaptive scheduler policy model from CSV")
    p.add_argument("--csv", required=True, help="Path to adaptive_scheduler_dataset.csv")
    p.add_argument("--out", required=True, help="Path to save model .joblib")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    # Expect dataset to have columns matching FEATURE_NAMES and target column 'policy'
    missing = [c for c in FEATURE_NAMES + ["policy"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    X = df[FEATURE_NAMES]
    y = df["policy"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=args.random_state, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    save_model(clf, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()


