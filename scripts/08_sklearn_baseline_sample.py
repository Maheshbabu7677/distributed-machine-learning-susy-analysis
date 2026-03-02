

from pathlib import Path
import csv
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SAMPLE_PARQUET_DIR = "data/samples/susy_parquet_sample"
OUT_CSV = "data/processed/model_results.csv"
SEED = 42


def load_sample_parquet(parquet_dir: str) -> pd.DataFrame:
    p = Path(parquet_dir)
    if not p.exists():
        raise FileNotFoundError(f"Missing sample parquet directory: {parquet_dir}")

    df = pd.read_parquet(parquet_dir)
    return df


def main():
    df = load_sample_parquet(SAMPLE_PARQUET_DIR)

    feature_cols = [f"f{i}" for i in range(1, 19)]
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in sample parquet.")
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")

    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        solver="saga",
        max_iter=200,
        n_jobs=-1
    )

    start = datetime.now()
    model.fit(X_train, y_train)
    train_seconds = (datetime.now() - start).total_seconds()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    row = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "mode": "sklearn_sample",
        "model": "LogisticRegression_sklearn",
        "train_seconds": train_seconds,
        "accuracy": acc,
        "f1": f1,
        "weightedPrecision": prec,
        "weightedRecall": rec,
        "roc_auc": roc,
        "pr_auc": pr,
    }

    print("sklearn sample metrics:", row)

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_time", "mode", "model", "train_seconds",
                  "accuracy", "f1", "weightedPrecision", "weightedRecall", "roc_auc", "pr_auc"]

    write_header = not Path(OUT_CSV).exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"Saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
