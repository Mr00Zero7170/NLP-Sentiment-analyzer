from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.sentiment_analyzer import BertSentimentAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sentiment model on labeled CSV")
    parser.add_argument(
        "--data",
        type=str,
        default="data/sample_eval.csv",
        help="CSV path containing columns: text,label",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Hugging Face model name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must include columns: text,label")

    analyzer = BertSentimentAnalyzer(model_name=args.model)
    preds = analyzer.predict_batch(df["text"].tolist())

    y_true = df["label"].str.upper().tolist()
    y_pred = [row["label"].upper() for row in preds]

    acc = accuracy_score(y_true, y_pred)
    labels = sorted(list(set(y_true) | set(y_pred)))

    print(f"Samples: {len(df)}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels))


if __name__ == "__main__":
    main()
