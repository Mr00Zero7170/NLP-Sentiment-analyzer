from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF + Logistic Regression baseline against DistilBERT"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sample_eval.csv",
        help="CSV path with columns: text,label",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test split ratio",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    from src.sentiment_analyzer import BertSentimentAnalyzer

    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must include columns: text,label")

    df["label"] = df["label"].str.upper().str.strip()
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    baseline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=200, random_state=args.random_state)),
        ]
    )
    baseline.fit(x_train, y_train)
    baseline_preds = baseline.predict(x_test)
    baseline_acc = accuracy_score(y_test, baseline_preds)

    analyzer = BertSentimentAnalyzer(model_name="distilbert-base-uncased-finetuned-sst-2-english")
    bert_preds = analyzer.predict_batch(x_test.tolist())
    bert_labels = [row["label"].upper() for row in bert_preds]
    bert_acc = accuracy_score(y_test.tolist(), bert_labels)

    print(f"Samples: {len(df)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Logistic Regression (TF-IDF) accuracy: {baseline_acc:.4f}")
    print(f"DistilBERT accuracy: {bert_acc:.4f}")


if __name__ == "__main__":
    main()
