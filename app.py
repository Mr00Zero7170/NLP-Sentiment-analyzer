import argparse
import json
from pathlib import Path

from src.sentiment_analyzer import BertSentimentAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT Sentiment Analyzer")
    parser.add_argument("--text", type=str, help="Single text input")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a .txt file (one text per line) for batch prediction",
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
    analyzer = BertSentimentAnalyzer(model_name=args.model)

    if args.text:
        print(json.dumps(analyzer.predict(args.text), indent=2))
        return

    if args.file:
        path = Path(args.file)
        lines = path.read_text(encoding="utf-8").splitlines()
        print(json.dumps(analyzer.predict_batch(lines), indent=2))
        return

    while True:
        user_input = input("Enter text (or 'q' to quit): ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            break
        print(json.dumps(analyzer.predict(user_input), indent=2))


if __name__ == "__main__":
    main()
