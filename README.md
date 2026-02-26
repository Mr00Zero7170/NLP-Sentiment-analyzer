# NLP Sentiment Analyzer (BERT + PyTorch + Transformers)

A Python sentiment analyzer built with:
- BERT-based model from Hugging Face Transformers
- PyTorch inference backend
- Command-line interface for single and batch predictions

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Single text prediction:

```bash
python app.py --text "I really love this project"
```

Batch prediction from file (`input.txt` with one text per line):

```bash
python app.py --file input.txt
```

Use a custom model:

```bash
python app.py --text "This is terrible" --model cardiffnlp/twitter-roberta-base-sentiment-latest
```

## Project Structure

- `app.py` - CLI entry point
- `src/sentiment_analyzer.py` - BERT sentiment analyzer logic
- `requirements.txt` - dependencies
