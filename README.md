# NLP Sentiment Analyzer (BERT + PyTorch + Transformers)
[![CI](https://github.com/Mr00Zero7170/NLP-Sentiment-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/Mr00Zero7170/NLP-Sentiment-analyzer/actions/workflows/ci.yml)

End-to-end sentiment analysis project with:
- Transformer-based inference (`distilbert-base-uncased-finetuned-sst-2-english`)
- Python CLI for single and batch prediction
- Streamlit web app
- FastAPI inference service
- Offline evaluation script with standard classification metrics
- Unit/API tests and Docker support

## Why this project is useful

This is not only a UI demo. It covers key recruiter-relevant skills:
- NLP model inference with PyTorch + Transformers
- API design and validation with FastAPI
- Basic ML evaluation workflow with metrics
- Testing and containerization for reproducibility

## Project Structure

- `app.py` - CLI entry point
- `streamlit_app.py` - Streamlit web interface
- `api/main.py` - FastAPI app (`/health`, `/predict`)
- `src/sentiment_analyzer.py` - model loading + inference logic
- `scripts/evaluate.py` - evaluation on labeled CSV
- `data/sample_eval.csv` - sample labeled dataset
- `tests/` - unit and API tests
- `Dockerfile` - containerized API runtime

## Setup

```bash
cd /Users/krishnabisht/Documents/learningC/python_learning/NLP-Sentiment-analyzer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1) CLI Usage

Single text:

```bash
python app.py --text "I really love this project"
```

Batch from text file (one sentence per line):

```bash
python app.py --file test_corpus.txt
```

Use custom Hugging Face model:

```bash
python app.py --text "This is terrible" --model cardiffnlp/twitter-roberta-base-sentiment-latest
```

## 2) Streamlit Web App

```bash
python -m streamlit run streamlit_app.py
```

Open the local URL shown in terminal (typically `http://localhost:8501`).

## 3) FastAPI Service

Run API:

```bash
uvicorn api.main:app --reload
```

Open docs:
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this"}'
```

## 4) Evaluation

Run on sample labeled data:

```bash
python scripts/evaluate.py --data data/sample_eval.csv
```

Expected outputs include:
- Accuracy
- Confusion matrix
- Precision/recall/F1 report

## 5) Tests

```bash
pytest -q
```

## 6) Docker (API)

Build and run:

```bash
docker build -t nlp-sentiment-analyzer .
docker run -p 8000:8000 nlp-sentiment-analyzer
```

Then test:

```bash
curl http://127.0.0.1:8000/health
```
