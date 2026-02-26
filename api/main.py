from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.sentiment_analyzer import BertSentimentAnalyzer

app = FastAPI(title="NLP Sentiment Analyzer API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze")


class PredictResponse(BaseModel):
    text: str
    label: str
    score: float


@lru_cache(maxsize=1)
def get_analyzer() -> BertSentimentAnalyzer:
    return BertSentimentAnalyzer()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict[str, str | float]:
    analyzer = get_analyzer()
    return analyzer.predict(request.text)
