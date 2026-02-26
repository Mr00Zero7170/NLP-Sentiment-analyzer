from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class SentimentPrediction:
    text: str
    label: str
    score: float


class BertSentimentAnalyzer:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        max_length: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _infer(self, texts: List[str]) -> List[SentimentPrediction]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with self._torch.no_grad():
            outputs = self.model(**encoded)
            probs = self._torch.nn.functional.softmax(outputs.logits, dim=-1)
            scores, preds = self._torch.max(probs, dim=-1)

        id2label = self.model.config.id2label
        results: List[SentimentPrediction] = []
        for text, pred_id, score in zip(texts, preds.tolist(), scores.tolist()):
            results.append(
                SentimentPrediction(
                    text=text,
                    label=id2label.get(pred_id, str(pred_id)),
                    score=round(float(score), 4),
                )
            )
        return results

    def predict_detailed(self, text: str) -> Dict[str, str | float | Dict[str, float]]:
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with self._torch.no_grad():
            outputs = self.model(**encoded)
            probs = self._torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        id2label = self.model.config.id2label
        probabilities: Dict[str, float] = {}
        for idx, prob in enumerate(probs.tolist()):
            label = id2label.get(idx, str(idx))
            probabilities[label] = round(float(prob), 4)

        best_label = max(probabilities, key=probabilities.get)
        best_score = probabilities[best_label]
        return {
            "text": text,
            "label": best_label,
            "score": best_score,
            "probabilities": probabilities,
        }

    def predict(self, text: str) -> Dict[str, str | float]:
        result = self.predict_detailed(text)
        return {
            "text": str(result["text"]),
            "label": str(result["label"]),
            "score": float(result["score"]),
        }

    def predict_batch(self, texts: Iterable[str]) -> List[Dict[str, str | float]]:
        cleaned = [t.strip() for t in texts if t and t.strip()]
        if not cleaned:
            return []
        results = self._infer(cleaned)
        return [
            {
                "text": r.text,
                "label": r.label,
                "score": r.score,
            }
            for r in results
        ]
