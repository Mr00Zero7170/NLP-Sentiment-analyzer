from src.sentiment_analyzer import BertSentimentAnalyzer


class DummyAnalyzer(BertSentimentAnalyzer):
    def __init__(self) -> None:
        pass

    def predict_detailed(self, text):
        label = "POSITIVE" if "good" in text.lower() else "NEGATIVE"
        score = 0.99
        return {
            "text": text,
            "label": label,
            "score": score,
            "probabilities": {
                "POSITIVE": score if label == "POSITIVE" else 0.01,
                "NEGATIVE": score if label == "NEGATIVE" else 0.01,
            },
        }

    def _infer(self, texts):
        out = []
        for text in texts:
            label = "POSITIVE" if "good" in text.lower() else "NEGATIVE"
            out.append(type("R", (), {"text": text, "label": label, "score": 0.99})())
        return out


def test_predict_single() -> None:
    analyzer = DummyAnalyzer()
    result = analyzer.predict("good product")
    assert result["label"] == "POSITIVE"
    assert result["score"] == 0.99


def test_predict_batch_filters_empty() -> None:
    analyzer = DummyAnalyzer()
    results = analyzer.predict_batch(["good", "", "bad"])
    assert len(results) == 2
    assert results[0]["label"] == "POSITIVE"
    assert results[1]["label"] == "NEGATIVE"
