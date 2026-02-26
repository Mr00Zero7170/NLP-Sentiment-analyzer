from src.sentiment_analyzer import BertSentimentAnalyzer


class DummyAnalyzer(BertSentimentAnalyzer):
    def __init__(self) -> None:
        pass

    def _infer(self, texts):
        out = []
        for t in texts:
            label = "POSITIVE" if "good" in t.lower() else "NEGATIVE"
            out.append(type("R", (), {"text": t, "label": label, "score": 0.99})())
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
