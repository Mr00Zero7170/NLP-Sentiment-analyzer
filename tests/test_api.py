from fastapi.testclient import TestClient

import api.main as main

client = TestClient(main.app)


class StubAnalyzer:
    def predict(self, text: str) -> dict[str, str | float]:
        return {"text": text, "label": "POSITIVE", "score": 0.88}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_success(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_analyzer", lambda: StubAnalyzer())
    response = client.post("/predict", json={"text": "great"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "POSITIVE"
    assert payload["score"] == 0.88


def test_predict_validation_error() -> None:
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
