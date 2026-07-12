import pytest


def test_predict_english_returns_valid_shape(client):
    resp = client.post("/predict", json={"text": "Have a nice day!", "language": "en"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["label"] in {"non_hateful", "hateful", "sarcastic"}
    assert body["label_id"] in {0, 1, 2}
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["is_hateful_score"] <= 1.0
    assert 0.0 <= body["is_sarcastic_score"] <= 1.0
    assert body["language"] == "en"
    assert body["model_used"] == "distilbert-dualhead-en"
    assert body["latency_ms"] > 0


def test_predict_bangla_returns_valid_shape(client):
    resp = client.post(
        "/predict", json={"text": "তুমি খুব ভালো কাজ করেছ", "language": "bn"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "bn"
    assert body["model_used"] == "distilbert-dualhead-bn"


def test_predict_confident_sarcasm_is_detected(client):
    # A strongly sarcastic sentence should trip the sarcasm-leaning label or
    # at minimum score high on is_sarcastic_score -- this is a coarse smoke
    # test, not a model-accuracy benchmark (accuracy is measured in Phase 6).
    resp = client.post(
        "/predict",
        json={"text": "Oh sure, because that plan worked out SO well last time.", "language": "en"},
    )
    body = resp.json()
    assert body["is_sarcastic_score"] > 0.5


@pytest.mark.parametrize("bad_language", ["fr", "EN", "", "english"])
def test_predict_rejects_unsupported_language(client, bad_language):
    resp = client.post("/predict", json={"text": "hello", "language": bad_language})
    assert resp.status_code == 400


def test_predict_rejects_empty_text(client):
    resp = client.post("/predict", json={"text": "", "language": "en"})
    assert resp.status_code == 422


def test_predict_rejects_whitespace_only_text(client):
    resp = client.post("/predict", json={"text": "   ", "language": "en"})
    assert resp.status_code == 422


def test_predict_rejects_missing_fields(client):
    resp = client.post("/predict", json={"text": "hello"})
    assert resp.status_code == 422

    resp = client.post("/predict", json={"language": "en"})
    assert resp.status_code == 422


def test_predict_rejects_text_over_max_length(client):
    resp = client.post("/predict", json={"text": "a" * 2001, "language": "en"})
    assert resp.status_code == 422
