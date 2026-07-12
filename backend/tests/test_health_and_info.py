def test_health_reports_both_models_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["models_loaded"]) == {"en", "bn"}


def test_model_info_lists_backbones(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available_languages"] == ["en", "bn"]
    assert body["backbones"]["en"] == "distilbert-base-uncased"
    assert body["backbones"]["bn"] == "distilbert-base-multilingual-cased"
