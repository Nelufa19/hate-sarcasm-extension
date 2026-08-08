import pytest


def test_predict_batch_returns_results_for_each_item(client):
    resp = client.post(
        "/predict/batch",
        json={
            "items": [
                {"text": "Have a nice day!", "language": "en", "id": "a"},
                {"text": "তুমি খুব ভালো কাজ করেছ", "language": "bn", "id": "b"},
            ]
        },
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert results[0]["id"] == "a"
    assert results[0]["language"] == "en"
    assert results[1]["id"] == "b"
    assert results[1]["language"] == "bn"


def test_predict_batch_rejects_empty_items(client):
    resp = client.post("/predict/batch", json={"items": []})
    assert resp.status_code == 422


def test_predict_batch_rejects_over_50_items(client):
    items = [{"text": "hello", "language": "en"} for _ in range(51)]
    resp = client.post("/predict/batch", json={"items": items})
    assert resp.status_code == 422


def test_predict_batch_propagates_per_item_validation_error(client):
    # one bad language among otherwise-valid items should fail the whole
    # batch with the same 400 the single /predict endpoint would raise
    resp = client.post(
        "/predict/batch",
        json={
            "items": [
                {"text": "hello", "language": "en"},
                {"text": "hello", "language": "fr"},
            ]
        },
    )
    assert resp.status_code == 400


def test_predict_id_is_optional_and_defaults_to_none(client):
    resp = client.post("/predict", json={"text": "hello", "language": "en"})
    assert resp.status_code == 200
    assert resp.json()["id"] is None


def test_predict_batch_preserves_order_across_interleaved_languages(client):
    # en/bn/en/bn interleaved -- results must come back in request order even
    # though the batch endpoint groups items by language internally to run
    # one real forward pass per language.
    items = [
        {"text": "Have a nice day!", "language": "en", "id": "0"},
        {"text": "তুমি খুব ভালো কাজ করেছ", "language": "bn", "id": "1"},
        {"text": "Congratulations on the win!", "language": "en", "id": "2"},
        {"text": "আজকে আবহাওয়া চমৎকার", "language": "bn", "id": "3"},
    ]
    resp = client.post("/predict/batch", json={"items": items})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["id"] for r in results] == ["0", "1", "2", "3"]
    assert [r["language"] for r in results] == ["en", "bn", "en", "bn"]


def test_predict_batch_matches_single_predict_scores(client):
    # Batching groups same-language items into one padded forward pass --
    # padding must not change any individual item's scores versus calling
    # /predict on it alone.
    text = "Have a nice day!"
    single = client.post("/predict", json={"text": text, "language": "en"}).json()

    batch = client.post(
        "/predict/batch",
        json={
            "items": [
                {"text": text, "language": "en"},
                {"text": "A much longer sentence to force padding to differ.", "language": "en"},
            ]
        },
    ).json()["results"][0]

    assert batch["label"] == single["label"]
    assert batch["confidence"] == pytest.approx(single["confidence"], abs=1e-4)
    assert batch["is_hateful_score"] == pytest.approx(single["is_hateful_score"], abs=1e-4)
    assert batch["is_sarcastic_score"] == pytest.approx(single["is_sarcastic_score"], abs=1e-4)
