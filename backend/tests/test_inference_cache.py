"""
The speed work (cache + dedup + length bucketing) is only acceptable if it
cannot change an answer. These tests pin the properties that make that true.
"""
import pytest

from app.inference_cache import cache


@pytest.fixture
def clean_cache():
    cache.clear()
    yield cache
    cache.clear()


BN = "তর বাপের হোগা দিয়া ট্রেন ভইরা দিমু।"


def test_cached_result_is_identical_to_the_uncached_one(client, clean_cache):
    first = client.post("/predict", json={"text": BN, "language": "bn"}).json()
    assert clean_cache.stats()["entries"] == 1
    second = client.post("/predict", json={"text": BN, "language": "bn"}).json()

    for field in ("label", "label_id", "non_hateful_prob", "hate_prob", "sarc_prob",
                  "is_hateful_score", "is_sarcastic_score", "confidence"):
        assert first[field] == second[field], field
    assert clean_cache.stats()["hits"] >= 1


def test_a_correction_still_overrides_an_already_cached_line(client, clean_feedback, clean_cache):
    """
    The ordering that makes the cache safe: corrections are consulted BEFORE
    the cache, so teaching something never has to wait for a cache eviction.
    """
    assert client.post("/predict", json={"text": BN, "language": "bn"}).json()["source"] == "model"

    client.post("/feedback", json={"text": BN, "language": "bn", "label": "hateful"})

    after = client.post("/predict", json={"text": BN, "language": "bn"}).json()
    assert after["source"] == "feedback_exact"
    assert after["label"] == "hateful"


def test_similarity_tier_still_fires_for_a_cached_line(client, clean_feedback, clean_cache):
    """
    The subtle one. Caching only probabilities would skip the forward pass and
    therefore skip the [CLS] vector, silently disabling feedback_similar for
    any line seen before. The vector is cached precisely to prevent that.
    """
    variant = "শুয়োরের বাচ্চারা"
    assert client.post("/predict", json={"text": variant, "language": "bn"}).json()["source"] == "model"

    client.post("/feedback", json={"text": "শুয়োরের বাচ্চা", "language": "bn", "label": "hateful"})

    after = client.post("/predict", json={"text": variant, "language": "bn"}).json()
    assert after["source"] == "feedback_similar", "cached line lost its similarity matching"
    assert after["label"] == "hateful"


def test_duplicates_in_one_batch_get_identical_results(client, clean_cache):
    """Dedup fans one score out to every copy -- verified bit-identical."""
    items = [{"text": BN, "language": "bn", "id": str(i)} for i in range(6)]
    items.append({"text": "আমি ভাত খাই", "language": "bn", "id": "other"})

    results = client.post("/predict/batch", json={"items": items}).json()["results"]
    dupes = [r for r in results if r["id"] != "other"]

    assert len({r["hate_prob"] for r in dupes}) == 1
    assert len({r["label"] for r in dupes}) == 1
    # One distinct text scored once, not six times.
    assert clean_cache.stats()["entries"] == 2


def test_batch_preserves_request_order_despite_length_bucketing(client, clean_cache):
    """
    Bucketing reorders texts internally to keep padding tight. Results must
    still come back matched to the caller's own ids, in the caller's order.
    """
    texts = [
        "আমি ভাত খাই",
        "আমাদের দেশের রাজনৈতিক পরিস্থিতি নিয়ে অনেক আলোচনা হচ্ছে এবং সাধারণ মানুষ "
        "প্রতিদিন নানা রকম সমস্যার মুখোমুখি হচ্ছে বিশেষ করে দ্রব্যমূল্যের ঊর্ধ্বগতির কারণে",
        "গোলাম আযম",
        BN,
        "hello there friend",
    ]
    langs = ["bn", "bn", "bn", "bn", "en"]
    items = [{"text": t, "language": l, "id": f"id{i}"} for i, (t, l) in enumerate(zip(texts, langs))]

    results = client.post("/predict/batch", json={"items": items}).json()["results"]
    assert [r["id"] for r in results] == [f"id{i}" for i in range(5)]
    assert [r["language"] for r in results] == langs

    # Each must equal what it scores alone -- bucketing changes padding width,
    # so this is the guard that padding never changes a decision.
    for item, batched in zip(items, results):
        single = client.post("/predict", json={"text": item["text"], "language": item["language"]}).json()
        assert single["label"] == batched["label"], item["text"][:30]
        assert single["hate_prob"] == pytest.approx(batched["hate_prob"], abs=1e-4)
        assert single["sarc_prob"] == pytest.approx(batched["sarc_prob"], abs=1e-4)
