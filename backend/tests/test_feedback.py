"""
Tests for user-taught overrides.

The behavioural contract these pin down:
  - a correction changes the next prediction, with no restart and no retrain
  - it survives page-level noise (list numbering) and spacing variants
  - it does NOT leak onto unrelated text
  - it costs no model inference on the exact path
  - it is durable, and undoable
"""
import json

import pytest

from app.config import FEEDBACK_PATH

# A line the shipped Bangla checkpoint gets wrong: a vulgar threat that lands
# in `sarcastic` rather than `hateful`. Exactly the case the correction
# mechanism exists for.
MISSED_THREAT = "তর বাপের হোগা দিয়া ট্রেন ভইরা দিমু।"
BENIGN = "আমি ভাত খাই"


def _predict(client, text, language="bn"):
    response = client.post("/predict", json={"text": text, "language": language})
    assert response.status_code == 200
    return response.json()


def test_correction_changes_the_next_prediction(client, clean_feedback):
    before = _predict(client, MISSED_THREAT)
    assert before["source"] == "model"

    stored = client.post(
        "/feedback",
        json={"text": MISSED_THREAT, "language": "bn", "label": "hateful"},
    )
    assert stored.status_code == 200
    assert stored.json()["total"] == 1

    after = _predict(client, MISSED_THREAT)
    assert after["label"] == "hateful"
    assert after["source"] == "feedback_exact"
    # Full intensity: the user asserted this outright.
    assert after["hate_prob"] == pytest.approx(1.0)
    assert after["non_hateful_prob"] + after["hate_prob"] + after["sarc_prob"] == pytest.approx(1.0)


def test_correction_applies_to_the_same_line_when_the_page_numbers_it(client, clean_feedback):
    """The store is keyed on normalized text, so teaching any one form of a
    numbered list entry teaches all of them."""
    client.post("/feedback", json={"text": MISSED_THREAT, "language": "bn", "label": "hateful"})

    for prefix in ["৭। ", "42. ", "১০৯) ", ""]:
        result = _predict(client, prefix + MISSED_THREAT)
        assert result["source"] == "feedback_exact", prefix
        assert result["label"] == "hateful", prefix


def test_correction_survives_spacing_variants(client, clean_feedback):
    """Tier 2: Bangla compounds get split and joined inconsistently by real
    writers, so whitespace must not defeat an exact match."""
    client.post("/feedback", json={"text": "খাঙ্কিরপোলা", "language": "bn", "label": "hateful"})

    for variant in ["খাঙ্কির পোলা", "খাঙ্কিরপোলা", "  খাঙ্কির  পোলা  "]:
        result = _predict(client, variant)
        assert result["source"] == "feedback_exact", variant
        assert result["label"] == "hateful", variant


def test_correction_generalizes_to_an_inflected_variant(client, clean_feedback):
    """Tier 3: an inflectional variant that survives neither exact tier still
    inherits the label via [CLS] similarity."""
    client.post("/feedback", json={"text": "শুয়োরের বাচ্চা", "language": "bn", "label": "hateful"})

    result = _predict(client, "শুয়োরের বাচ্চারা")
    assert result["source"] == "feedback_similar"
    assert result["label"] == "hateful"
    assert result["match_similarity"] >= 0.95
    assert result["matched_text"] == "শুয়োরের বাচ্চা"


def test_correction_does_not_leak_onto_unrelated_text(client, clean_feedback):
    """
    The failure mode that matters most. A store of taught slurs must not drag
    benign text -- especially bare proper nouns, where over-triggering on
    identity terms is the classic hate-speech failure -- into a hateful
    verdict. `গোলাম আযম` and `হিটলার` are the two the similarity index came
    closest to mislabelling during threshold calibration.
    """
    for slur in ["খাঙ্কিরপোলা", "শুয়োরের বাচ্চা", "তুই একটা হারামি", "ধোন"]:
        client.post("/feedback", json={"text": slur, "language": "bn", "label": "hateful"})

    for benign in [BENIGN, "আজকে আবহাওয়া খুব সুন্দর", "গোলাম আযম", "হিটলার", "বুশ", "জামাত"]:
        result = _predict(client, benign)
        assert result["source"] == "model", f"{benign} wrongly inherited a taught label"


def test_non_hateful_correction_clears_a_false_positive(client, clean_feedback):
    """The un-blur direction: marking something clean must override a model
    verdict just as strongly as marking something hateful."""
    text = "আজকে আবহাওয়া খুব সুন্দর"  # model calls this sarcastic (0.70)
    assert _predict(client, text)["label"] != "non_hateful"

    client.post("/feedback", json={"text": text, "language": "bn", "label": "non_hateful"})

    after = _predict(client, text)
    assert after["label"] == "non_hateful"
    assert after["source"] == "feedback_exact"
    assert after["non_hateful_prob"] == pytest.approx(1.0)


def test_exact_correction_skips_model_inference(client, clean_feedback):
    """
    The performance contract: a taught line is answered from a dict, before
    tokenization. Asserted via model_used rather than a wall-clock threshold,
    which would be flaky on a loaded machine.
    """
    client.post("/feedback", json={"text": MISSED_THREAT, "language": "bn", "label": "hateful"})

    result = _predict(client, MISSED_THREAT)
    assert result["model_used"] == "user-feedback-bn"
    assert result["is_hateful_score"] == 0.0  # the broken aux heads never ran


def test_batch_mixes_taught_and_untaught_items_in_order(client, clean_feedback):
    client.post("/feedback", json={"text": MISSED_THREAT, "language": "bn", "label": "hateful"})

    response = client.post(
        "/predict/batch",
        json={
            "items": [
                {"text": BENIGN, "language": "bn", "id": "a"},
                {"text": "৯। " + MISSED_THREAT, "language": "bn", "id": "b"},
                {"text": "hello there friend", "language": "en", "id": "c"},
                {"text": "৪২।", "language": "bn", "id": "d"},  # normalizes to nothing
            ]
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]

    assert [r["id"] for r in results] == ["a", "b", "c", "d"]
    assert results[1]["source"] == "feedback_exact"
    assert results[1]["label"] == "hateful"
    assert results[0]["source"] == "model"
    assert results[2]["source"] == "model"
    assert results[3]["label"] == "non_hateful"


def test_correction_is_persisted_as_retrainable_jsonl(client, clean_feedback):
    """Corrections are training data, not throwaway UI state -- the on-disk
    record has to carry everything a retrain needs."""
    client.post("/feedback", json={"text": MISSED_THREAT, "language": "bn", "label": "hateful"})
    clean_feedback.flush()

    lines = [json.loads(l) for l in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    record = [r for r in lines if r.get("text") == MISSED_THREAT][-1]
    assert record["label"] == "hateful"
    assert record["label_id"] == 1
    assert record["language"] == "bn"
    assert record["normalized"]
    assert record["created_at"]


def test_reteaching_overwrites_the_previous_label(client, clean_feedback):
    client.post("/feedback", json={"text": MISSED_THREAT, "language": "bn", "label": "sarcastic"})
    assert _predict(client, MISSED_THREAT)["label"] == "sarcastic"

    client.post("/feedback", json={"text": MISSED_THREAT, "language": "bn", "label": "hateful"})
    assert _predict(client, MISSED_THREAT)["label"] == "hateful"
    assert client.get("/feedback").json()["count"] == 1


def test_delete_reverts_to_the_model_verdict(client, clean_feedback):
    client.post("/feedback", json={"text": MISSED_THREAT, "language": "bn", "label": "hateful"})
    assert _predict(client, MISSED_THREAT)["source"] == "feedback_exact"

    removed = client.request(
        "DELETE", "/feedback", json={"text": MISSED_THREAT, "language": "bn"}
    )
    assert removed.status_code == 200
    assert _predict(client, MISSED_THREAT)["source"] == "model"

    # Second delete has nothing to remove, and says so.
    assert client.request(
        "DELETE", "/feedback", json={"text": MISSED_THREAT, "language": "bn"}
    ).status_code == 404


def test_corrections_are_scoped_per_language(client, clean_feedback):
    """`en` and `bn` have separate models and separate embedding spaces, so a
    correction taught for one must never be consulted for the other."""
    client.post("/feedback", json={"text": "this is fine", "language": "en", "label": "hateful"})

    assert _predict(client, "this is fine", language="en")["source"] == "feedback_exact"
    assert _predict(client, "this is fine", language="bn")["source"] == "model"


def test_listing_exposes_the_taught_set(client, clean_feedback):
    client.post("/feedback", json={"text": MISSED_THREAT, "language": "bn", "label": "hateful"})
    client.post("/feedback", json={"text": BENIGN, "language": "bn", "label": "non_hateful"})

    listing = client.get("/feedback").json()
    assert listing["count"] == 2
    assert {e["label"] for e in listing["entries"]} == {"hateful", "non_hateful"}
    assert client.get("/health").json()["feedback_entries"] == 2


def test_unteachable_input_is_rejected_not_stored(client, clean_feedback):
    """Text that normalizes to nothing (emoji only, here) carries no lesson,
    and storing an empty key would then match every line whose own
    normalization came out empty."""
    response = client.post("/feedback", json={"text": "😀🙂", "language": "bn", "label": "hateful"})
    assert response.status_code == 422
    assert client.get("/feedback").json()["count"] == 0


def test_unknown_label_is_rejected(client, clean_feedback):
    response = client.post(
        "/feedback", json={"text": BENIGN, "language": "bn", "label": "very_bad"}
    )
    assert response.status_code == 422


def test_unsupported_language_is_rejected(client, clean_feedback):
    response = client.post(
        "/feedback", json={"text": "hola", "language": "es", "label": "hateful"}
    )
    assert response.status_code == 400
