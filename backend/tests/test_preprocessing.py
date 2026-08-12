"""
Covers the inference-time normalization added to close the train/serve gap,
and the new probability fields the extension scales its blur intensity from.
"""
import pytest

from app.preprocessing import (
    normalize_for_model,
    preprocess_bangla,
    preprocess_english,
    strip_list_prefix,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Bangla digits + danda, the dominant form on the reported page.
        ("১। শুয়োরের বাচ্চা", "শুয়োরের বাচ্চা"),
        # No space after the separator.
        ("২১.হাউয়ার পো", "হাউয়ার পো"),
        # ASCII digits with a Bangla danda -- real pages mix them freely.
        ("58। বাপের আগে বাল ফালায়", "বাপের আগে বাল ফালায়"),
        ("51. গোলাম আযম", "গোলাম আযম"),
        # Decimal-style marker: only the marker goes, the content number stays.
        ("১০৯.১ ইঞ্চি", "১ ইঞ্চি"),
        ("১০০. ইউনুস (সুদখোর)", "ইউনুস (সুদখোর)"),
    ],
)
def test_strip_list_prefix_removes_page_numbering(raw, expected):
    assert strip_list_prefix(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        # No separator -> the number is part of the sentence, not a marker.
        "১০ জন মানুষ",
        "2024 was a rough year",
        # Nothing but a marker: stripping would empty the line, so it's kept.
        "৭।",
        "12.",
    ],
)
def test_strip_list_prefix_leaves_real_content_alone(raw):
    assert strip_list_prefix(raw) == raw


def test_bangla_cleaning_matches_training_pipeline():
    # Emoji dropped, NFC-normalized, whitespace collapsed -- as Phase 2 did.
    assert preprocess_bangla("তর ভোদায়  বিলাই 😀 হ্যাচড়া") == "তর ভোদায় বিলাই হ্যাচড়া"


def test_english_cleaning_matches_training_pipeline():
    assert preprocess_english("RT @bob Check #this http://x.co OUT!!") == "check this out!!"


def test_normalize_strips_prefix_then_cleans():
    assert normalize_for_model("৪৫. আল বদর 🔥", "bn") == "আল বদর"


def test_normalize_is_a_passthrough_for_unknown_language():
    # Language validation belongs to the endpoint; the cleaner must not raise.
    assert normalize_for_model("  a   b  ", "fr") == "a b"


# ---- endpoint-level -----------------------------------------------------


def test_predict_returns_full_probability_distribution(client):
    resp = client.post("/predict", json={"text": "Have a nice day!", "language": "en"})
    body = resp.json()

    total = body["non_hateful_prob"] + body["hate_prob"] + body["sarc_prob"]
    assert total == pytest.approx(1.0, abs=1e-5)
    assert body["confidence"] == pytest.approx(
        max(body["non_hateful_prob"], body["hate_prob"], body["sarc_prob"])
    )


def test_line_that_normalizes_to_nothing_scores_non_hateful(client):
    # A bare list marker is not an error -- the extension sends one result per
    # queued line and must get exactly one back.
    resp = client.post("/predict", json={"text": "৭।", "language": "bn"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "non_hateful"


def test_batch_mixes_scoreable_and_empty_lines_in_order(client):
    resp = client.post(
        "/predict/batch",
        json={
            "items": [
                {"id": "a", "text": "১। শুয়োরের বাচ্চা", "language": "bn"},
                {"id": "b", "text": "...", "language": "bn"},
                {"id": "c", "text": "আজকের আবহাওয়া খুব সুন্দর।", "language": "bn"},
            ]
        },
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["id"] for r in results] == ["a", "b", "c"]
    assert results[1]["label"] == "non_hateful"


def test_list_prefix_no_longer_suppresses_a_known_slur(client):
    """
    Regression guard for the measured prefix effect: 'খাঙ্কিরপোলা' alone
    scores hateful, but '৭। খাঙ্কিরপোলা' flipped to non_hateful before
    normalization existed. Both must now land on the same verdict.
    """
    bare = client.post("/predict", json={"text": "খাঙ্কিরপোলা", "language": "bn"}).json()
    prefixed = client.post("/predict", json={"text": "৭। খাঙ্কিরপোলা", "language": "bn"}).json()
    assert bare["label"] == prefixed["label"]
    assert bare["hate_prob"] == pytest.approx(prefixed["hate_prob"], abs=1e-6)
