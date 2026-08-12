"""
Central place for paths and constants. Kept separate from schemas/registry so
that changing a checkpoint path or label mapping never requires touching
validation or inference logic.
"""
import os
from pathlib import Path

# MODELS_DIR defaults to backend/models/ (self-contained, works inside a
# Docker image with no surrounding repo) but can be overridden via env var
# for local dev against the research repo's own Phase 6 checkpoints, e.g.:
#   MODELS_DIR="../../Phase 6 (Architectures)/distilbert-base (Lightweight model)"
_default_models_dir = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", _default_models_dir))

MAX_LEN = 128  # must match MAX_LEN used at training time (tokenizer truncation)

CHECKPOINTS = {
    "en": {
        "backbone": "distilbert-base-uncased",
        "model_path": MODELS_DIR / "Lightweight English" / "model.pt",
        "tokenizer_path": MODELS_DIR / "Lightweight English",
    },
    "bn": {
        "backbone": "distilbert-base-multilingual-cased",
        "model_path": MODELS_DIR / "Lightweight Bangla" / "model.pt",
        "tokenizer_path": MODELS_DIR / "Lightweight Bangla",
    },
}

LABELS = {0: "non_hateful", 1: "hateful", 2: "sarcastic"}

SUPPORTED_LANGUAGES = tuple(CHECKPOINTS.keys())

# ---- user-taught overrides (app/feedback.py) ------------------------------
# Append-only JSONL of human corrections. Doubles as labelled training data
# for the next retrain, so it lives outside models/ and is meant to be kept.
FEEDBACK_PATH = Path(
    os.environ.get("FEEDBACK_PATH", Path(__file__).resolve().parents[1] / "data" / "feedback.jsonl")
)

# Cosine similarity on the encoder's [CLS] vector above which an untaught line
# inherits a taught line's label. Calibrated, not guessed: cross-similarity of
# 36 benign Bangla lines against 49 taught slurs/threats from ood_eval_bn.csv
# gives these false-link rates (a benign line wrongly inheriting a slur label):
#
#     0.90 -> 41.7%     0.92 -> 25.0%     0.94 -> 0.0%
#     0.91 -> 36.1%     0.93 ->  2.8%     0.95 -> 0.0%
#
# The cliff between 0.92 and 0.94 is steep, so 0.95 sits just past it with
# margin. Exact matches ignore this entirely.
FEEDBACK_SIM_THRESHOLD = float(os.environ.get("FEEDBACK_SIM_THRESHOLD", "0.95"))

# Short taught examples need a much higher bar. Broken down by length, every
# near-miss above came from a SHORT taught side: benign text reaches 0.938
# against taught examples of <=10 chars ("বুশ" ~ "ধোন"), but only 0.80 against
# taught examples of >=16. That is the degenerate-embedding problem -- a
# one-word string's [CLS] vector is close to the generic "short Bangla token"
# vector, so cosine stops carrying much meaning.
#
# Real spelling variants of short slurs still clear 0.98 comfortably
# (শুয়োরের বাচ্চা ~ শুয়োরের বাচ্চারা = 0.9989, মাগীর ~ মাগির = 0.9967,
# হারামি ~ হারামী = 0.9956), so this costs almost no genuine generalization
# while removing the entire class of short-string false links.
FEEDBACK_SIM_SHORT_CHARS = int(os.environ.get("FEEDBACK_SIM_SHORT_CHARS", "16"))
FEEDBACK_SIM_SHORT_THRESHOLD = float(os.environ.get("FEEDBACK_SIM_SHORT_THRESHOLD", "0.98"))

# ---- inference cache (app/inference_cache.py) -----------------------------
# Memoized raw model outputs keyed on normalized text. Each entry holds the
# [CLS] vector (hidden_size floats) plus three probabilities, so ~3 KB per
# entry at hidden_size=768 -- 4000 entries is roughly 12 MB.
INFERENCE_CACHE_MAX_ENTRIES = int(os.environ.get("INFERENCE_CACHE_MAX_ENTRIES", "4000"))

# Ceiling on live entries; oldest are evicted first. At 5k the similarity
# matmul is still well under a millisecond, so this is a memory bound rather
# than a latency one. Evicted entries stay in the JSONL for retraining.
FEEDBACK_MAX_ENTRIES = int(os.environ.get("FEEDBACK_MAX_ENTRIES", "5000"))
