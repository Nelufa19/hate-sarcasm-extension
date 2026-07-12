"""
Central place for paths and constants. Kept separate from schemas/registry so
that changing a checkpoint path or label mapping never requires touching
validation or inference logic.
"""
from pathlib import Path

# hate-sarcasm-extension/backend/app/config.py -> repo root is 4 parents up
REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE6_DIR = REPO_ROOT / "Phase 6 (Architectures)" / "distilbert-base (Lightweight model)"

MAX_LEN = 128  # must match MAX_LEN used at training time (tokenizer truncation)

CHECKPOINTS = {
    "en": {
        "backbone": "distilbert-base-uncased",
        "model_path": PHASE6_DIR / "Lightweight English" / "model.pt (2).zip",
        "tokenizer_path": PHASE6_DIR / "Lightweight English",
    },
    "bn": {
        "backbone": "distilbert-base-multilingual-cased",
        "model_path": PHASE6_DIR / "Lightweight Bangla" / "model.pt",
        "tokenizer_path": PHASE6_DIR / "Lightweight Bangla",
    },
}

LABELS = {0: "non_hateful", 1: "hateful", 2: "sarcastic"}

SUPPORTED_LANGUAGES = tuple(CHECKPOINTS.keys())
