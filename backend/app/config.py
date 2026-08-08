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
