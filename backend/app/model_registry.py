"""
Loads each checkpoint exactly once and keeps it resident in memory for the
life of the server process. Without this, every /predict call would have to
re-download/re-instantiate the DistilBERT backbone and re-load ~500MB+ of
weights from disk — several seconds per request instead of milliseconds.
"""
import logging

import torch
from transformers import AutoTokenizer

from .architecture import LightweightDualHeadModel
from .config import CHECKPOINTS

logger = logging.getLogger("hate_sarcasm_api")


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, LightweightDualHeadModel] = {}
        self._tokenizers: dict[str, AutoTokenizer] = {}

    def load_all(self) -> None:
        for lang, cfg in CHECKPOINTS.items():
            model_path = cfg["model_path"]
            tokenizer_path = cfg["tokenizer_path"]

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint for language '{lang}' not found at {model_path}. "
                    "If this is the XLM-R checkpoint, it may still be an "
                    "un-pulled Git LFS pointer file."
                )

            logger.info("Loading tokenizer for '%s' from %s", lang, tokenizer_path)
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

            logger.info("Building %s backbone for '%s'", cfg["backbone"], lang)
            model = LightweightDualHeadModel(cfg["backbone"])

            logger.info("Loading weights for '%s' from %s", lang, model_path)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()

            self._models[lang] = model
            self._tokenizers[lang] = tokenizer
            logger.info("Model ready: '%s'", lang)

    def get(self, language: str) -> tuple[LightweightDualHeadModel, AutoTokenizer]:
        if language not in self._models:
            raise KeyError(language)
        return self._models[language], self._tokenizers[language]

    def loaded_languages(self) -> list[str]:
        return list(self._models.keys())


registry = ModelRegistry()
