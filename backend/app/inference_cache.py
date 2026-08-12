"""
Memoized model outputs, keyed on the normalized text.

Measured on the real page this project was debugged against
(allzinone.blogspot.com's 175-slur list -- 652 extracted lines, 181 Bangla /
471 English), a first load costs ~3.2 s of inference. A re-scan with a warm
cache costs ~3 ms: the extension re-scans on every MutationObserver fire, on
re-enable, and after every correction, so the same text is scored again and
again across a browsing session.

What is cached is the RAW MODEL OUTPUT, never a finished verdict. That
distinction is what keeps the cache from changing behaviour:

  * User corrections are consulted BEFORE the cache on every request, so a
    correction still overrides a cached model score immediately.
  * The [CLS] vector is cached alongside the probabilities, so the
    similarity tier still runs on a cache hit. Caching only the probabilities
    would silently disable feedback_similar for any previously-seen line --
    a real behaviour change disguised as an optimization.

Keyed on the normalized text, so "৭। খাঙ্কিরপোলা" and "খাঙ্কিরপোলা" share one
entry, exactly as the feedback store does.
"""
import threading
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np

from .config import INFERENCE_CACHE_MAX_ENTRIES

# probs (3,), hate_score, sarc_score, cls (hidden,)
CachedScore = Tuple[np.ndarray, float, float, np.ndarray]


class InferenceCache:
    """
    Plain LRU. Bounded by entry count rather than bytes: the [CLS] vector
    dominates each entry at hidden_size * 4 bytes (~3 KB for a 768-dim
    encoder), so the default cap is a few tens of MB at worst.
    """

    def __init__(self, max_entries: int = INFERENCE_CACHE_MAX_ENTRIES):
        self._max_entries = max_entries
        self._store: "OrderedDict[Tuple[str, str], CachedScore]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, language: str, normalized: str) -> Optional[CachedScore]:
        key = (language, normalized)
        with self._lock:
            value = self._store.get(key)
            if value is None:
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(self, language: str, normalized: str, value: CachedScore) -> None:
        key = (language, normalized)
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, float]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
            }


cache = InferenceCache()
