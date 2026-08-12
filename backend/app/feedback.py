"""
User-taught overrides: "this line IS hateful", "this one is NOT".

Why this exists
---------------
The shipped Bangla checkpoint misses most out-of-distribution slurs (38.8%
recall on the OOD fixture vs 91.4% in-distribution -- see
docs/MODEL_DEFECTS_AND_DETECTION.md). Retraining fixes that properly, but a
retrain is an offline, GPU-bound, hours-long operation. This store is the
online half: the user corrects a miss once, and every later occurrence of
that text is flagged instantly, on every page, without touching the model.

Everything written here also accumulates into a JSONL file that is directly
usable as labelled training data for the next retrain -- so corrections are
not throwaway UI state, they are a growing, human-verified dataset that
targets exactly the distribution the model is currently weakest on.

Three-tier matching, cheapest and safest first
----------------------------------------------
1. EXACT, on the *normalized* text. The important one: an O(1) dict hit that
   happens BEFORE tokenization, so a taught line costs zero model inference.
   Keying on the normalized form (not the raw form) means "৭। খাঙ্কিরপোলা",
   "খাঙ্কিরপোলা" and "21) খাঙ্কিরপোলা" all collapse to the same entry for
   free -- teach it once, in any of its numbered forms.

2. WHITESPACE-INSENSITIVE EXACT. Same dict lookup with all spaces removed, so
   "খাঙ্কির পোলা" hits an entry taught as "খাঙ্কিরপোলা". Bangla compounds get
   split and joined inconsistently by real writers, making this the single
   most common variant pattern -- and because it is still an exact string
   comparison, it generalizes with none of the risk of tier 3.

3. SIMILARITY, on the encoder's [CLS] embedding. Catches inflectional variants
   that survive neither exact tier (শুয়োরের বাচ্চা -> শুয়োরের বাচ্চারা). It
   reuses the forward pass the request was already going to do, so its only
   added cost is one (B, H) @ (H, N) matmul -- microseconds at realistic N.

   The threshold is calibrated against ood_eval_bn.csv rather than guessed,
   and is length-dependent, because short taught examples produce
   near-degenerate embeddings (see config.py for the measured numbers). A
   wrong generalization here silently mislabels unrelated text, which is much
   worse than making the user teach one extra variant, so both thresholds are
   set past the point where measured false links hit zero.

Nothing here blocks a request. Disk writes go to a daemon writer thread, and
the embedding index for entries restored from disk is warmed on a background
thread at startup -- until it finishes, exact matching already works.
"""
import json
import logging
import queue
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    FEEDBACK_MAX_ENTRIES,
    FEEDBACK_PATH,
    FEEDBACK_SIM_SHORT_CHARS,
    FEEDBACK_SIM_SHORT_THRESHOLD,
    FEEDBACK_SIM_THRESHOLD,
    LABELS,
)
from .preprocessing import normalize_for_model

logger = logging.getLogger("hate_sarcasm_api")

LABEL_IDS = {name: id_ for id_, name in LABELS.items()}

# NUL can't occur in the normalized text (the cleaners strip control chars),
# so it's a collision-free separator for the composite index key.
_KEY_SEP = "\x00"

_WHITESPACE_RE = re.compile(r"\s+")


def _index_key(language: str, normalized: str) -> str:
    return f"{language}{_KEY_SEP}{normalized}"


def _squash_key(language: str, normalized: str) -> str:
    """Index key with all whitespace removed -- see tier 2 in the module
    docstring."""
    return f"{language}{_KEY_SEP}{_WHITESPACE_RE.sub('', normalized)}"


@dataclass
class FeedbackEntry:
    language: str
    normalized: str
    label_id: int
    text: str  # as the user submitted it, kept verbatim for retraining
    created_at: str

    @property
    def key(self) -> str:
        return _index_key(self.language, self.normalized)

    @property
    def label(self) -> str:
        return LABELS[self.label_id]

    def to_record(self) -> dict:
        return {
            "text": self.text,
            "normalized": self.normalized,
            "language": self.language,
            "label": self.label,
            "label_id": self.label_id,
            "created_at": self.created_at,
        }


class FeedbackStore:
    def __init__(
        self,
        path: Path = FEEDBACK_PATH,
        max_entries: int = FEEDBACK_MAX_ENTRIES,
        sim_threshold: float = FEEDBACK_SIM_THRESHOLD,
        short_chars: int = FEEDBACK_SIM_SHORT_CHARS,
        short_threshold: float = FEEDBACK_SIM_SHORT_THRESHOLD,
    ):
        self._path = Path(path)
        self._max_entries = max_entries
        self._sim_threshold = sim_threshold
        self._short_chars = short_chars
        self._short_threshold = short_threshold

        # Ordered so eviction at the cap is oldest-first.
        self._entries: "OrderedDict[str, FeedbackEntry]" = OrderedDict()
        # Whitespace-stripped key -> index key (tier 2). Rebuilt wholesale on
        # mutation rather than maintained incrementally: two entries can share
        # a squashed key, so incremental deletion would need refcounting to
        # avoid dropping a still-live alias. Entries change at human rate, so
        # an O(N) rebuild is far cheaper than that bug.
        self._squashed: Dict[str, str] = {}
        # key -> L2-normalized [CLS] vector. Populated at add() time for new
        # entries, and by warm_index() for entries restored from disk.
        self._vectors: Dict[str, np.ndarray] = {}
        # language -> (keys, matrix, per-candidate thresholds). Invalidated on
        # any mutation and rebuilt from self._vectors on demand -- a pure
        # np.stack, no re-embedding, so invalidation is cheap.
        self._matrix_cache: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]] = {}

        # RLock because lookup paths call _matrix_for() while already held.
        self._lock = threading.RLock()

        self._write_queue: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None

    # ---- persistence ---------------------------------------------------

    def load(self) -> None:
        """
        Replay the append-only log. Later records for the same key win, and a
        `"deleted": true` record acts as a tombstone -- that is what makes an
        append-only file safe to also use as the delete mechanism, with no
        rewriting and no risk of a partial in-place edit corrupting history.
        """
        with self._lock:
            self._entries.clear()
            self._vectors.clear()
            self._squashed.clear()
            self._matrix_cache.clear()

            if not self._path.exists():
                logger.info("Feedback store: no file at %s (starting empty)", self._path)
                return

            replayed = 0
            with self._path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # A torn final line (process killed mid-append) must
                        # not make the whole store unloadable.
                        logger.warning(
                            "Feedback store: skipping malformed line %d in %s",
                            line_no, self._path,
                        )
                        continue

                    key = _index_key(record["language"], record["normalized"])
                    if record.get("deleted"):
                        self._entries.pop(key, None)
                    else:
                        self._entries[key] = FeedbackEntry(
                            language=record["language"],
                            normalized=record["normalized"],
                            label_id=int(record["label_id"]),
                            text=record.get("text", record["normalized"]),
                            created_at=record.get("created_at", ""),
                        )
                    replayed += 1

            self._evict_to_cap()
            self._reindex()
            logger.info(
                "Feedback store: %d live entries from %d records (%s)",
                len(self._entries), replayed, self._path,
            )

    def _writer_loop(self) -> None:
        while True:
            record = self._write_queue.get()
            if record is None:
                return
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:
                # A failed disk write must never take down the request path or
                # the thread; the in-memory override still works for this
                # process, it just won't survive a restart.
                logger.error("Feedback store: write failed: %s", exc)

    def _append_async(self, record: dict) -> None:
        with self._lock:
            if self._writer_thread is None:
                self._writer_thread = threading.Thread(
                    target=self._writer_loop, name="feedback-writer", daemon=True
                )
                self._writer_thread.start()
        self._write_queue.put(record)

    def flush(self, timeout: float = 5.0) -> None:
        """
        Block until queued writes have hit disk. For tests and shutdown only --
        the request path deliberately never waits for this.

        Polls rather than using Queue.join(), because join() requires the
        consumer to call task_done() for every item, and putting that
        bookkeeping in the writer loop buys nothing for a queue that the
        request path never waits on anyway.
        """
        waited = 0.0
        while waited < timeout:
            if self._write_queue.empty():
                # The item may have been dequeued but not yet written; give
                # the writer a moment to finish the open/write/close.
                time.sleep(0.02)
                if self._write_queue.empty():
                    return
            time.sleep(0.01)
            waited += 0.01

    # ---- mutation ------------------------------------------------------

    def add(
        self,
        text: str,
        language: str,
        label: str,
        vector: Optional[np.ndarray] = None,
    ) -> FeedbackEntry:
        """
        Teach one example. Raises ValueError on an unknown label or on text
        that normalizes to nothing (a bare list marker carries no lesson).
        """
        if label not in LABEL_IDS:
            raise ValueError(f"Unknown label '{label}'. Expected one of {sorted(LABEL_IDS)}")

        normalized = normalize_for_model(text, language)
        if not normalized:
            raise ValueError("text normalizes to nothing; there is no content to learn from")

        entry = FeedbackEntry(
            language=language,
            normalized=normalized,
            label_id=LABEL_IDS[label],
            text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            # Re-teaching the same text moves it to the most-recent position,
            # so the cap evicts genuinely stale entries rather than ones the
            # user keeps reaffirming.
            self._entries.pop(entry.key, None)
            self._entries[entry.key] = entry
            if vector is not None:
                self._vectors[entry.key] = _unit(vector)
            else:
                self._vectors.pop(entry.key, None)
            self._evict_to_cap()
            self._reindex()

        self._append_async(entry.to_record())
        return entry

    def remove(self, text: str, language: str) -> bool:
        normalized = normalize_for_model(text, language)
        key = _index_key(language, normalized)
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                # Fall back to the whitespace-insensitive alias, so a
                # correction can be undone by any spacing variant of the text
                # that triggered it -- the same way it was matched.
                aliased = self._squashed.get(_squash_key(language, normalized))
                if aliased:
                    entry = self._entries.pop(aliased, None)
            if entry is None:
                return False
            self._vectors.pop(entry.key, None)
            self._reindex()
            normalized = entry.normalized
        self._append_async({
            "normalized": normalized,
            "language": language,
            "deleted": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        return True

    def _evict_to_cap(self) -> None:
        while len(self._entries) > self._max_entries:
            key, _ = self._entries.popitem(last=False)
            self._vectors.pop(key, None)
            self._matrix_cache.clear()

    def _reindex(self) -> None:
        """Rebuild the whitespace-insensitive alias map and drop the stacked
        matrices. Call under the lock after any change to _entries. Insertion
        order means the most recently taught entry wins a squash collision."""
        self._squashed = {
            _squash_key(entry.language, entry.normalized): entry.key
            for entry in self._entries.values()
        }
        self._matrix_cache.clear()

    # ---- lookup --------------------------------------------------------

    def lookup_exact(self, language: str, normalized: str) -> Optional[FeedbackEntry]:
        """
        Tiers 1 and 2: exact on the normalized text, then exact ignoring
        whitespace. Both are dict hits, so this is safe to call before
        tokenization on every request.
        """
        with self._lock:
            entry = self._entries.get(_index_key(language, normalized))
            if entry is not None:
                return entry
            aliased = self._squashed.get(_squash_key(language, normalized))
            return self._entries.get(aliased) if aliased else None

    def lookup_similar(
        self, language: str, vectors: np.ndarray
    ) -> List[Optional[Tuple[FeedbackEntry, float]]]:
        """
        Tier 3. Nearest taught example per row of `vectors` (raw [CLS] outputs,
        shape (B, H)), or None where nothing clears that candidate's threshold.
        """
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        empty: List[Optional[Tuple[FeedbackEntry, float]]] = [None] * len(vectors)

        with self._lock:
            index = self._matrix_for(language)
            if index is None:
                return empty
            keys, matrix, thresholds = index

            sims = _unit(vectors) @ matrix.T  # (B, N), both sides unit-norm

            # Each candidate carries its own bar (short taught examples need a
            # higher one), so the winner must be the best among candidates
            # that actually clear *their* bar -- not the global argmax, which
            # could be a short candidate sitting below its own threshold and
            # would then mask a longer, qualifying one behind it.
            qualifying = np.where(sims >= thresholds[None, :], sims, -np.inf)
            best = qualifying.argmax(axis=1)
            best_sim = qualifying[np.arange(len(vectors)), best]

            out: List[Optional[Tuple[FeedbackEntry, float]]] = []
            for row in range(len(vectors)):
                score = float(best_sim[row])
                if not np.isfinite(score):
                    out.append(None)
                    continue
                entry = self._entries.get(keys[int(best[row])])
                out.append((entry, score) if entry is not None else None)
            return out

    def _matrix_for(
        self, language: str
    ) -> Optional[Tuple[List[str], np.ndarray, np.ndarray]]:
        cached = self._matrix_cache.get(language)
        if cached is not None:
            return cached
        keys = [k for k in self._entries if k.startswith(f"{language}{_KEY_SEP}") and k in self._vectors]
        if not keys:
            return None
        matrix = np.stack([self._vectors[k] for k in keys])
        thresholds = np.array(
            [
                self._short_threshold
                if len(self._entries[k].normalized) < self._short_chars
                else self._sim_threshold
                for k in keys
            ],
            dtype=np.float32,
        )
        self._matrix_cache[language] = (keys, matrix, thresholds)
        return keys, matrix, thresholds

    # ---- embedding index -----------------------------------------------

    def keys_needing_vectors(self) -> List[Tuple[str, str, str]]:
        """(key, language, normalized) for entries with no cached vector."""
        with self._lock:
            return [
                (entry.key, entry.language, entry.normalized)
                for entry in self._entries.values()
                if entry.key not in self._vectors
            ]

    def set_vectors(self, pairs: List[Tuple[str, np.ndarray]]) -> None:
        with self._lock:
            for key, vector in pairs:
                # Skip keys deleted while the warm-up batch was in flight.
                if key in self._entries:
                    self._vectors[key] = _unit(vector)
            self._matrix_cache.clear()  # entries unchanged, so no full _reindex

    def warm_index(self, embed: Callable[[str, List[str]], np.ndarray], batch_size: int = 32) -> None:
        """
        Embed every entry restored from disk. Meant to run on a background
        thread: exact matching is already live before this completes, so a
        slow warm-up degrades generalization temporarily, never correctness.
        """
        pending = self.keys_needing_vectors()
        if not pending:
            return

        by_language: Dict[str, List[Tuple[str, str]]] = {}
        for key, language, normalized in pending:
            by_language.setdefault(language, []).append((key, normalized))

        for language, items in by_language.items():
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                try:
                    vectors = embed(language, [text for _, text in chunk])
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Feedback store: warm-up embed failed for '%s': %s", language, exc)
                    break
                self.set_vectors([(key, vectors[i]) for i, (key, _) in enumerate(chunk)])

        logger.info("Feedback store: embedding index warmed (%d entries)", len(pending))

    # ---- introspection --------------------------------------------------

    def all_entries(self) -> List[FeedbackEntry]:
        with self._lock:
            return list(self._entries.values())

    def count(self) -> int:
        with self._lock:
            return len(self._entries)


def _unit(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize so a dot product is exactly cosine similarity. The epsilon
    floor keeps an all-zero vector (impossible in practice, but cheap to
    guard) from producing NaNs that would poison every later comparison."""
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


store = FeedbackStore()
