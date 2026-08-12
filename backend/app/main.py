"""
FastAPI inference server for the Lightweight DistilBERT dual-head models
(English + Bangla). Serves /predict, /health, /model-info.

Run locally:
    uvicorn app.main:app --reload --port 8000
Then open http://localhost:8000/docs
"""
import logging
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .config import CHECKPOINTS, LABELS, MAX_LEN, SUPPORTED_LANGUAGES
from .feedback import FeedbackEntry, store as feedback_store
from .inference_cache import cache as inference_cache
from .logging_config import configure_logging
from .model_registry import registry
from .preprocessing import normalize_for_model
from .schemas import (
    FeedbackDeleteRequest,
    FeedbackEntryOut,
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)

configure_logging()
logger = logging.getLogger("hate_sarcasm_api")

# Cap on how many distinct texts go through the model in one forward pass.
# Independent of the request's own 50-item limit: after dedup and cache hits a
# request may need far fewer, and a chunk that is too large hurts rather than
# helps, since batching gains flatten out past ~20 (measured 3.84 ms/line at 20
# vs 3.72 at 50) while peak memory keeps growing.
_MAX_FORWARD_BATCH = 32


def _embed(language: str, texts: list[str]) -> np.ndarray:
    """[CLS] vectors for already-normalized texts. Used by the feedback store
    to build its similarity index; the request path gets its vectors free
    from the prediction forward pass instead of calling this."""
    model, tokenizer = registry.get(language)
    inputs = tokenizer(
        texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        _, _, _, cls = model(inputs["input_ids"], inputs["attention_mask"], return_cls=True)
    return cls.numpy()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: loading models")
    registry.load_all()

    # Exact matching is live the moment load() returns. Embedding the restored
    # entries for similarity matching is deferred to a daemon thread so a large
    # accumulated feedback file can never delay the server accepting traffic --
    # a cold index degrades generalization for a few seconds, nothing more.
    feedback_store.load()
    threading.Thread(
        target=feedback_store.warm_index,
        args=(_embed,),
        name="feedback-warmup",
        daemon=True,
    ).start()

    logger.info(
        "Startup complete: models=%s feedback_entries=%d",
        registry.loaded_languages(),
        feedback_store.count(),
    )
    yield
    logger.info("Shutdown")


app = FastAPI(
    title="Hate & Sarcasm Detection API",
    description="Serves the capstone project's Lightweight DistilBERT dual-head models.",
    version="0.1.0",
    lifespan=lifespan,
)

class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    """
    Chrome (104+) sends a `Access-Control-Request-Private-Network: true`
    preflight header whenever the request's origin is "more public" than
    its target -- which is exactly our case: the extension's background
    service worker (chrome-extension://<id>) calling into localhost:8000.
    Without this middleware, that preflight gets no matching response
    header, Chrome silently fails the request, and the browser reports it
    to JS as a generic `TypeError: Failed to fetch` with no indication
    this is a PNA issue at all.

    CORSMiddleware (below) doesn't know about this header, so this runs
    as its own middleware to add the one line Chrome is actually checking
    for: https://developer.chrome.com/blog/private-network-access-preflight
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.headers.get("access-control-request-private-network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response


# Middleware order matters here. Starlette's add_middleware() prepends to
# the stack, so the LAST middleware added ends up OUTERMOST. CORSMiddleware
# must be added first (inner) so it stays innermost; PrivateNetworkAccessMiddleware
# added second (outer) then gets a chance to see and mutate the *final*
# response -- including CORSMiddleware's own short-circuited OPTIONS
# preflight response, which it sends directly without calling further
# inner middleware. If the order were reversed, our PNA header would never
# be added to preflight responses at all.

# Local dev: allow any origin. Before real deployment, restrict this to the
# extension's actual chrome-extension://<id> origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(PrivateNetworkAccessMiddleware)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=registry.loaded_languages(),
        feedback_entries=feedback_store.count(),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        available_languages=list(SUPPORTED_LANGUAGES),
        architecture="DistilBERT dual-head (shared encoder, hate + sarcasm auxiliary heads, fused 3-class output)",
        backbones={lang: cfg["backbone"] for lang, cfg in CHECKPOINTS.items()},
    )


def _run_inference(req: PredictRequest) -> PredictResponse:
    """
    Shared by /predict and /predict/batch so the two endpoints can never
    drift apart in behavior (validation, scoring, logging all happen once).
    """
    start = time.perf_counter()

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty or whitespace-only")

    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{req.language}'. Supported: {SUPPORTED_LANGUAGES}",
        )

    try:
        model, tokenizer = registry.get(req.language)
    except KeyError:
        raise HTTPException(status_code=503, detail=f"Model for '{req.language}' is not loaded")

    # Match training-time cleaning and drop page list-numbering. Validation
    # above deliberately runs against the RAW text: a line that is only a
    # list marker ("৭।") normalizes to something the model can't score, and
    # that should read as "nothing to flag", not as a 422 for the caller.
    model_text = normalize_for_model(text, req.language)
    if not model_text:
        return _non_hateful_response(req, start)

    # A previously-taught line short-circuits here, before tokenization: an
    # O(1) dict hit, zero model inference. This is what keeps the correction
    # mechanism from adding load to the backend as the store grows.
    taught = feedback_store.lookup_exact(req.language, model_text)
    if taught is not None:
        return _feedback_response(req, start, taught, 1.0, "feedback_exact")

    # Anything scored before in this session skips the forward pass entirely.
    # The cached [CLS] vector is what lets the similarity tier below still run
    # on a hit -- see inference_cache.py for why that matters.
    cached = inference_cache.get(req.language, model_text)
    if cached is None:
        inputs = tokenizer(
            model_text,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits, hate_logit, sarc_logit, cls = model(
                inputs["input_ids"], inputs["attention_mask"], return_cls=True
            )
            cached = (
                torch.softmax(logits, dim=-1)[0].numpy(),
                float(torch.sigmoid(hate_logit)[0].item()),
                float(torch.sigmoid(sarc_logit)[0].item()),
                cls[0].numpy(),
            )
        inference_cache.put(req.language, model_text, cached)

    probs, hate_score, sarc_score, cls_vec = cached
    label_id = int(probs.argmax())

    # Near-duplicate of something already taught (a spelling variant, extra
    # punctuation). Reuses the [CLS] vector, so the only added cost is one
    # small matmul.
    similar = feedback_store.lookup_similar(req.language, cls_vec)[0]
    if similar is not None:
        entry, similarity = similar
        return _feedback_response(req, start, entry, similarity, "feedback_similar")

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "predict lang=%s label=%s confidence=%.4f latency_ms=%.2f id=%s",
        req.language,
        LABELS[label_id],
        float(probs[label_id]),
        latency_ms,
        req.id,
    )

    return PredictResponse(
        label=LABELS[label_id],
        label_id=label_id,
        non_hateful_prob=float(probs[0]),
        hate_prob=float(probs[1]),
        sarc_prob=float(probs[2]),
        is_hateful_score=hate_score,
        is_sarcastic_score=sarc_score,
        confidence=float(probs[label_id]),
        language=req.language,
        model_used=f"distilbert-dualhead-{req.language}",
        latency_ms=round(latency_ms, 2),
        id=req.id,
    )


def _non_hateful_response(req: PredictRequest, start: float) -> PredictResponse:
    """
    Used when normalization leaves nothing scoreable (a line that was pure
    punctuation, emoji, or a bare list marker). Returning a definite
    non-hateful verdict rather than an error keeps the extension's batch
    handling uniform -- every item it sends gets exactly one result back.
    """
    return PredictResponse(
        label=LABELS[0],
        label_id=0,
        non_hateful_prob=1.0,
        hate_prob=0.0,
        sarc_prob=0.0,
        is_hateful_score=0.0,
        is_sarcastic_score=0.0,
        confidence=1.0,
        language=req.language,
        model_used=f"distilbert-dualhead-{req.language}",
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        id=req.id,
    )


def _feedback_response(
    req: PredictRequest,
    start: float,
    entry: FeedbackEntry,
    strength: float,
    source: str,
) -> PredictResponse:
    """
    Build a verdict from a user correction rather than the model.

    `strength` becomes the taught class's probability, which is what drives the
    extension's blur intensity. An exact match uses 1.0 -- the user asserted
    this outright, so it should render at full intensity. A similarity match
    uses the cosine score, so a borderline generalization visibly reads as
    less certain than the example it was derived from.
    """
    probs = [(1.0 - strength) / 2.0] * 3
    probs[entry.label_id] = strength

    return PredictResponse(
        label=entry.label,
        label_id=entry.label_id,
        non_hateful_prob=probs[0],
        hate_prob=probs[1],
        sarc_prob=probs[2],
        # The auxiliary heads are broken (see schemas.py) and a taught verdict
        # never consulted them; report them as uninformative rather than
        # inventing a number.
        is_hateful_score=0.0,
        is_sarcastic_score=0.0,
        confidence=strength,
        language=req.language,
        model_used=f"user-feedback-{req.language}",
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        id=req.id,
        source=source,
        matched_text=entry.text if source == "feedback_similar" else None,
        match_similarity=round(strength, 4) if source == "feedback_similar" else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    return _run_inference(req)


def _run_inference_batch(items: list[PredictRequest]) -> list[PredictResponse]:
    """
    One real batched forward pass per language, instead of N sequential
    single-item passes. Items are grouped by language (each language has its
    own model/tokenizer), tokenized together so padding is shared across the
    group, and run through the model once as a genuine batch -- CPU/GPU
    matrix multiplication scales far better across a batch dimension than
    across N separate Python-level calls, each paying its own fixed
    tokenization/dispatch overhead.

    Validation happens first, per item, so a single bad item in a batch
    still fails with the same status code _run_inference would raise for it
    alone -- grouping is purely an execution-order optimization, not a
    change in what counts as a valid request.
    """
    start = time.perf_counter()

    texts: list[str] = []
    for item in items:
        text = item.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="text must not be empty or whitespace-only")
        if item.language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{item.language}'. Supported: {SUPPORTED_LANGUAGES}",
            )
        # Same normalization as the single-item path -- see _run_inference.
        texts.append(normalize_for_model(text, item.language))

    # index -> built response, filled in per language group, then reassembled
    # in original request order.
    results: list[PredictResponse] = [None] * len(items)  # type: ignore[list-item]

    # Items whose text normalized away entirely are answered directly and
    # excluded from the tensor groups below -- tokenizing an empty batch
    # would otherwise produce a degenerate all-padding forward pass.
    scoreable = [i for i, t in enumerate(texts) if t]
    for i in range(len(items)):
        if not texts[i]:
            results[i] = _non_hateful_response(items[i], start)

    # Exact user corrections are resolved before tokenization and dropped from
    # the tensor groups, so a page full of already-taught lines costs no model
    # inference at all -- the batch shrinks instead of the server working
    # harder. Same short-circuit as the single-item path.
    still_scoreable = []
    for i in scoreable:
        taught = feedback_store.lookup_exact(items[i].language, texts[i])
        if taught is not None:
            results[i] = _feedback_response(items[i], start, taught, 1.0, "feedback_exact")
        else:
            still_scoreable.append(i)
    scoreable = still_scoreable

    # ---- three cost reductions, none of which change what the model sees ----
    #
    # Measured on the real 652-line page this was debugged against: 3169 ms ->
    # 1630 ms (1.94x), with 0 of 652 argmax decisions changed and 0 threshold
    # crossings at either 0.50 or 0.85.
    #
    # 1. DEDUPLICATE. Real pages repeat text (11.3% of that page's lines were
    #    duplicates). Scoring one copy and fanning the result out to the rest
    #    is exactly equivalent -- verified bit-identical, max |dP| = 0.0.
    #
    # 2. CACHE. Anything scored earlier in the session is answered without a
    #    forward pass at all.
    #
    # 3. BUCKET BY LENGTH. This is the big one, and the least obvious. A batch
    #    pads to its LONGEST member, so a single 106-token paragraph sitting in
    #    a batch of 20 nine-token list entries makes all 20 cost 106 tokens of
    #    compute. Sorting by token length before chunking keeps long lines
    #    together: a measured 434 ms -> 94 ms on exactly that shape.
    #
    # On (1) and (2) the outputs are bit-identical. (3) changes the padding
    # width, and float reductions run in a different order as a result, so it
    # perturbs probabilities by ~1e-6. That is not new behaviour -- scoring a
    # line alone versus in a batch already moves it ~8e-7 today -- and on the
    # real page the closest any line came to a decision boundary was 0.0211,
    # a safety factor of ~7,500x. No line was within 1e-4 of a threshold.

    # (lang, normalized) -> the request indices sharing that exact text.
    by_text: "dict[tuple[str, str], list[int]]" = {}
    for i in scoreable:
        by_text.setdefault((items[i].language, texts[i]), []).append(i)

    # Cache first, so anything already scored costs no forward pass.
    pending: "dict[str, list[tuple[str, str]]]" = {}
    scored: "dict[tuple[str, str], tuple]" = {}
    for key in by_text:
        lang, text = key
        hit = inference_cache.get(lang, text)
        if hit is not None:
            scored[key] = hit
        else:
            pending.setdefault(lang, []).append(key)

    for lang, keys in pending.items():
        try:
            model, tokenizer = registry.get(lang)
        except KeyError:
            raise HTTPException(status_code=503, detail=f"Model for '{lang}' is not loaded")

        # Sort by tokenized length so each chunk pads to something close to its
        # own members' length instead of to the page's longest line.
        keys.sort(key=lambda k: len(tokenizer(k[1])["input_ids"]))

        for chunk_start in range(0, len(keys), _MAX_FORWARD_BATCH):
            chunk = keys[chunk_start:chunk_start + _MAX_FORWARD_BATCH]
            inputs = tokenizer(
                [k[1] for k in chunk],
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                logits, hate_logits, sarc_logits, cls = model(
                    inputs["input_ids"], inputs["attention_mask"], return_cls=True
                )
                probs = torch.softmax(logits, dim=-1).numpy()
                hate_scores = torch.sigmoid(hate_logits).numpy()
                sarc_scores = torch.sigmoid(sarc_logits).numpy()
                cls_np = cls.numpy()

            for pos, key in enumerate(chunk):
                value = (probs[pos], float(hate_scores[pos]), float(sarc_scores[pos]), cls_np[pos])
                inference_cache.put(key[0], key[1], value)
                scored[key] = value

    # Similarity runs per language over the distinct texts, on one matmul --
    # including cache hits, whose [CLS] vector is cached precisely so this tier
    # keeps working. Skipping it for cached lines would silently disable
    # feedback_similar for anything seen before.
    languages_present = {lang for lang, _ in by_text}
    for lang in languages_present:
        keys = [k for k in by_text if k[0] == lang]
        if not keys:
            continue
        similar = feedback_store.lookup_similar(lang, np.stack([scored[k][3] for k in keys]))

        for pos, key in enumerate(keys):
            probs_row, hate_score, sarc_score, _ = scored[key]
            for i in by_text[key]:
                if similar[pos] is not None:
                    entry, similarity = similar[pos]
                    results[i] = _feedback_response(
                        items[i], start, entry, similarity, "feedback_similar"
                    )
                    continue

                label_id = int(probs_row.argmax())
                results[i] = PredictResponse(
                    label=LABELS[label_id],
                    label_id=label_id,
                    non_hateful_prob=float(probs_row[0]),
                    hate_prob=float(probs_row[1]),
                    sarc_prob=float(probs_row[2]),
                    is_hateful_score=hate_score,
                    is_sarcastic_score=sarc_score,
                    confidence=float(probs_row[label_id]),
                    language=lang,
                    model_used=f"distilbert-dualhead-{lang}",
                    latency_ms=0.0,  # filled in below, once total batch time is known
                    id=items[i].id,
                )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    for r in results:
        r.latency_ms = latency_ms

    logger.info(
        "predict_batch size=%d languages=%s latency_ms=%.2f",
        len(items),
        sorted(languages_present),
        latency_ms,
    )

    return results


@app.post("/predict/batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    return PredictBatchResponse(results=_run_inference_batch(req.items))


# ---- user corrections -----------------------------------------------------
# These take effect on the very next /predict call -- the in-memory index is
# updated before the response returns. There is no retrain, no reload, and no
# propagation delay. The JSONL these append to is separately usable as
# labelled training data for the next real retrain.


def _entry_out(entry: FeedbackEntry) -> FeedbackEntryOut:
    return FeedbackEntryOut(
        text=entry.text,
        normalized=entry.normalized,
        language=entry.language,
        label=entry.label,
        label_id=entry.label_id,
        created_at=entry.created_at,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def add_feedback(req: FeedbackRequest) -> FeedbackResponse:
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{req.language}'. Supported: {SUPPORTED_LANGUAGES}",
        )

    # Embed now, synchronously, so this example can generalize to variants
    # starting with the very next request. It's one forward pass on one short
    # string, on a human-triggered action -- and FastAPI runs this sync handler
    # in a threadpool, so it doesn't stall the event loop. If the model isn't
    # available the correction is still stored; it just matches exactly-only
    # until the next restart warms its vector.
    vector = None
    try:
        normalized = normalize_for_model(req.text, req.language)
        if normalized:
            vector = _embed(req.language, [normalized])[0]
    except KeyError:
        logger.warning("Feedback stored without vector: model '%s' not loaded", req.language)

    try:
        entry = feedback_store.add(req.text, req.language, req.label, vector=vector)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    logger.info(
        "feedback stored lang=%s label=%s total=%d text=%r",
        entry.language, entry.label, feedback_store.count(), entry.normalized[:80],
    )
    return FeedbackResponse(stored=True, entry=_entry_out(entry), total=feedback_store.count())


@app.get("/feedback", response_model=FeedbackListResponse)
def list_feedback() -> FeedbackListResponse:
    entries = feedback_store.all_entries()
    return FeedbackListResponse(count=len(entries), entries=[_entry_out(e) for e in entries])


@app.delete("/feedback", response_model=FeedbackResponse)
def delete_feedback(req: FeedbackDeleteRequest) -> FeedbackResponse:
    """Undo one correction. 404 if that text was never taught, so a client can
    tell 'removed it' apart from 'there was nothing to remove'."""
    if not feedback_store.remove(req.text, req.language):
        raise HTTPException(status_code=404, detail="No stored correction matches that text")
    return FeedbackResponse(stored=False, entry=None, total=feedback_store.count())
