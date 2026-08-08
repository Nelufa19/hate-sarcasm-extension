"""
FastAPI inference server for the Lightweight DistilBERT dual-head models
(English + Bangla). Serves /predict, /health, /model-info.

Run locally:
    uvicorn app.main:app --reload --port 8000
Then open http://localhost:8000/docs
"""
import logging
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .config import CHECKPOINTS, LABELS, SUPPORTED_LANGUAGES
from .logging_config import configure_logging
from .model_registry import registry
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)

configure_logging()
logger = logging.getLogger("hate_sarcasm_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: loading models")
    registry.load_all()
    logger.info("Startup complete: models=%s", registry.loaded_languages())
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
    return HealthResponse(status="ok", models_loaded=registry.loaded_languages())


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

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits, hate_logit, sarc_logit = model(
            inputs["input_ids"], inputs["attention_mask"]
        )
        probs = torch.softmax(logits, dim=-1)[0]
        label_id = int(torch.argmax(probs).item())

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "predict lang=%s label=%s confidence=%.4f latency_ms=%.2f id=%s",
        req.language,
        LABELS[label_id],
        probs[label_id].item(),
        latency_ms,
        req.id,
    )

    return PredictResponse(
        label=LABELS[label_id],
        label_id=label_id,
        is_hateful_score=torch.sigmoid(hate_logit)[0].item(),
        is_sarcastic_score=torch.sigmoid(sarc_logit)[0].item(),
        confidence=probs[label_id].item(),
        language=req.language,
        model_used=f"distilbert-dualhead-{req.language}",
        latency_ms=round(latency_ms, 2),
        id=req.id,
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
        texts.append(text)

    # index -> built response, filled in per language group, then reassembled
    # in original request order.
    results: list[PredictResponse] = [None] * len(items)  # type: ignore[list-item]

    languages_present = {item.language for item in items}
    for lang in languages_present:
        group_indices = [i for i, item in enumerate(items) if item.language == lang]

        try:
            model, tokenizer = registry.get(lang)
        except KeyError:
            raise HTTPException(status_code=503, detail=f"Model for '{lang}' is not loaded")

        group_texts = [texts[i] for i in group_indices]
        inputs = tokenizer(
            group_texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits, hate_logits, sarc_logits = model(
                inputs["input_ids"], inputs["attention_mask"]
            )
            probs = torch.softmax(logits, dim=-1)
            hate_scores = torch.sigmoid(hate_logits)
            sarc_scores = torch.sigmoid(sarc_logits)

        for pos, i in enumerate(group_indices):
            label_id = int(torch.argmax(probs[pos]).item())
            results[i] = PredictResponse(
                label=LABELS[label_id],
                label_id=label_id,
                is_hateful_score=hate_scores[pos].item(),
                is_sarcastic_score=sarc_scores[pos].item(),
                confidence=probs[pos][label_id].item(),
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
