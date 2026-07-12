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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import CHECKPOINTS, LABELS, SUPPORTED_LANGUAGES
from .logging_config import configure_logging
from .model_registry import registry
from .schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse

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

# Local dev: allow any origin. Before real deployment, restrict this to the
# extension's actual chrome-extension://<id> origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
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
        "predict lang=%s label=%s confidence=%.4f latency_ms=%.2f",
        req.language,
        LABELS[label_id],
        probs[label_id].item(),
        latency_ms,
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
    )
