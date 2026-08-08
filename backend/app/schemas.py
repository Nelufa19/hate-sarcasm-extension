"""
Pydantic models define the API's request/response contract. FastAPI uses
these to validate incoming JSON automatically (returning 422 on mismatch)
and to generate the /docs schema — no separate validation code needed.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .config import SUPPORTED_LANGUAGES


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Raw text to classify. Trimmed and validated non-empty.",
    )
    language: str = Field(
        ...,
        description=f"One of: {', '.join(SUPPORTED_LANGUAGES)}",
    )
    id: Optional[str] = Field(
        default=None,
        description="Optional client-supplied identifier, echoed back in the "
        "response so a caller (e.g. the extension) can match results to DOM "
        "elements without relying on array order.",
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    label: str
    label_id: int
    is_hateful_score: float
    is_sarcastic_score: float
    confidence: float
    language: str
    model_used: str
    latency_ms: float
    id: Optional[str] = None


class PredictBatchRequest(BaseModel):
    items: list[PredictRequest] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Batch of texts to classify in one round trip. Capped at "
        "50 so a single request can't force the server to run an unbounded "
        "number of inferences.",
    )


class PredictBatchResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


class ModelInfoResponse(BaseModel):
    available_languages: list[str]
    architecture: str
    backbones: dict[str, str]
