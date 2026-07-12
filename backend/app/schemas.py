"""
Pydantic models define the API's request/response contract. FastAPI uses
these to validate incoming JSON automatically (returning 422 on mismatch)
and to generate the /docs schema — no separate validation code needed.
"""
from pydantic import BaseModel, ConfigDict, Field

from .config import SUPPORTED_LANGUAGES


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Raw text to classify. Trimmed and validated non-empty.",
    )
    language: str = Field(
        ...,
        description=f"One of: {', '.join(SUPPORTED_LANGUAGES)}",
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


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


class ModelInfoResponse(BaseModel):
    available_languages: list[str]
    architecture: str
    backbones: dict[str, str]
