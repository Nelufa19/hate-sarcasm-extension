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

    # The fused 3-class softmax, exposed in full. These are the ONLY
    # trustworthy per-class scores this model produces, and they are what
    # the extension scales its blur intensity from. They sum to 1.0.
    non_hateful_prob: float = Field(..., description="P(non-hateful) from the fused 3-class head.")
    hate_prob: float = Field(..., description="P(hateful) from the fused 3-class head. Drives blur intensity.")
    sarc_prob: float = Field(..., description="P(sarcastic) from the fused 3-class head. Drives the yellow tint.")

    # DEPRECATED -- DO NOT USE FOR SCORING OR DISPLAY.
    #
    # These come from the model's auxiliary sigmoid heads, which are broken
    # in every currently-shipped checkpoint. The training loop in
    # "Phase 6 (Architectures)/distilbert-base (Lightweight model)/
    # lightweight-model.ipynb" reads:
    #
    #     loss = ALPHA*crit_b(hl, ls) + BETA*crit_b(sl, ls) + GAMMA*crit_3(...)
    #                          ^^ should be lh
    #
    # so the hate head was optimized against the SARCASM target and `lh`
    # (labels_hate) was never used. Measured over 3,000 held-out rows:
    # corr(hate_head, sarc_head) = 0.9997 (bn) / 0.9999 (en), while
    # corr(hate_head, y == hateful) is NEGATIVE (-0.36 bn / -0.48 en).
    #
    # In other words `is_hateful_score` is a second sarcasm score that
    # anti-correlates with actual hate. They are kept in the response only
    # so existing clients don't break on a missing key; once a checkpoint
    # trained with the corrected loss ships, repoint them at the real heads
    # and delete this comment.
    is_hateful_score: float = Field(
        ..., deprecated=True, description="BROKEN in current checkpoints -- see schemas.py. Use hate_prob."
    )
    is_sarcastic_score: float = Field(
        ..., deprecated=True, description="Unreliable in current checkpoints. Use sarc_prob."
    )

    confidence: float
    language: str
    model_used: str
    latency_ms: float
    id: Optional[str] = None

    # Where this verdict came from:
    #   "model"             -- the checkpoint decided it
    #   "feedback_exact"    -- a user-taught example with identical normalized text
    #   "feedback_similar"  -- a user-taught example within the cosine threshold
    # Defaulted so this stays a purely additive field for existing clients.
    source: str = "model"
    # For "feedback_similar" only: the cosine similarity to the taught example,
    # and that example's text -- so a surprising override is traceable to the
    # correction that caused it rather than looking like a model misfire.
    matched_text: Optional[str] = None
    match_similarity: Optional[float] = None


class FeedbackRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The text being corrected. Stored verbatim; matched on its normalized form.",
    )
    language: str = Field(..., description=f"One of: {', '.join(SUPPORTED_LANGUAGES)}")
    label: str = Field(
        ...,
        description="What this text actually is: 'hateful', 'sarcastic', or 'non_hateful'. "
        "'non_hateful' is how a false positive gets un-flagged.",
    )


class FeedbackEntryOut(BaseModel):
    text: str
    normalized: str
    language: str
    label: str
    label_id: int
    created_at: str


class FeedbackResponse(BaseModel):
    stored: bool
    entry: Optional[FeedbackEntryOut] = None
    total: int = Field(..., description="Live taught examples after this operation.")


class FeedbackListResponse(BaseModel):
    count: int
    entries: list[FeedbackEntryOut]


class FeedbackDeleteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    language: str


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
    feedback_entries: int = 0


class ModelInfoResponse(BaseModel):
    available_languages: list[str]
    architecture: str
    backbones: dict[str, str]
