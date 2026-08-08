# Code Walkthrough — Understanding Every File, Line by Line

This doc teaches you what you actually built, why each piece exists, and how the pieces fit together. Read it in order — each section assumes you understood the one before it. After this, use `INTERVIEW_DRILL.md` to test yourself.

The code lives in `hate-sarcasm-extension/backend/`. Nine files, in the order data flows through them at request time:

```
HTTP request
    │
    ▼
main.py            ← receives the request, orchestrates everything
    │
    ├─→ schemas.py       (validates the request shape)
    ├─→ config.py         (where are the models, what languages exist)
    ├─→ model_registry.py (holds the already-loaded models)
    │       └─→ architecture.py (defines what a model even IS)
    └─→ logging_config.py (how we record what happened)
```

---

## 1. `config.py` — the single source of truth

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE6_DIR = REPO_ROOT / "Phase 6 (Architectures)" / "distilbert-base (Lightweight model)"

MAX_LEN = 128

CHECKPOINTS = {
    "en": {
        "backbone": "distilbert-base-uncased",
        "model_path": PHASE6_DIR / "Lightweight English" / "model.pt (2).zip",
        "tokenizer_path": PHASE6_DIR / "Lightweight English",
    },
    "bn": {
        "backbone": "distilbert-base-multilingual-cased",
        "model_path": PHASE6_DIR / "Lightweight Bangla" / "model.pt",
        "tokenizer_path": PHASE6_DIR / "Lightweight Bangla",
    },
}

LABELS = {0: "non_hateful", 1: "hateful", 2: "sarcastic"}
SUPPORTED_LANGUAGES = tuple(CHECKPOINTS.keys())
```

### Why this file exists at all

Every other file needs to answer questions like "where is the English model's weight file" or "what does class `2` mean." Without this file, those facts would be copy-pasted into `main.py`, `model_registry.py`, and the tests — and the day you move a checkpoint, you'd have to hunt through every file to update it. **One dictionary, one place, everything else reads from it.** This is the "single source of truth" principle, and it's the first thing an interviewer looks for when they ask you to review someone else's API code.

### `Path(__file__).resolve().parents[3]`

`__file__` is `config.py`'s own path. `.resolve()` makes it absolute (no `..` or symlinks). `.parents[3]` walks up 3 directories: `app/` → `backend/` → `hate-sarcasm-extension/` → repo root. This is how the server finds the `Phase 6 (Architectures)/` folder regardless of what directory you happen to run `uvicorn` from. **Hardcoding an absolute path like `/Users/faisal/...` would break the moment anyone else clones the repo** — this is why you compute it relative to the file's own location instead.

### Why `CHECKPOINTS` is a dict keyed by language code, not a list

Because at request time, the lookup is always "given `language="en"`, get me its config" — a dict gives you that in O(1) with a clean `KeyError` if the key's missing. A list would mean looping and comparing every time. This shape also directly determines `SUPPORTED_LANGUAGES = tuple(CHECKPOINTS.keys())` — the set of valid languages is *derived from* what's actually configured, not maintained separately. If you add a third language, you add one dict entry and `SUPPORTED_LANGUAGES` updates itself. That's not an accident — it's the same "single source of truth" idea applied one level deeper.

### Why `model.pt (2).zip` for English but `model.pt` for Bangla

This isn't a stylistic choice — it reflects what actually exists on disk. Your teammate's export process produced a `.zip` for the English checkpoint (still a valid `torch.save` archive internally — `torch.load()` reads it directly, no manual unzip needed) and a raw `.pt` for Bangla. The config just records the truth of what's there. This is worth remembering for an interview: **config should describe reality, not force reality into a clean pattern it doesn't have.**

---

## 2. `architecture.py` — defining what a "model" is

This is the most important file to understand deeply, and the easiest to get subtly wrong.

### The problem it solves

A `.pt` checkpoint file is **just numbers** — a dictionary mapping layer names (like `"hate_head.0.weight"`) to tensors of weights. Those numbers are meaningless without a Python class that says "layer `hate_head.0` is a `Linear(768, 128)`, and here's the order operations happen in." `architecture.py` is that class. It must be **structurally identical** to whatever class was used to train the checkpoint — not similar, not equivalent, identical — because `model.load_state_dict(state_dict)` matches saved tensors to the model's layers by name and shape.

### Why we copied it from the notebook instead of writing it fresh

```python
class LightweightDualHeadModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.hate_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

        self.sarcasm_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size + 128 + 128, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))
```

Three sub-networks:
- **`encoder`** — the pretrained DistilBERT backbone. Turns raw text into a 768-dimensional vector per token ("hidden states"). This is the part that already understands language in general, before this project's training ever happened.
- **`hate_head`** and **`sarcasm_head`** — small classifiers (`768 → 128 → 1`) that each try to answer one yes/no question from that shared representation. This is the "dual-head" idea from your capstone: instead of one classifier guessing 3 classes blindly, two specialist heads each learn one narrower signal.
- **`fusion_mlp`** — takes the shared representation *plus* both heads' intermediate opinions and combines them into the final 3-way decision (`non_hateful` / `hateful` / `sarcastic`).

### The part that will actually come up in an interview: `forward()`

```python
def forward(self, input_ids, attention_mask, token_type_ids=None):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])

    hate_hidden = F.relu(self.hate_head[0](cls_output))
    hate_hidden = self.dropout(hate_hidden)
    hate_logit  = self.hate_head[3](hate_hidden)

    sarc_hidden = F.relu(self.sarcasm_head[0](cls_output))
    sarc_hidden = self.dropout(sarc_hidden)
    sarc_logit  = self.sarcasm_head[3](sarc_hidden)

    fused = torch.cat([cls_output, hate_hidden, sarc_hidden], dim=1)
    logits_3class = self.fusion_mlp(fused)

    return logits_3class, hate_logit.squeeze(-1), sarc_logit.squeeze(-1)
```

Notice `hate_head` was defined as an `nn.Sequential` with 4 layers (`Linear, ReLU, Dropout, Linear` — indices 0,1,2,3), but `forward()` never calls `self.hate_head(cls_output)` directly. Instead it manually calls `self.hate_head[0](...)` (the first `Linear`), applies `F.relu` itself, applies `self.dropout` itself (the *shared* dropout module, not `hate_head[2]`), then calls `self.hate_head[3](...)` (the second `Linear`). It reimplements the Sequential's internal steps by hand instead of just running it end to end.

**Why does this matter?** If you "simplified" this to `self.hate_head(cls_output)`, the model would build fine and `load_state_dict` would succeed with no errors — the weight shapes match either way. But the *outputs would be numerically different*, because the two versions apply dropout differently (the real training code reuses `self.dropout` for both the shared representation and inside each head; a naive rewrite would use `hate_head`'s own internal dropout layer instead). At inference time with `model.eval()`, dropout is disabled anyway, so in this specific project this particular mismatch happens to be harmless — but the deeper lesson is what matters: **you must copy training code exactly, never "clean it up," because you cannot always predict which shortcuts are safe.**

### `outputs.last_hidden_state[:, 0, :]` — what is this?

DistilBERT turns a sentence into a sequence of vectors, one per token, shaped `(batch, num_tokens, 768)`. `[:, 0, :]` takes every batch item's **first token's** vector — that's the `[CLS]` token, a special token BERT-family models prepend to every input specifically to serve as a summary of the whole sentence after training. This one line is "collapse a sentence into a single vector."

### Why `squeeze(-1)` on the two auxiliary logits

`hate_head`'s last layer is `Linear(128, 1)` — it outputs shape `(batch, 1)`, a column vector of single numbers. `squeeze(-1)` removes that trailing size-1 dimension so you get shape `(batch,)` — a flat list of numbers, one per input. This matters because `BCEWithLogitsLoss` (used during training) expects that flat shape to line up with the flat label tensor; a shape mismatch there is a classic PyTorch bug.

---

## 3. `model_registry.py` — load once, reuse forever

```python
class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, LightweightDualHeadModel] = {}
        self._tokenizers: dict[str, AutoTokenizer] = {}

    def load_all(self) -> None:
        for lang, cfg in CHECKPOINTS.items():
            ...
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            model = LightweightDualHeadModel(cfg["backbone"])
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            self._models[lang] = model
            self._tokenizers[lang] = tokenizer

    def get(self, language: str):
        if language not in self._models:
            raise KeyError(language)
        return self._models[language], self._tokenizers[language]

registry = ModelRegistry()
```

### Why this exists as a class instead of just loading models as global variables

You *could* write `model_en = load_model("en")` as a bare module-level line. The class exists because:
1. **It's testable.** `conftest.py`'s fixture controls exactly when loading happens (once per test session), which you can't do cleanly with import-time side effects.
2. **It has a clean failure mode.** `.get()` raising `KeyError` for an unknown language is explicit and catchable — a bare dict lookup elsewhere in the code would raise the same error in a less intentional-looking place.
3. **`registry = ModelRegistry()` at the bottom is a singleton** — one instance, imported by both `main.py` and the tests, so everyone shares the same loaded models instead of each accidentally loading their own copy.

### The four steps inside `load_all()`, and why each one exists

1. `AutoTokenizer.from_pretrained(str(tokenizer_path))` — loads the *saved* tokenizer files (`tokenizer.json`, `tokenizer_config.json`) from disk, not from the internet. Passing a local folder path instead of a model name like `"distilbert-base-uncased"` is what makes this offline/reproducible — the tokenizer this loads is guaranteed to match exactly what tokenized the training data.
2. `LightweightDualHeadModel(cfg["backbone"])` — this is the important, easy-to-miss detail: even though you're about to overwrite every weight with the checkpoint, you still have to **construct the model with the right backbone name first**, because that's what determines the encoder's architecture (number of layers, hidden size) that the checkpoint's weights need to slot into. This line also downloads/loads `distilbert-base-uncased`'s pretrained weights from HuggingFace as a *starting point* — wasted work, but harmless, since step 3 immediately replaces them.
3. `torch.load(model_path, map_location="cpu", weights_only=False)` — reads the `.pt` file into a Python dict of tensors. `map_location="cpu"` matters because the checkpoint might have been saved from a GPU (`.cuda()`) session on Kaggle; without this, loading would fail or attempt to grab a GPU that doesn't exist on this machine. `weights_only=False` is required here because this checkpoint is a raw `state_dict`, and some PyTorch versions default to a stricter loading mode meant for arbitrary pickled objects; if you ever load a checkpoint from an untrusted source, this flag is a real security consideration (arbitrary code execution via pickle) — worth knowing even though it's not a risk for weights you trained yourself.
4. `model.eval()` — switches off `Dropout` and freezes `BatchNorm` statistics. Training mode has stochastic behavior (randomly zeroing neurons) that helps prevent overfitting; inference wants deterministic, reproducible output. **Forgetting `.eval()` is one of the most common real-world PyTorch bugs** — the model still "works" but gives slightly different answers on identical input every call.

### Why raise `FileNotFoundError` with a specific message about Git LFS

```python
if not model_path.exists():
    raise FileNotFoundError(
        f"Checkpoint for language '{lang}' not found at {model_path}. "
        "If this is the XLM-R checkpoint, it may still be an "
        "un-pulled Git LFS pointer file."
    )
```

This is defensive coding aimed at a **specific, known failure mode you already hit in this exact project** (the XLM-R checkpoints are still 135-byte LFS pointers, not real weights). A generic `FileNotFoundError` from `torch.load` itself would just say "file not found" — true, but not actionable. This version tells future-you (or a teammate) *why* it's probably missing and what to check first, saving the 20 minutes of confusion you actually went through debugging this earlier in the session.

---

## 4. `schemas.py` — the API's contract, enforced automatically

```python
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="...")
    language: str = Field(..., description="...")

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
```

### Why Pydantic instead of hand-written `if` checks

Without this, `main.py`'s `/predict` function would need:
```python
data = await request.json()
if "text" not in data or not isinstance(data["text"], str):
    raise HTTPException(422, "text is required and must be a string")
if len(data["text"]) > 2000:
    raise HTTPException(422, "text too long")
# ...repeat for every field, every endpoint...
```
Instead, declaring `text: str = Field(..., min_length=1, max_length=2000)` gets you all of that for free, and FastAPI runs it *before* your function body even starts — your `predict()` function only ever sees already-valid data. This is what "the type hints are the validation" means in practice, and it's the concrete answer to "how did you implement input validation" in an interview.

`...` (Python's `Ellipsis` object) as the first argument to `Field` means "this field is required, no default value." `Field(default="auto", ...)` would mean optional-with-a-default, if you needed that instead.

### Why `language: str` has no `pattern` regex restricting it to `"en"|"bn"`

This was a deliberate choice, not an oversight: I could have written `Field(..., pattern="^(en|bn)$")`, but that duplicates the list of valid languages in *two* places — here and in `config.SUPPORTED_LANGUAGES`. Instead, `schemas.py` only validates *shape* ("is this a non-empty string"), and `main.py`'s `/predict` function validates *value* ("is this string one of our actual supported languages") by checking against `SUPPORTED_LANGUAGES` directly. One source of truth again, and as a side benefit you get a clearer error message (`400 Unsupported language 'fr'. Supported: ('en', 'bn')`) instead of Pydantic's more generic regex-mismatch message.

### `model_config = ConfigDict(protected_namespaces=())`

Pydantic reserves any field starting with `model_` for its own internal use (things like `model_config` itself), and warns if you define a field like `model_used` that collides with that reserved prefix. This line tells Pydantic "I know, I'm doing it on purpose, stop warning me." It's cosmetic — the code works either way — but a clean test run with zero warnings is something you should be able to show in an interview without explaining away noise.

### Why two separate response schemas for health and model-info instead of one generic dict

`HealthResponse` and `ModelInfoResponse` are tiny, single-purpose classes. This means `/health`'s documented response in `/docs` is exactly `{"status": str, "models_loaded": [str]}` — nothing more, nothing speculative. A generic `dict` return type would technically work but would show up in `/docs` as "returns an object" with no field-level detail, defeating the entire point of using FastAPI's schema generation.

---

## 5. `logging_config.py` — structured logs

```python
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)
```

### "Structured logging" — what the word actually means

Compare two log lines:
```
2026-07-13 predict lang=en label=sarcastic confidence=0.9961 latency_ms=105.9
```
vs.
```json
{"timestamp": "2026-07-13T...", "level": "INFO", "logger": "hate_sarcasm_api", "message": "predict lang=en label=sarcastic confidence=0.9961 latency_ms=105.9"}
```
The first is only readable by a human eyeballing a terminal. The second is a JSON object — a program (a log aggregator like Datadog, CloudWatch, or even a local `jq` command) can parse it, filter by `level == "ERROR"`, or compute average `latency_ms` across a million lines, without regex-scraping free text. "Structured" specifically means *machine-parseable*, not just "formatted nicely." This is a real production concern: at scale, nobody reads logs by eye; they query them.

Note that in this project, the *message itself* (`"predict lang=en label=..."`) is still a free-text string built with `%s` placeholders in `main.py`. A fully structured system would instead pass `lang`, `label`, `confidence` as separate JSON fields, not bake them into one string. This implementation is a reasonable middle ground for a small project — worth naming as a known limitation and possible next step in an interview, which shows judgment rather than just "it's done."

### `root.propagate = False`

Without this, log messages would travel up to Python's root logger too, potentially getting printed twice if something else configures logging elsewhere (like `uvicorn` itself, which has its own loggers). Setting `propagate = False` keeps this app's logger self-contained and prevents duplicate output.

---

## 6. `main.py` — where everything meets

### The `lifespan` pattern

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: loading models")
    registry.load_all()
    logger.info("Startup complete: models=%s", registry.loaded_languages())
    yield
    logger.info("Shutdown")

app = FastAPI(..., lifespan=lifespan)
```

This is FastAPI's official mechanism for "run this code once when the server starts, and optionally more code once when it stops." Everything before `yield` runs at startup; everything after runs at shutdown. `registry.load_all()` living here — not at module import time, not inside the `/predict` function — is *the* answer to "how do you avoid reloading a multi-hundred-MB model on every request." It loads exactly once, when `uvicorn` boots the app, and stays in memory (inside the `registry` singleton) for every request afterward.

### Why `/predict`'s validation happens in this specific order

```python
text = req.text.strip()
if not text:
    raise HTTPException(422, ...)              # 1. is the input even usable?

if req.language not in SUPPORTED_LANGUAGES:
    raise HTTPException(400, ...)               # 2. is the request well-formed?

try:
    model, tokenizer = registry.get(req.language)
except KeyError:
    raise HTTPException(503, ...)               # 3. is the *server* able to serve it?
```

This ordering is deliberate and each status code means something different:
- **422** ("Unprocessable Entity") — the client sent something that fails validation of the data itself (empty/whitespace text). This is a client-side mistake.
- **400** ("Bad Request") — the client asked for something that doesn't exist in this API's vocabulary (an unsupported language code). Also a client-side mistake, but a different flavor — the shape was fine, the *value* wasn't.
- **503** ("Service Unavailable") — this would only trigger if a language passed the 400 check (meaning it's in `SUPPORTED_LANGUAGES`) but somehow isn't in the registry's loaded models — e.g., if that checkpoint failed to load at startup. This is a **server-side** problem, not the client's fault, which is exactly what 503 communicates. In today's codebase this branch is unreachable in practice (since `SUPPORTED_LANGUAGES` is derived from the same `CHECKPOINTS` dict the registry loads from), but it's there as a defensive guard for a future world where, say, loading `bn` fails but `en` succeeds and the server starts anyway in a degraded state.

Getting HTTP status codes right (4xx = client's fault, 5xx = server's fault) is a small thing that signals real API design maturity in an interview.

### The actual inference block

```python
inputs = tokenizer(text, truncation=True, max_length=128, padding=True, return_tensors="pt")

with torch.no_grad():
    logits, hate_logit, sarc_logit = model(inputs["input_ids"], inputs["attention_mask"])
    probs = torch.softmax(logits, dim=-1)[0]
    label_id = int(torch.argmax(probs).item())
```

- `tokenizer(text, ...)` — converts the raw string into token IDs the model understands, e.g. `"hello"` → `[101, 7592, 102]`. `truncation=True, max_length=128` caps how long an input can be (must match the length used at training time — `MAX_LEN = 128` in the original notebook, mirrored in `config.py`). `padding=True` pads shorter sequences so the batch has a uniform shape (irrelevant here with batch size 1, but harmless and keeps the code correct if batching were added later). `return_tensors="pt"` returns PyTorch tensors instead of plain Python lists, since the model expects tensors.
- `torch.no_grad()` — tells PyTorch not to track operations for gradient computation. During training, PyTorch builds a computation graph so it can run backpropagation; at inference time nothing gets trained, so building that graph is pure wasted memory and compute. This context manager is one of the most universally-expected lines in any inference-serving code — its absence is an instant red flag in a code review.
- `torch.softmax(logits, dim=-1)` — turns the 3 raw model outputs (arbitrary real numbers, called "logits") into 3 numbers that sum to 1.0 and can be read as probabilities.
- `[0]` — the model always returns a batch dimension even for a single input (shape `(1, 3)`); `[0]` extracts the single actual result (shape `(3,)`).
- `torch.argmax(probs)` — the index of the highest-probability class, i.e. the model's actual prediction.

### `torch.sigmoid()` for the auxiliary heads, vs `torch.softmax()` for the main label

This is a real, testable distinction:
- The **3-class head** (`non_hateful` / `hateful` / `sarcastic`) is **mutually exclusive** — a text is exactly one of the three. Softmax is correct here because it forces the three probabilities to compete and sum to 1.
- The **hate head** and **sarcasm head** are each an *independent binary question* — "is this hateful, yes/no" and "is this sarcastic, yes/no" — and a text could in principle be both. Sigmoid squashes each one separately into a 0–1 range **without** forcing them to sum to anything, because they're not competing against each other. Using softmax there would be a genuine bug — it would understate confidence any time both heads legitimately fire.

### Latency measurement

```python
start = time.perf_counter()
...
latency_ms = (time.perf_counter() - start) * 1000
```

`time.perf_counter()` is a monotonic, high-resolution clock — it only ever moves forward and isn't affected by system clock adjustments (NTP sync, DST changes), unlike `time.time()`. For measuring elapsed duration, `perf_counter()` is always the correct choice; `time.time()` is for "what wall-clock time is it right now," a different question. This is a small detail interviewers sometimes probe specifically because it's easy to get wrong and rarely taught explicitly.

### CORS

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

Browsers enforce the "same-origin policy": JavaScript running on one origin (e.g. a Chrome extension, which has its own `chrome-extension://<id>` origin) is blocked by the browser itself from calling `fetch()` on a different origin (`http://localhost:8000`) unless the *server* explicitly opts in via CORS headers. Without this middleware, the extension's `fetch("http://localhost:8000/predict")` call would fail in the browser console with a CORS error — not a bug in your Python code at all, but invisible unless you know to look for it. `allow_origins=["*"]` is intentionally permissive for local development; the code comment above it flags that this needs tightening (to the extension's actual origin) before any real deployment — an important thing to say out loud in an interview, since leaving `"*"` in production is a genuine, commonly-cited security misconfiguration.

---

## 7. `tests/conftest.py` and the test files — why they're shaped this way

```python
@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
```

`scope="session"` is the single most important line in the whole test suite from a practical standpoint. Without it, pytest's default fixture scope is "function" — meaning every single test function would get its own fresh `TestClient`, which means every test would re-trigger `lifespan` and reload both DistilBERT checkpoints from disk. With two models at hundreds of MB each, that's the difference between a 3-second test run and a multi-minute one. `scope="session"` says "build this once, share it across every test in this run."

`with TestClient(app) as c:` — the `with` block is what actually triggers `lifespan`'s startup code (`registry.load_all()`) when the client is created, and its shutdown code when the block exits. Using `TestClient(app)` without the `with` wouldn't run `lifespan` at all in some FastAPI/Starlette versions — an easy mistake that silently gives you tests against an app with zero models loaded.

### Why the tests are split into two files

`test_health_and_info.py` covers the two read-only, no-model-inference endpoints. `test_predict.py` covers the one endpoint that actually runs a model. This isn't just organizational neatness — it reflects that these are testing genuinely different concerns: "does the API report its own state correctly" vs. "does the API correctly validate and process a prediction request." Separating them means if `test_predict.py` starts failing, you already know it's not a health-check problem.

### The shape of `test_predict.py`'s assertions

```python
assert body["label"] in {"non_hateful", "hateful", "sarcastic"}
assert 0.0 <= body["confidence"] <= 1.0
```

These tests deliberately do **not** assert an exact expected label for arbitrary input (e.g. `assert body["label"] == "sarcastic"` for some input text). That would make the test suite an accuracy benchmark, which is the wrong job for a unit test — model accuracy is what Phase 6's evaluation notebooks already measured (F1 scores against a labeled test set). What this test suite verifies is **contract correctness**: given valid input, do we get back a well-formed response, with values in the right range, matching the right schema. The one exception is `test_predict_confident_sarcasm_is_detected`, which checks a single deliberately-obvious sarcastic sentence scores `> 0.5` on `is_sarcastic_score` — this is a coarse smoke test to catch a totally broken model (e.g. wrong checkpoint loaded, architecture mismatch), not a real accuracy claim. The comment in that test says so explicitly.

### `@pytest.mark.parametrize`

```python
@pytest.mark.parametrize("bad_language", ["fr", "EN", "", "english"])
def test_predict_rejects_unsupported_language(client, bad_language):
    resp = client.post("/predict", json={"text": "hello", "language": bad_language})
    assert resp.status_code == 400
```

This runs the same test body four times, once per value in the list, and reports each as a separate named test result (`test_predict_rejects_unsupported_language[fr]`, `[EN]`, etc.). It's the alternative to copy-pasting the same test four times with a different string each time. `"EN"` (uppercase) is in the list on purpose — it verifies the API is case-sensitive about language codes and doesn't silently accept `"EN"` as equivalent to `"en"`, a subtle bug class worth explicitly guarding against.

---

## 8. `verify_checkpoints.py` — why this exists outside the test suite

This script isn't part of `pytest` on purpose. Its job is different: **a fast, standalone, human-readable sanity check you run once by hand** when something feels wrong, without pytest's fixture machinery or assertions getting in the way of just *looking at* what the model predicted for a known sentence. When we hit the Python 3.9 `X | None` syntax error earlier, this script is exactly what caught it — cheaply, with a readable traceback, before it would have shown up as a more confusing pytest collection error.

---

## Recap: how a single request actually flows through this system

1. Chrome extension sends `POST /predict {"text": "...", "language": "en"}`.
2. FastAPI matches the URL to `main.predict()`, and — before your code runs at all — Pydantic validates the JSON against `PredictRequest` (is `text` a non-empty string ≤2000 chars, is `language` present).
3. Inside `predict()`: strip whitespace, re-check `text` isn't now empty, check `language` is actually supported (`400` if not).
4. `registry.get("en")` returns the already-loaded English model and tokenizer — no disk I/O, no download, because `lifespan` loaded it at server startup.
5. Tokenizer turns text into tensors; model runs a forward pass inside `torch.no_grad()`; softmax + sigmoid turn raw logits into probabilities.
6. Response is packed into a `PredictResponse`, which FastAPI serializes to JSON and returns — and because `response_model=PredictResponse` was declared on the route, FastAPI also validates that *your own return value* matches the schema, catching bugs where you accidentally return the wrong shape.
7. A structured JSON log line is written recording what happened.

Every one of those seven steps is something you now understand well enough to explain, defend, and modify.
