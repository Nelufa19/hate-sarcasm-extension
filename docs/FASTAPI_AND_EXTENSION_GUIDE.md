# Learning Guide: FastAPI + Browser Extension Integration

This is a teaching doc, not just a spec. It explains FastAPI from first principles, then walks through exactly how it plugs into the two Lightweight DistilBERT dual-head models from the capstone project, and finally how a browser extension talks to it. Read top to bottom the first time; use it as reference after.

---

## Part 1 — What FastAPI actually is, and why it fits here

### The problem it solves

Your models currently exist as `.pt` files that only mean something inside a Python process that has `torch` and `transformers` loaded and the right class definitions in scope. A browser extension is JavaScript running in Chrome — it cannot import a `.pt` file. It can only do one thing: make HTTP requests.

**FastAPI's job**: keep the model loaded once in a long-running Python process, and expose "give me a prediction for this text" as an HTTP endpoint — a URL that accepts a request and returns JSON. The extension calls that URL. It never touches Python, torch, or the model file directly.

```
Browser Extension (JS)  --HTTP POST-->  FastAPI server (Python, model loaded in RAM)  --returns-->  JSON prediction
```

### Why FastAPI specifically (vs Flask/Django)

- **Pydantic validation built in** — you declare the shape of a request as a Python class, and FastAPI rejects malformed requests automatically before your code even runs. This directly satisfies your Phase 1 checklist item "Input validation."
- **Async native** — matters less at junior-project scale, but it's the modern standard and what interviewers expect you to name.
- **Automatic docs** — every FastAPI app gets a free interactive page at `/docs` where you can test endpoints in a browser without writing a single line of client code. Enormous for demoing to yourself or an interviewer.
- **Type hints as documentation** — the function signature *is* the contract. No separate schema file to keep in sync.

### The core mental model

A FastAPI app is just a Python object (`app = FastAPI()`) with functions attached to URL paths via decorators:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
```

Run it with `uvicorn` (an ASGI server — the actual program that listens on a network port and hands requests to your FastAPI app):

```bash
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` — that page exists with zero extra code, generated from your function signatures.

That's the entire mental model. Everything else in this project is: (1) more endpoints, (2) request/response shapes declared with Pydantic, (3) a model loaded once at startup instead of a toy dict.

---

## Part 2 — Pydantic: how request validation works

Pydantic models are classes where each attribute has a type. FastAPI uses them to parse and validate the JSON body of incoming requests.

```python
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(..., pattern="^(en|bn)$")
    model_variant: str = Field(default="auto", pattern="^(auto|xlmr|distilbert)$")
```

If a request comes in missing `text`, or with `language: "fr"`, FastAPI returns a `422 Unprocessable Entity` with a precise error message — **before** your prediction code runs. You get input validation for free just by declaring types. This is the answer to "Input validation" in your Phase 1 plan — you don't hand-write `if not text: raise ...` everywhere; you declare the schema once.

Response shape works the same way, and matters because your models output more than one number (3-class label + hate-head + sarcasm-head confidences):

```python
class PredictResponse(BaseModel):
    label: str                 # "non_hateful" | "hateful" | "sarcastic"
    label_id: int
    is_hateful_score: float
    is_sarcastic_score: float
    confidence: float
    language: str
    model_used: str
    latency_ms: float
```

Declaring this explicitly is also what makes `/docs` show a clean example response, and it's what an interviewer means by "did you validate your API contract."

---

## Part 3 — Mapping this project's actual models onto the API

### What you're serving (confirmed from the notebook source)

Both variants share one architecture shape (`LightweightDualHeadModel` in `Phase 6 (Architectures)/distilbert-base (Lightweight model)/lightweight-model.ipynb`):

```python
class LightweightDualHeadModel(nn.Module):
    def __init__(self, model_name, num_classes=3, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)   # DistilBERT backbone
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.hate_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1))
        self.sarcasm_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1))
        # ...fusion layer producing a 3-class logit...
```

Two independently trained checkpoints of this same class exist:

| | Backbone | Checkpoint | Tokenizer |
|---|---|---|---|
| English | `distilbert-base-uncased` | `Phase 6 (Architectures)/distilbert-base (Lightweight model)/Lightweight English/model.pt (2).zip` | same folder, `tokenizer.json` |
| Bangla | `distilbert-base-multilingual-cased` (multilingual DistilBERT) | `Phase 6 (Architectures)/distilbert-base (Lightweight model)/Lightweight Bangla/model.pt` | same folder, `tokenizer.json` |

Both are real, fully-downloaded weights (confirmed earlier — Bangla 540MB raw `.pt`, English 267MB inside a zip that is itself a valid `torch.save` archive; you can `torch.load()` the zip path directly, no manual unzip needed).

**Note for later:** the two XLM-RoBERTa dual-head checkpoints (`xlm-roberta-base (shared encoder)/dualhead_xlmr_English` and `..._Bangla`) are currently *not* usable — they're Git LFS pointer files (135 bytes) that never fully uploaded, most likely a GitHub LFS storage quota issue. The 4-model, user-selectable design below is written so adding them back later is a 10-line change, not a redesign — but for now we serve only the 2 working Lightweight models.

### The model registry pattern

You need one Python object that knows how to load and hold all your models, so the FastAPI endpoint doesn't reload a model from disk on every request (that would take seconds per call — unacceptable for a browser extension).

```python
# app/model_registry.py
import torch
from transformers import AutoTokenizer
from .architecture import LightweightDualHeadModel

CHECKPOINTS = {
    "en": {
        "path": "../../Phase 6 (Architectures)/distilbert-base (Lightweight model)/Lightweight English/model.pt (2).zip",
        "backbone": "distilbert-base-uncased",
    },
    "bn": {
        "path": "../../Phase 6 (Architectures)/distilbert-base (Lightweight model)/Lightweight Bangla/model.pt",
        "backbone": "distilbert-base-multilingual-cased",
    },
}

class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def load_all(self):
        for lang, cfg in CHECKPOINTS.items():
            tok = AutoTokenizer.from_pretrained(cfg["backbone"])
            model = LightweightDualHeadModel(cfg["backbone"])
            state = torch.load(cfg["path"], map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            model.eval()
            self.tokenizers[lang] = tok
            self.models[lang] = model

    def get(self, lang: str):
        return self.models[lang], self.tokenizers[lang]

registry = ModelRegistry()
```

This is the object your Phase 1 checklist item **"Load model once at startup"** refers to. FastAPI has a dedicated lifecycle hook for exactly this:

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .model_registry import registry

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load_all()   # runs once, when the server boots
    yield
    # (optional) cleanup on shutdown goes here

app = FastAPI(lifespan=lifespan)
```

Everything after this loads once. Every subsequent request reuses the already-loaded model in RAM — this is the difference between a ~2-second cold model load and a ~50-150ms inference call.

### Since you currently have only the DistilBERT family, why design for 4 models at all?

Because the *shape* of the decision (which language, which architecture) doesn't change — only the registry's dict of available keys does. Building the registry as a dict keyed by `(language, variant)` now means "add XLM-R" later is: pull the real weights, add two dict entries, done. No endpoint code changes. This is generally good API design — the client contract (`language`, `model_variant`) shouldn't need to change just because the server adds a model.

### The `/predict` endpoint itself

```python
# app/main.py (continued)
import time
from fastapi import HTTPException
from .schemas import PredictRequest, PredictResponse
import torch

LABELS = {0: "non_hateful", 1: "hateful", 2: "sarcastic"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()

    if req.language not in registry.models:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {req.language}")

    model, tokenizer = registry.get(req.language)

    inputs = tokenizer(req.text, truncation=True, max_length=128,
                        padding=True, return_tensors="pt")

    with torch.no_grad():
        logits, hate_logit, sarc_logit = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=-1)[0]
        label_id = int(torch.argmax(probs).item())

    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        label=LABELS[label_id],
        label_id=label_id,
        is_hateful_score=torch.sigmoid(hate_logit).item(),
        is_sarcastic_score=torch.sigmoid(sarc_logit).item(),
        confidence=probs[label_id].item(),
        language=req.language,
        model_used=f"distilbert-dualhead-{req.language}",
        latency_ms=round(latency_ms, 2),
    )
```

Every piece of your Phase 2 checklist maps directly onto this function:
- **Confidence scores** → `probs[label_id].item()`, plus the two auxiliary head scores.
- **Latency measurement** → the `time.perf_counter()` wrap, returned in the response itself so the extension can display it, and loggable server-side.
- **Structured logging** → wrap this in a logging call (`logger.info(json.dumps({...}))`) instead of `print` — structured meaning machine-parseable (JSON lines), not prose.

### `/health` and `/model-info`

```python
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(registry.models.keys())}

@app.get("/model-info")
def model_info():
    return {
        "available_languages": list(registry.models.keys()),
        "architecture": "DistilBERT dual-head (shared encoder, hate + sarcasm auxiliary heads)",
        "note": "XLM-RoBERTa dual-head variant not yet available (pending LFS re-upload)",
    }
```

`/health` is what a deploy platform (or Docker `HEALTHCHECK`, or a future Kubernetes readiness probe) polls to know the process is alive and models are actually loaded — not just that the port is open.

---

## Part 4 — The `model_variant` / 4-model design (for when XLM-R is restored)

Your original ask was: default to best-available model per language, but let the user pick. Here's the shape that supports it without overengineering it now:

```python
class PredictRequest(BaseModel):
    text: str
    language: str          # "en" | "bn"
    model_variant: str = "auto"   # "auto" | "distilbert" | "xlmr"
```

Registry keys become `(language, variant)` tuples once XLM-R is added:

```python
CHECKPOINTS = {
    ("en", "distilbert"): {...},
    ("bn", "distilbert"): {...},
    ("en", "xlmr"): {...},   # add later
    ("bn", "xlmr"): {...},   # add later
}
BEST_PER_LANGUAGE = {"en": "xlmr", "bn": "xlmr"}  # per your capstone's own F1 results — XLM-R dual-head wins both languages
```

`"auto"` resolves to `BEST_PER_LANGUAGE[language]`. Right now, since XLM-R isn't loadable, `BEST_PER_LANGUAGE` should just point at `"distilbert"` for both languages until the weights are fixed — one line to change later, not a rewrite.

**Don't build this dict-of-tuples now if you're only shipping 2 models.** Start with the simple `{"en": ..., "bn": ...}` registry shown in Part 3. Refactor to the tuple-keyed version only when you actually add the second architecture — adding an unused abstraction today is exactly the kind of premature complexity worth avoiding, and it's also a more honest MVP.

---

## Part 5 — Project structure

```
hate-sarcasm-extension/
├── backend/
│   ├── .venv/                  # local Python env (gitignored)
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py             # FastAPI app, endpoints
│   │   ├── schemas.py          # Pydantic request/response models
│   │   ├── model_registry.py   # loads + holds models
│   │   └── architecture.py     # LightweightDualHeadModel class definition
│   └── Dockerfile              # Phase 3
├── extension/
│   ├── manifest.json           # Chrome extension manifest (v3)
│   ├── popup.html
│   ├── popup.js
│   └── content_script.js       # reads selected text from the page
└── docs/
    └── FASTAPI_AND_EXTENSION_GUIDE.md   # this file
```

The `architecture.py` file matters more than it looks: **the model class definition must be byte-identical to the one used during training**, because `load_state_dict` matches weights to layers by name. Copy it verbatim from the notebook (`LightweightDualHeadModel`), don't rewrite it from memory.

---

## Part 6 — How the browser extension talks to this server

### Chrome Extension basics (Manifest V3)

A Chrome extension has, at minimum:
- `manifest.json` — declares permissions, what scripts run where.
- A **popup** (`popup.html` + `popup.js`) — the small window that opens when you click the extension icon.
- A **content script** (optional) — JS injected into the actual web page, used here to grab the user's selected text.

```json
// manifest.json
{
  "manifest_version": 3,
  "name": "Hate & Sarcasm Detector",
  "version": "0.1.0",
  "permissions": ["activeTab", "scripting"],
  "host_permissions": ["http://localhost:8000/*"],
  "action": { "default_popup": "popup.html" },
  "content_scripts": [{
    "matches": ["<all_urls>"],
    "js": ["content_script.js"]
  }]
}
```

`host_permissions` is important and easy to miss: Chrome blocks extension network calls to hosts not explicitly listed here. Point it at wherever your FastAPI server actually lives (`localhost:8000` in dev, your deployed URL in Phase 3).

### Reading selected text

```javascript
// content_script.js
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "getSelection") {
    sendResponse({ text: window.getSelection().toString() });
  }
});
```

### Calling the API from the popup

```javascript
// popup.js
document.getElementById("checkBtn").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const { text } = await chrome.tabs.sendMessage(tab.id, { action: "getSelection" });

  if (!text) {
    document.getElementById("result").innerText = "Select some text first.";
    return;
  }

  const language = document.getElementById("langSelect").value; // "en" | "bn"

  const res = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language, model_variant: "auto" }),
  });

  if (!res.ok) {
    document.getElementById("result").innerText = "Error: " + res.status;
    return;
  }

  const data = await res.json();
  document.getElementById("result").innerText =
    `${data.label} (confidence ${(data.confidence * 100).toFixed(1)}%, ${data.latency_ms}ms)`;
});
```

This is the entire client-side integration: one `fetch` call, one JSON body matching your `PredictRequest` schema, one JSON response matching `PredictResponse`. The Pydantic models on the server side *are* the contract this JavaScript is written against — if you change a field name in `schemas.py`, this JS breaks until updated. That coupling is normal and expected; it's why `/docs` (FastAPI's auto-generated schema page) is useful even for a solo project — it's always the current source of truth for what the extension must send.

### CORS — the thing that will trip you up first

Browsers block a webpage/extension from calling `fetch()` on a different origin (`localhost:8000`) unless the server explicitly allows it. Without this, your first API call from the extension will fail silently in the console with a CORS error, not a helpful error from your own code.

```python
# app/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this before real deployment
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

`allow_origins=["*"]` is fine for local development. Before Phase 3 deployment, narrow it to your actual extension's origin (`chrome-extension://<your-extension-id>`).

---

## Part 7 — Running it end to end (dev loop)

```bash
cd hate-sarcasm-extension/backend
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then:
1. Open `http://localhost:8000/docs`, expand `/predict`, click "Try it out," paste some text, confirm you get a real prediction. Do this before touching the extension at all — isolate backend correctness first.
2. `chrome://extensions` → enable Developer Mode → "Load unpacked" → select the `extension/` folder.
3. Select text on any webpage, click the extension icon, click check.

---

## Part 8 — Why this maps to your interview prep goal

Recapping the connection to your earlier "what can I actually learn from this for junior MLOps/infra interviews" question, now made concrete:

| Your Phase 1–3 checklist item | What you just built it with |
|---|---|
| Export trained model | Already done upstream (`.pt` checkpoints) — you're consuming, not producing, and that's fine to say plainly in an interview |
| FastAPI inference service, `POST /predict` | Parts 1–3 above |
| Input validation | Pydantic `PredictRequest` (Part 2) |
| Docker | Next doc/step — Dockerfile wraps exactly this `app/` folder |
| Load model once at startup | `lifespan` handler + `ModelRegistry` (Part 3) |
| Structured logging | Wrap the `/predict` handler body in JSON-line logging |
| `/health`, `/model-info` | Part 3 |
| Confidence scores, latency measurement | Returned directly in `PredictResponse` |
| GitHub Actions, container registry, deploy, monitoring | Phase 3 of your plan — comes after the Dockerfile |

This is now something you can describe honestly and in detail in an interview: *"I built a FastAPI inference server around a pre-trained multi-task model, with Pydantic-validated request/response schemas, models loaded once at startup via a lifespan hook, and a health/model-info endpoint — then wired a Chrome extension to it via fetch, with CORS configured for cross-origin calls."* Every clause of that sentence is something you can open the code and point to.
