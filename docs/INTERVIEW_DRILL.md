# Interview Drill — Test Yourself

Read `CODE_WALKTHROUGH.md` first. This doc is for active recall, not re-reading — cover the answer, say yours out loud, then check. Organized easy → hard, grouped by topic. Answers reference the actual file/line so you can jump back if you're fuzzy.

---

## Round 1 — FastAPI fundamentals

**Q1. What does `@app.post("/predict", response_model=PredictResponse)` actually do — name three distinct things.**
> Answer: (1) Registers a route — `POST /predict` calls this function. (2) `response_model=PredictResponse` makes FastAPI validate and serialize your return value against that schema, and also documents the exact response shape in `/docs`. (3) It generates OpenAPI schema automatically, so `/docs` shows a working "Try it out" form with zero extra code.

**Q2. Why do we run `uvicorn`, not just `python main.py`?**
> Answer: FastAPI apps are ASGI applications — they define *what* to do with a request, but something has to actually open a network socket, accept TCP connections, parse HTTP, and hand off each request. `uvicorn` is that ASGI server. FastAPI itself has no built-in "listen on a port" capability.

**Q3. What HTTP status code would you return for: (a) missing required field, (b) unsupported language value, (c) model failed to load at startup? Why different codes?**
> Answer: (a) 422 Unprocessable Entity — client sent malformed data. (b) 400 Bad Request — well-formed but semantically invalid value. (c) 503 Service Unavailable — server-side problem, not the client's fault. 4xx = client's mistake, 5xx = server's mistake; picking the right one tells API consumers who needs to fix what.

**Q4. What's the difference between path parameters, query parameters, and a request body — and which did we use here, and why?**
> Answer: Path params (`/predict/{id}`) are part of the URL structure; query params (`?lang=en`) are key-value pairs after `?`; a request body is JSON sent in the POST body. We used a request body because we're sending structured data (text + language) that doesn't belong in a URL — URLs have length limits and aren't meant for free-text payloads, and a body lets Pydantic validate the whole structured shape at once.

**Q5. What is CORS, and what specifically breaks without it in this project?**
> Answer: Browsers block JavaScript on one origin from calling `fetch()` on a different origin unless the *server* sends CORS headers allowing it. Without `CORSMiddleware`, the Chrome extension's `fetch("http://localhost:8000/predict")` would fail in the browser console — not because of a Python bug, but because the browser itself refuses to let the response through to the extension's JS.

---

## Round 2 — Pydantic and validation

**Q6. What's the difference between `Field(...)` and `Field(default="auto")`?**
> Answer: `...` (Ellipsis) marks the field as required with no default — a request missing it fails validation. `default="auto"` makes it optional; if omitted, Pydantic fills in `"auto"` automatically.

**Q7. Why does `PredictRequest` NOT restrict `language` to a regex like `^(en|bn)$`, when we clearly only support two languages?**
> Answer: To avoid duplicating the list of supported languages in two places (the schema and `config.SUPPORTED_LANGUAGES`). Instead, the schema validates *shape* (non-empty string), and `main.py`'s endpoint validates *value* against `SUPPORTED_LANGUAGES` directly, giving one source of truth and a clearer, more specific error message.

**Q8. If I send `{"text": "hello", "language": "en", "extra_field": 123}`, what happens by default in Pydantic v2, and why might that matter?**
> Answer: By default, Pydantic v2 silently ignores extra fields not declared on the model. It matters because a client typo or a stale field from an old API version won't get flagged — if you wanted strict validation instead, you'd set `model_config = ConfigDict(extra="forbid")`. Worth knowing as a deliberate trade-off, not an oversight, in this project.

**Q9. What does `response_model=PredictResponse` protect against that a plain `return {...}` dict wouldn't?**
> Answer: It validates your *own* return value before sending it to the client. If a bug caused you to accidentally return a field with the wrong type (e.g. `confidence` as a string instead of float), FastAPI would catch that at the boundary instead of silently shipping malformed JSON to the client.

**Q10. Why did `PredictResponse` need `model_config = ConfigDict(protected_namespaces=())`?**
> Answer: Pydantic reserves the `model_` prefix for its own internals and warns when a user-defined field (`model_used`) collides with it. The config line explicitly opts out of that protection since we're intentionally using `model_used` as a field name, not conflicting with any real Pydantic internal.

---

## Round 3 — PyTorch / model-serving specifics

**Q11. Why is `torch.no_grad()` used at inference time, and what would happen (functionally and performance-wise) if you forgot it?**
> Answer: `no_grad()` disables gradient tracking — the bookkeeping PyTorch needs for backpropagation during training. At inference, nothing gets trained, so tracking gradients is wasted memory and compute (extra tensors kept alive for a backward pass that will never run). Forgetting it wouldn't break correctness, but it would mean higher memory usage and slower inference — a real, measurable production cost, not just a style issue.

**Q12. What does `model.eval()` actually change at the tensor-operation level? Name the two most common layers it affects.**
> Answer: `Dropout` — during training it randomly zeroes a fraction of activations to reduce overfitting; in `eval()` mode it's a no-op (passes everything through). `BatchNorm` — during training it uses the current batch's statistics; in `eval()` mode it uses fixed running statistics accumulated during training. Forgetting `.eval()` at inference gives nondeterministic or subtly wrong outputs even with correct weights loaded.

**Q13. Why does the model use `softmax` for the 3-class output but `sigmoid` for the hate/sarcasm auxiliary heads?**
> Answer: The 3-class output is mutually exclusive (a text is exactly one of non_hateful/hateful/sarcastic) — softmax forces the probabilities to compete and sum to 1, which is correct for a single-choice decision. The two auxiliary heads are independent binary questions that aren't mutually exclusive (a text could theoretically be both hateful and sarcastic) — sigmoid scores each independently in [0,1] without forcing a trade-off between them.

**Q14. Explain `outputs.last_hidden_state[:, 0, :]` to someone who's never seen a transformer.**
> Answer: The encoder outputs one vector per input token, shape `(batch, num_tokens, hidden_size)`. Index `0` on the token axis grabs the special `[CLS]` token that BERT-family models prepend to every input — through training, that token's final vector becomes a learned summary of the whole sentence. This line converts "a sequence of token vectors" into "one vector representing the whole input."

**Q15. Why does `model_registry.py` call `torch.load(..., map_location="cpu", weights_only=False)` instead of just `torch.load(path)`?**
> Answer: `map_location="cpu"` is needed because the checkpoint may have been saved from a GPU session (Kaggle); without it, loading could fail or try to allocate a GPU that doesn't exist locally. `weights_only=False` is required because the checkpoint is a plain state_dict saved in a way that some PyTorch versions' stricter default loading mode doesn't accept without this flag. (Bonus: mention that `weights_only=False` is a real security consideration if loading a checkpoint from an untrusted source, since it permits arbitrary pickled objects — not a risk here since we trust our own training pipeline.)

**Q16. Why must `architecture.py`'s `forward()` be copied exactly from the training notebook instead of rewritten "more cleanly"?**
> Answer: `load_state_dict()` matches saved tensors to a model's layers by parameter name and shape — it says nothing about *how* those layers are used inside `forward()`. Two different `forward()` implementations can both successfully load the same checkpoint (same layer names/shapes) yet compute different outputs, if they apply operations (like dropout, or the order of concatenation) differently. Structural compatibility (loading) and behavioral correctness (matching the actual trained computation) are two separate guarantees — only exact copying guarantees both.

**Q17. What's the actual difference between a "checkpoint," a "state_dict," and a "model" in this codebase?**
> Answer: The **model** (`LightweightDualHeadModel`) is a Python class defining architecture — layers and how data flows through them, but with randomly-initialized or pretrained-but-not-fine-tuned weights until you load something into it. A **state_dict** is a plain Python dict mapping layer names to weight tensors — just numbers, no code. A **checkpoint** is that state_dict serialized to disk (the `.pt` file). "Loading a checkpoint" means reading the file back into a state_dict, then calling `model.load_state_dict(state_dict)` to inject those numbers into an already-constructed model instance.

---

## Round 4 — System design / "why did you build it this way"

**Q18. Walk me through what happens, end to end, when the server process starts up — before any request arrives.**
> Answer: `uvicorn` imports `app.main`, which builds the `FastAPI` app with `lifespan=lifespan`. On startup, `uvicorn` enters the `lifespan` context manager, running everything before `yield` — which calls `registry.load_all()`. That loads each language's tokenizer from its local folder, constructs a fresh `LightweightDualHeadModel` with the right backbone, loads the checkpoint's state_dict from disk, injects it via `load_state_dict`, and switches to `.eval()` mode. Only after this completes does `uvicorn` start accepting connections — so the very first request is already fast, because model loading already happened.

**Q19. Why is model loading centralized in a `ModelRegistry` singleton instead of loading models directly inside the `/predict` function?**
> Answer: If loading happened inside `/predict`, every single request would reload a multi-hundred-MB checkpoint from disk and rebuild the DistilBERT backbone — turning a ~100ms request into a multi-second one, repeatedly. Centralizing loading in a singleton that's populated once at startup (via `lifespan`) means the (slow) load happens exactly once, and every request afterward reuses the same in-memory model — the entire reason inference is fast.

**Q20. Your `config.py` computes paths relative to `Phase 6 (Architectures)/` in the research repo, several directories above the backend. What's the risk of that design, and how would you change it before a real deployment?**
> Answer: It couples the API server to the exact folder layout of the research repo — if someone reorganizes the `Phase 6` folders, or if this backend is ever deployed without the rest of the repo checked out, model loading breaks. Before real deployment (Phase 3, Docker), the right fix is to copy just the needed checkpoint files into the Docker image (or a mounted volume/object storage) at a fixed, backend-owned path, decoupling the server from the research repo's structure entirely.

**Q21. Why does the test suite use `scope="session"` on the `client` fixture? What would happen without it, concretely?**
> Answer: Without `scope="session"`, pytest's default is function scope — a fresh `TestClient` (and therefore a fresh `lifespan` run, reloading both checkpoints) for every single test function. With 13 tests and two multi-hundred-MB models, that would mean the server startup sequence runs 13 times instead of once, turning a ~3.5 second test run into minutes, for no correctness benefit — the models don't need to be reloaded between tests since nothing mutates them.

**Q22. Why doesn't the test suite assert exact predicted labels for arbitrary input text?**
> Answer: That would conflate two different kinds of testing. Unit/API tests should verify the *contract* — given valid input, is the response well-formed, in the right ranges, matching the schema. Model *accuracy* (does it predict the right label) is a machine learning evaluation question, already answered rigorously in the capstone's Phase 6 notebooks using labeled test sets and F1 scores — re-litigating that with a handful of hardcoded assertions in an API test would be both redundant and statistically meaningless (n=1 isn't an accuracy benchmark).

**Q23. The English and Bangla models were trained with different backbones (`distilbert-base-uncased` vs `distilbert-base-multilingual-cased`). Why not use one multilingual model for both languages, to simplify the registry?**
> Answer: (Reasoning question — connect to the actual capstone findings.) The capstone project's own transfer-learning experiments found that a single model trained on one language transfers very poorly zero-shot to another (F1 collapsing to ~0.26–0.33) — languages need either dedicated backbones or targeted fine-tuning. A single generic multilingual model would likely underperform two language-specific ones, which is presumably why the original training used a matched, language-appropriate backbone for each.

**Q24. If `/predict` started timing out under load, what would you check first, and what would you change?**
> Answer: First check: is this CPU inference (no GPU) — DistilBERT forward passes on CPU are the likely bottleneck at any real concurrency, and `uvicorn`'s default single worker processes requests one at a time for CPU-bound work like this. Next steps in rough order: measure actual per-request latency (already logged), consider batching concurrent requests together, consider running multiple `uvicorn` workers behind a load balancer, and only then consider GPU inference or a smaller/quantized model if latency is still unacceptable.

---

## Round 5 — Rapid fire (one-line answers)

- **What's an ASGI server?** A program that handles the actual network/HTTP layer and calls into an async-capable Python app (like FastAPI) per request — `uvicorn` is one.
- **What does `--reload` do in `uvicorn app.main:app --reload`?** Watches source files and restarts the server automatically on change — dev convenience only, never used in production.
- **Why `time.perf_counter()` instead of `time.time()` for latency measurement?** `perf_counter()` is monotonic and high-resolution, unaffected by system clock adjustments — correct for measuring elapsed duration; `time.time()` answers "what time is it," a different question.
- **What does `torch.argmax(probs)` return?** The index of the highest value in the tensor — here, the predicted class ID.
- **Why `.item()` after `torch.sigmoid(hate_logit)[0]`?** Converts a single-element PyTorch tensor into a plain Python float, since Pydantic response models need native Python types, not tensors.
- **What's the purpose of `tokenizer(..., padding=True)` when we only ever send one text at a time?** Harmless at batch size 1; keeps the code correct and unchanged if batched inference were added later, since padding matters once a batch has sequences of different lengths.
- **Why is `.gitignore` excluding `.venv/`?** Virtual environments are large, machine-specific, and fully reproducible from `requirements.txt` — committing them bloats the repo for zero benefit.
- **What's the difference between `GET /health` and `GET /model-info`?** `/health` answers "is the process alive and are models loaded" (a boolean-ish liveness signal); `/model-info` answers "what exactly is loaded" (descriptive metadata) — different consumers (a deploy platform's health check vs. a human/debugging client) want different things.

---

## Self-check: can you do this without the doc?

Close this file and, out loud or on paper, explain:
1. The full request lifecycle for `POST /predict`, from the extension's `fetch()` call to the JSON response.
2. Why the model has to be loaded exactly once, and where that happens in the code.
3. The difference between softmax and sigmoid, and why each is used where it is.
4. Two things you'd change about this code before calling it production-ready.

If you can do all four cleanly, you're ready to talk about this project in an interview.
