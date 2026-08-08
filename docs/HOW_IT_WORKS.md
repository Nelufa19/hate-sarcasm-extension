# How the Hate & Sarcasm Blur Extension Works

This is a single, comprehensive walkthrough of everything in `hate-sarcasm-extension/` —
what runs, in what order, and why. It covers two halves that talk over HTTP:

```
┌─────────────────────────────┐         ┌──────────────────────────────────┐
│   backend/  (FastAPI)       │ <-HTTP-> │   extension/  (Chrome MV3)        │
│   loads 2 ML models,        │          │   watches the page, extracts     │
│   classifies text           │          │   text, blurs hateful content     │
└─────────────────────────────┘         └──────────────────────────────────┘
```

Read it top to bottom once — each section builds on the last.

---

## Part 1 — What the model actually is

### The architecture: one shared body, two specialist heads, one fusion decision

The model class is `LightweightDualHeadModel` (`backend/app/architecture.py`). It is **not**
a plain "text in, one label out" classifier. It's built from three parts:

```
input text
    │
    ▼
DistilBERT encoder (self.encoder)         ← general language understanding
    │
    ▼
[CLS] token vector (768 numbers, one per input)
    │
    ├──────────────┬──────────────┐
    ▼              ▼              │
hate_head      sarcasm_head       │
"is this        "is this          │
hateful?"       sarcastic?"       │
(0-1 score)     (0-1 score)       │
    │              │              │
    └──────┬───────┴──────────────┘
           ▼
     fusion_mlp (combines the shared vector + both heads' opinions)
           │
           ▼
   3-class decision: non_hateful | hateful | sarcastic
```

- **`encoder`** — a pretrained DistilBERT backbone. It turns a sentence into a sequence of
  768-dimensional vectors, one per token. This part already understands language in general
  (grammar, word meaning) before any of this project's own training happened.
- **`hate_head`** and **`sarcasm_head`** — two small independent classifiers
  (`768 → 128 → 1`) that each answer one narrower yes/no question from the same shared
  sentence representation. This is the "dual-head" design: instead of guessing 3 classes
  blindly from scratch, two specialists each learn one focused signal first.
- **`fusion_mlp`** — takes the shared sentence vector **plus** both heads' intermediate
  opinions and combines everything into the final 3-way call.

### Why this design exists

Hate speech and sarcasm interact in a specific way this project cares about: sarcasm is
often used to disguise hateful intent ("oh sure, because [group] are SO trustworthy...").
A single generic classifier tends to conflate "this is sarcastic" with "this is fine,"
because most sarcasm in ordinary training data is harmless. Giving the model a dedicated
sarcasm-detection head and a dedicated hate-detection head — then fusing both signals for
the final call — is meant to let the model separately reason about *tone* and *toxicity*
instead of forcing one signal to stand in for both.

### Two independently-trained checkpoints, not one shared model

```python
# backend/app/config.py
CHECKPOINTS = {
    "en": {"backbone": "distilbert-base-uncased", ...},
    "bn": {"backbone": "distilbert-base-multilingual-cased", ...},
}
```

There is no single multilingual model doing both languages. English and Bangla each have
their own fully-trained copy of `LightweightDualHeadModel`, with different backbones. This
matters for the demo: **language selection is not cosmetic** — sending Bangla text through
the English checkpoint (or vice versa) will not error, but the prediction is meaningless,
because that checkpoint's weights were never trained on that language at all. The
extension's `detectLanguage()` (covered in Part 4) exists specifically to route text to
the correct checkpoint automatically.

### What a "checkpoint" actually is, concretely

A `.pt` file (`backend/models/Lightweight English/model.pt`, `.../Lightweight Bangla/model.pt`)
is just a dictionary mapping layer names (`"hate_head.0.weight"`, `"fusion_mlp.3.bias"`, ...)
to tensors of trained numbers. It is meaningless without `architecture.py`'s class to tell
Python what those numbers mean and what order operations happen in. Loading a checkpoint is
a two-step process every time the server starts:

1. Build a fresh `LightweightDualHeadModel(backbone_name)` — random/pretrained-but-unfinetuned
   weights, correct *shape* only.
2. `model.load_state_dict(state_dict)` — overwrite every layer's weights with the trained
   numbers from the `.pt` file, matched by name and shape.

`architecture.py`'s class must be **structurally identical**, line for line, to whatever
class produced the checkpoint during training — not "equivalent," not "cleaned up."
`forward()` deliberately reimplements the `hate_head`/`sarcasm_head` Sequential blocks by
indexing `hate_head[0]`, `hate_head[3]` instead of calling `self.hate_head(x)` directly,
because that's exactly how the training code applied dropout. A "simplified" rewrite would
still load without any error (same layer names/shapes) but would silently compute different
numbers — the file's own docstring warns against "cleaning this up" for exactly this reason.

### How a raw prediction becomes a label

```python
inputs = tokenizer(text, truncation=True, max_length=128, padding=True, return_tensors="pt")

with torch.no_grad():
    logits, hate_logit, sarc_logit = model(inputs["input_ids"], inputs["attention_mask"])
    probs = torch.softmax(logits, dim=-1)[0]      # 3-class probabilities, sum to 1.0
    label_id = int(torch.argmax(probs).item())     # index of the winning class

is_hateful_score  = torch.sigmoid(hate_logit)[0].item()   # independent 0-1 score
is_sarcastic_score = torch.sigmoid(sarc_logit)[0].item()  # independent 0-1 score
```

Two different squashing functions are used on purpose, for two different kinds of question:

- **`softmax` on the 3-class output** — `non_hateful` / `hateful` / `sarcastic` are
  **mutually exclusive**: a text is exactly one of the three in this scheme. Softmax forces
  the three probabilities to compete and sum to 1, which is the correct shape for a
  single-choice decision.
- **`sigmoid` on each auxiliary head** — "is this hateful" and "is this sarcastic" are
  **independent binary questions** answered separately; a text could in principle score
  high on both. Sigmoid gives each an unconstrained 0–1 score without forcing a trade-off
  between them. Using softmax here would be a real bug — it would understate one score any
  time both legitimately fire.

`LABELS = {0: "non_hateful", 1: "hateful", 2: "sarcastic"}` (`config.py`) maps the winning
index to the string the rest of the system uses. The **final label the extension acts on**
is this 3-class `label`, not the two auxiliary scores directly — but `is_hateful_score` and
`is_sarcastic_score` are still returned in every response so a caller can see *why*, e.g.
distinguish "confidently hateful" from "borderline."

### `torch.no_grad()` and `model.eval()`

Both appear because this is *inference*, not training:
- `torch.no_grad()` — skips building the computation graph PyTorch needs for
  backpropagation. Nothing is being trained here, so tracking gradients is pure wasted
  memory/compute.
- `model.eval()` (set once, at load time, in `model_registry.py`) — turns off `Dropout`
  (which randomly zeroes activations during training to prevent overfitting) so that the
  same input always produces the same output. Forgetting this is one of the most common
  real-world PyTorch bugs: the model still "works" but gives nondeterministic answers.

---

## Part 2 — The FastAPI backend (`backend/app/`)

### Why a server exists at all

The Chrome extension is JavaScript. It cannot `import torch` or open a `.pt` file — a
browser sandbox has no access to a Python runtime. The backend's entire job is: **load both
models once, keep them resident in memory, and expose "classify this text" as an HTTP
endpoint** that a browser's `fetch()` can call.

```
Chrome extension (JS)  --HTTP POST-->  FastAPI process (Python, models loaded in RAM)  --JSON-->  extension
```

### File map and what each one owns

| File | Owns |
|---|---|
| `config.py` | Where the checkpoints live, backbone names, label mapping — the one place all of that is defined |
| `architecture.py` | The model class (copied verbatim from training) |
| `model_registry.py` | Loading both models once at startup, holding them in memory |
| `schemas.py` | Request/response shape validation (Pydantic) |
| `logging_config.py` | Structured (JSON-lines) logging |
| `main.py` | The actual HTTP endpoints, request orchestration, CORS |

### Loading models once, at startup — not per request

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load_all()   # runs once, when uvicorn boots
    yield
    ...

app = FastAPI(..., lifespan=lifespan)
```

`registry.load_all()` (in `model_registry.py`) loads both tokenizers and both model
checkpoints from disk exactly once, when the server process starts — not inside the
`/predict` function. If loading happened per-request, every single call would re-read
hundreds of MB of weights from disk and rebuild the DistilBERT backbone, turning a
~100ms request into several seconds, repeatedly. The `ModelRegistry` singleton holds
the already-loaded models for the lifetime of the process; every request just reuses them.

`MODELS_DIR` defaults to `backend/models/` (self-contained — works inside the Docker
image with nothing else checked out) but can be overridden via an env var to point at the
research repo's own training-output folders during local development.

### The endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Is the process alive, which language models actually loaded |
| `GET /model-info` | Architecture description + which backbone backs each language |
| `POST /predict` | Classify one piece of text |
| `POST /predict/batch` | Classify up to 50 texts in one HTTP round trip |

`/predict` and `/predict/batch` both funnel through one shared function,
`_run_inference()`, so the two endpoints can never drift apart in validation or scoring
behavior — `/predict/batch` is literally `[_run_inference(item) for item in req.items]`.

**Request validation order, and why each status code is different:**
```python
text = req.text.strip()
if not text:
    raise HTTPException(422, ...)   # malformed input — client's data itself is bad

if req.language not in SUPPORTED_LANGUAGES:
    raise HTTPException(400, ...)  # well-formed but semantically invalid value

try:
    model, tokenizer = registry.get(req.language)
except KeyError:
    raise HTTPException(503, ...)  # server-side: this language isn't actually loaded
```
422 and 400 are both the client's fault (bad shape vs. bad value); 503 would mean the
*server* failed to load a model it claims to support — a different kind of failure,
signaled with a different code.

**Pydantic validation happens before your function body even runs.** `PredictRequest`
declares `text: str = Field(..., min_length=1, max_length=10000)` and `language: str`
(required, no regex restriction — value-checking happens against `SUPPORTED_LANGUAGES`
in `main.py` instead, so the list of valid languages exists in exactly one place). A
request missing a field, or with `text` too long, gets rejected automatically with a
`422` before any model runs.

The optional `id` field on `PredictRequest`/`PredictResponse` exists purely for the
extension's benefit: when a batch of 50 texts comes back as 50 results, the client needs
to know *which DOM element* each result belongs to without relying on array order (which
isn't guaranteed to be preserved, and is fragile if any item errors independently). The
extension assigns a client-side `id` per queued item and matches results back up by that id.

### CORS and the Private Network Access header

Two separate browser-security mechanisms had to be satisfied, and they're easy to
conflate:

1. **CORS** — browsers block a page/extension's JavaScript from reading a cross-origin
   `fetch()` response unless the *server* opts in via `Access-Control-Allow-*` headers.
   `CORSMiddleware` in `main.py` handles this (`allow_origins=["*"]` for local dev — the
   code notes this must be tightened before any real deployment).
2. **Private Network Access (PNA)** — a *separate*, newer Chrome check. When a request's
   origin is "more public" than its target (here: the extension's background worker,
   `chrome-extension://<id>`, calling into `localhost:8000`), Chrome sends an extra
   preflight header (`Access-Control-Request-Private-Network: true`) that plain
   `CORSMiddleware` doesn't know how to answer. Without a response echoing
   `Access-Control-Allow-Private-Network: true`, Chrome silently fails the request as a
   generic `TypeError: Failed to fetch` — no indication it's a PNA issue at all. `main.py`
   adds a small custom `PrivateNetworkAccessMiddleware` specifically to answer this header.
   **Middleware order matters here**: `CORSMiddleware` must be added first (innermost) and
   `PrivateNetworkAccessMiddleware` second (outermost), because Starlette's stack runs the
   last-added middleware first/outermost, and the PNA header has to be attached to the
   *final* response — including CORS's own short-circuited preflight `OPTIONS` response.

### Structured logging

Every request logs one JSON line (`logging_config.py`'s `JsonFormatter`) instead of a
free-text string — `{"timestamp": ..., "level": "INFO", "message": "predict lang=en label=hateful confidence=0.94 ..."}`.
"Structured" means machine-parseable: a log aggregator (or a local `jq` command) can
filter by `level=="ERROR"` or compute average latency across a run, without regex-scraping
plain text.

### Deployment shape (Dockerfile)

The backend ships as a self-contained Docker image (Hugging Face Spaces, per `README.md`'s
frontmatter): checkpoints (~770MB total) are baked directly into the image at build time
rather than downloaded at container start, because the target platform's free tier has no
persistent disk across rebuilds — baking them in means the container works immediately
with no external dependency at boot.

---

## Part 3 — The Chrome extension: four files, one pipeline

```
manifest.json   → declares what runs where, and what the extension is allowed to touch
background.js   → the ONLY piece of code that makes network calls
content_script.js → injected into every page; watches, extracts, batches, blurs
popup.js/html   → the toolbar popup UI (on/off switch + backend status)
```

### Why the network call lives in `background.js`, not `content_script.js`

This is the single most non-obvious design decision in the extension, and it's called
out explicitly in the code comments. A content script's `fetch()` executes as **the
visited page's own origin** — not the extension's origin. So a content script running on
`https://some-public-site.com` that tries to reach `http://localhost:8000` looks to Chrome
like *that public site* reaching into the user's local network. That triggers Chrome's
Local Network Access permission prompt, repeated **per site** (since a different "site" is
making the request every time), which is a broken UX for something meant to run silently
on every page.

A background service worker's origin is always `chrome-extension://<id>` — covered by the
extension's own declared `host_permissions`, and not subject to that per-site prompt. So
the content script never calls `fetch()` directly; it sends a `chrome.runtime.sendMessage`
to the background worker, which does the actual `POST /predict/batch` and relays the
result back.

```javascript
// content_script.js -> background.js
chrome.runtime.sendMessage({ type: "PREDICT_BATCH", items: [...] })

// background.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== "PREDICT_BATCH") return false;
  fetch(BATCH_ENDPOINT, { method: "POST", ... })
    .then(...).then(data => sendResponse({ ok: true, data }))
    .catch(err => sendResponse({ ok: false, error: String(err) }));
  return true; // keep the async channel open until sendResponse fires
});
```

### The problem `content_script.js` solves: pages change forever

A static page (view-source, everything present at load) would be easy — walk the DOM
once, classify every text block, blur the bad ones, done. Real feed-style pages
(Twitter/X, Reddit, any infinite-scroll site) are **dynamic**: new content streams in as
the user scrolls or the page's own JS re-renders, with no full page reload. "Scan once at
load" would miss almost everything the user actually sees. The content script has to keep
noticing new content indefinitely, without re-scanning the whole page repeatedly, without
flooding its own backend, and without janking scroll performance. Four browser mechanisms
work together to do this.

### Step 1 — Noticing new content: `MutationObserver`

```javascript
const mutationObserver = new MutationObserver(handleMutations);
mutationObserver.observe(container, { childList: true, subtree: true });
```

This fires whenever DOM nodes are added/removed under `container` — no polling needed,
the browser calls back automatically. `container` is `document.getElementById("feed") ||
document.body` (`findFeedContainer()`) — narrower than `document.body` where a feed
container id is known, since watching the entire page on a React-heavy site can fire
mutation callbacks constantly for reasons unrelated to new content (style recalculation,
focus changes).

**Debouncing**: mutations are collected into `pendingMutationNodes` and only actually
processed 200ms after the last mutation (`MUTATION_DEBOUNCE_MS`), so a burst of rapid
inserts (fast scrolling) is handled as one batch instead of triggering the whole pipeline
once per node.

### Step 2 — Extracting "one unit of text" per new node

There is **no per-site selector table** — no hardcoded `SITE_SELECTORS` dict, no
Twitter/X-specific `data-testid` lookup. Every page, seen for the first time or not, goes
through one generic scorer, `findContentBlocksGeneric()`, which ports the actual
node-scoring algorithm from **Mozilla's Readability.js** (the engine behind Firefox's
Reader View) rather than a hand-rolled approximation of it:

- **Tag base score** — every candidate (`p, div, li, td, span, article, section`) starts
  with a score from its tag: `DIV +5`, `PRE/TD/BLOCKQUOTE +3`, list/form tags `-3`,
  heading/`TH` tags `-5` (`TAG_BASE_SCORE`). Divs are more likely to be content wrappers;
  list/heading markup usually isn't the post body itself.
- **Class/id weighting** (`getClassWeight`) — className and id are each checked
  independently against four regexes ported directly from Readability's source:
  `UNLIKELY_CANDIDATES` (nav/banner/sidebar/comment/social/etc → `-25`, unless it also
  matches the `OK_MAYBE_ITS_A_CANDIDATE` exception list), `POSITIVE_PATTERN`
  (article/content/post/entry/etc → `+25`), `NEGATIVE_PATTERN`
  (footer/promo/widget/tags/etc → `-25`).
- **Comma + length scoring** (`scoreNode`) — `1 + commaCount + min(floor(textLength/100), 3)`.
  Real sentences accumulate commas and length; UI chrome ("Reply", "3 likes") doesn't.
- **Ancestor score propagation** (`propagateScoreToAncestors`) — a scored node's score
  isn't kept to itself: it's added to its parent in full, to its grandparent at half
  weight, and further ancestors at rapidly diminishing weight (`level * 3`). This is what
  lets a post whose text is split across several nested `<span>`s (common in React-built
  feeds) accumulate score at the shared post-container level, instead of scoring as many
  disconnected, individually-too-small fragments.
- **Link density** (`linkDensity`) — the fraction of an element's text that lives inside
  `<a>` tags, with hash-only/`javascript:` links discounted to 0.3x weight (they're usually
  UI chrome like "reply" or "···", not real link content). A block above `0.5` density is
  rejected outright as navigation, not prose.
- **Picking candidates: repeated-list detection first** (`isRepeatedListContainer`) —
  Readability.js is single-winner by design: across a whole page it picks exactly *one*
  "top candidate" region, because Reader View's job is "this page is one article, find the
  main content." A feed page is a different shape entirely: many structurally similar
  sibling posts under one shared container. Naively ranking every scored node by its
  (ancestor-propagated) score and picking greedily, top-down, breaks immediately here —
  propagation means the shared container accumulates the *sum* of every post's score, so
  it always outscores any individual post and would collapse N posts into a single result.
  Before accepting any node as a candidate, `pickTopCandidates` checks whether its children
  look like a repeated list: at least 3 children share a tag name, all score `> 0`, and
  their scores stay within a `2.5x` ratio of each other. If so, the node is treated as a
  list wrapper, not a candidate itself — each qualifying child is recursed into separately
  instead. This is a lightweight version of *data record extraction* (the same idea behind
  Liu/Grossman/Zhai's MDR algorithm for detecting repeated records in a page).
- **Picking the right specificity within one item** — once a subtree is confirmed to be a
  single item (not a repeated list), the real content node still isn't necessarily the
  outermost element: ancestor propagation means a wrapper `<div class="post">` always scores
  at least as high as its own best-scoring child (username `<div class="meta">` scores
  negative and is excluded; body `<div class="body">` scores positive) — so the wrapper
  looks like the top scorer even though the `.body` div is the real content. The selection
  walks from the outside in, and descends from a node into its single child only when
  **both** hold: (1) exactly one child clears a noise floor — `realScoredChildren` filters
  out children scoring far below the group's own strongest child (a relative threshold, not
  a fixed number, since real content can legitimately score as low as ~12 on a short tweet or
  as high as ~90+ on a long article; a fixed floor tuned for one scale wrongly discards real
  content at the other), so a trivial date-header or widget title sitting next to the one
  real child doesn't block descent — and if there's instead genuinely more than one
  substantial child (e.g. a tweet fragmented across multiple inline `<span>`s, or a caption
  sitting beside an embedded quote block), there's no single "more specific" child to prefer,
  so the current node is kept as-is, holding all of them together; and (2) the current node
  has no meaningful direct text of its own (its *own* text — not descendant text — scores
  `<= 0` or is empty), meaning it isn't contributing real content beyond that one child. This
  correctly resolves `.post` → `.body` (single scored child, `.post` has no own text) while
  correctly *not* over-descending into one span of a multi-span tweet (multiple scored
  children block descent) or losing a caption next to a quote block (also multiple scored
  children).
- **Splitting genuinely different regions apart** (`findDissimilarRegions`) — real pages
  aren't always "one item" or "a list of similar items." A classic multi-column blog layout
  (main article + one or more sidebars, each full of unrelated widgets) is neither: its
  children don't share a tag/score pattern (so `isRepeatedListContainer` correctly says no),
  and there's more than one substantial child (so the single-child descend rule correctly
  refuses to pick just one). Without a third rule, the whole layout wrapper would be accepted
  as a single giant candidate — confirmed on a real page, this produced a ~21,000-character
  "post" spanning the article, both sidebars, and footer widgets, which got blurred in its
  entirety from one hateful fragment anywhere inside it. `findDissimilarRegions` recurses
  into each child that both clears the noise floor *and* is itself a `isStructuralRegion` —
  a container that still has more than one of its own scored descendants (i.e. it isn't close
  to resolved yet), as opposed to a simple, already-resolved element like a caption `<p>` or
  an embedded `<blockquote>` (left alone for sibling-merging to handle instead). A region
  candidate is also rejected outright if its own tag/class score is negative — a "Subscribe
  to Posts/Comments" sidebar widget can still accumulate a positive *propagated* score from
  small descendants even though it's explicitly boilerplate (`NEGATIVE_PATTERN` matches
  "widget"), so propagated score alone isn't trusted for a container's own merit. This rule
  fires even for a single qualifying region (not just ≥2): once the noise floor excludes an
  irrelevant sibling, the one remaining real region still needs descending into, just reached
  by a different path than the original single-child case. Results reached this way are
  tracked separately and skipped by sibling-merging (below) — otherwise a high-scoring
  already-split region can get re-absorbed as a "merged sibling" of a different split region,
  silently undoing the split and producing duplicate, overlapping picks.
- **Sibling merging** — for each picked top candidate, sibling elements are pulled in if
  they clear Readability's sibling threshold (`max(10, topCandidateScore * 0.2)`), or, for
  `<p>` specifically, pass its length/link-density/sentence-ending-punctuation special case.
  This recovers text split by an inline ad, emoji image, or mention link sitting between
  two fragments of the same post.

**Deliberately not ported**: Readability's `charThreshold`/`isProbablyReaderable` gate and
its "retry with progressively looser filtering until ~500 characters is reached" loop. That
machinery exists to decide *whether Reader View should run on this page at all* for
long-form articles — applying it here would reject exactly the content this extension cares
about most, since a single tweet or comment is often under 100 characters. Only the scoring
primitives are reused; there's no length-based rejection.

### Step 3 — Only scanning what's actually about to be seen: `IntersectionObserver`

Extracted candidate elements aren't sent for scoring immediately — a feed page can hold
hundreds of posts in the DOM (virtualized lists keep off-screen content mounted for
smooth scrolling), most of which the user will never scroll to.

```javascript
const viewportObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      viewportObserver.unobserve(entry.target);   // scan once, then stop watching
      scoreElementIfTextPresent(entry.target);
    }
  },
  { rootMargin: "200px" }
);
```

`rootMargin: "200px"` grows the effective viewport by 200px on every side, so scoring
starts *slightly before* the user's eyes actually reach an element — avoiding a visible
flash of unblurred content followed by a jarring cover-up a moment later.

**The combined pipeline so far:**
```
MutationObserver:  "a new post-like element was inserted somewhere"
        │
        ▼
IntersectionObserver (attached to that element): "...and it's about to scroll into view"
        │
        ▼
extract its text → queue it for scoring
```

### Step 4 — Batching requests instead of firing one per post

A user scrolling quickly can trigger dozens of "about to be visible" events per second.
Firing one HTTP request per post would hammer the backend for near-zero benefit (most
scrolled-past posts are never really read).

```javascript
function queueForScoring(element, text) {
  pendingQueue.push({ id, text, language: detectLanguage(text), element });
  if (batchTimer) clearTimeout(batchTimer);
  batchTimer = setTimeout(flushQueue, BATCH_DEBOUNCE_MS);  // 200ms
}
```

Everything queued within a 200ms window gets flushed together as one
`PREDICT_BATCH` message (capped at `BATCH_MAX_ITEMS = 50` per request, matching the
backend's own `PredictBatchRequest` cap). A second safeguard, `MAX_CONCURRENT_BATCHES = 3`,
caps how many batch requests can be in flight at once — if that limit is hit, `flushQueue`
just reschedules itself instead of piling up more concurrent connections, so a sudden
scroll burst can't open unlimited simultaneous requests to the local server.

### Language detection per item

```javascript
function detectLanguage(text) {
  return /[ঀ-৿]/.test(text) ? "bn" : "en";
}
```
A cheap heuristic — presence of any character in the Bangla Unicode block routes that
item to the Bangla checkpoint; otherwise it's sent as English. This runs per queued item,
so a single batch can legitimately mix `"en"` and `"bn"` requests, each hitting its own
model server-side.

### Step 5 — Applying the blur, and matching results back to elements

```javascript
for (const result of response.data.results) {
  const element = elementById.get(result.id);
  if (element && BLUR_LABELS.has(result.label)) {
    applyBlur(element);
  }
}
```
`BLUR_LABELS = new Set(["hateful"])` — **only** the `hateful` label triggers a blur.
Text classified `sarcastic` (without also being independently hateful) is deliberately
left alone; the extension's purpose is hiding hate, not hiding sarcasm generally.

```javascript
function applyBlur(element) {
  element.classList.add("hs-blurred");
  blurredElements.add(element);
}
```
```css
.hs-blurred {
  filter: blur(8px);
  user-select: none;
  transition: filter 0.15s ease;
}
```
Plain CSS `filter: blur()` is applied directly to the matched text element (injected once
via `injectStyles()`), not an overlay — simple, cheap, works everywhere, and (per the
design doc) avoids `backdrop-filter`'s extra GPU cost when many items might be blurred
during fast scrolling.

**There is no click-to-reveal.** This extension's current behavior is deliberately
stricter than the original design doc's proposal: once something is blurred, the *only*
way to see it again is turning the extension off entirely from the popup — which reveals
everything currently blurred on the page. This is a real, intentional product decision,
not an oversight (see the top-of-file comment in `content_script.js`).

### The on/off toggle, and why re-enabling needs a full rescan

State lives in `chrome.storage.local` (`{ enabled: true|false }`), shared between the
popup and every content script instance, so flipping it takes effect immediately on
already-open tabs — not just on next page load.

```javascript
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local" || !("enabled" in changes)) return;
  isEnabled = changes.enabled.newValue;
  if (isEnabled) rescanWholePage();
  else revealAll();
});
```

- **Turning it off** (`revealAll()`) strips `.hs-blurred` from every currently-blurred
  element and clears them from `scoredNodes` too — not just visually unblurring them.
  This matters: if a post's scored status weren't cleared, `rescanWholePage()` on
  re-enable would see it in `scoredNodes` and skip it forever, leaving it permanently
  unblurred even though the model would still flag it.
- **Turning it back on** (`rescanWholePage()`) can't just rely on `IntersectionObserver`
  firing again — that API only fires on a *change* of intersection state, and posts
  already sitting in the viewport before re-enabling never register a "change." So
  re-enabling forces an immediate, direct scoring pass over everything already on the
  page that hasn't been scored yet, instead of waiting on a browser event that won't come.

Two separate `WeakSet`s track state per element: `observedNodes` ("is a viewport observer
already attached — don't attach twice") and `scoredNodes` ("has this actually been sent
for scoring at least once"). They're deliberately different: while disabled, an element
can become `observed` (the intersection observer still fires; `queueForScoring` just
no-ops on the disabled check) without ever becoming `scored` — re-enabling has to catch
those up directly rather than assuming "observed" implies "already handled." Both are
`WeakSet`s (not arrays) specifically so entries are garbage-collected once a node is
removed from the DOM (e.g. scrolled far away and unmounted by a virtualized list),
preventing an unbounded memory leak over a long browsing session. `blurredElements` is a
regular `Set`, not a `WeakSet`, because it needs to be actively iterated on toggle-off —
entries are removed on reveal, so it can't grow without bound either.

### Cleanup

```javascript
window.addEventListener("beforeunload", () => {
  mutationObserver.disconnect();
  viewportObserver.disconnect();
});
```
Disconnecting both observers on unload prevents an orphaned observer holding a live
reference to DOM subtrees that would otherwise be freed — a real memory-leak pattern in
long-lived single-page apps if skipped.

### The popup

`popup.html`/`popup.js` is the toolbar UI: a single on/off toggle (backed by
`chrome.storage.local`) and a live backend status check (`fetch("http://localhost:8000/health")`
— shown as "online (en, bn)" or "unreachable"). It has no other controls; there's no
per-site configuration UI, no sensitivity slider — the only lever the user has is the
whole-extension on/off switch.

### The bundled test page

`extension/test-page/index.html` is a self-contained fake feed (`.post` divs, scored and
picked up by the same generic `findContentBlocksGeneric` extractor used on every other
site — no special-casing) with buttons to inject new posts one at a time or in a
5-at-once burst — used to validate the MutationObserver → 
IntersectionObserver → batch → blur pipeline against a controlled page before ever
pointing the extension at a real site.

---

## Part 4 — The full request lifecycle, start to finish

```
1. Page loads. content_script.js runs at document_idle, calls start():
     - injects the .hs-blurred CSS rule
     - finds the feed container (a #feed element if present, or document.body)
     - scans whatever's already in the DOM (findContentBlocksGeneric)
     - attaches the MutationObserver to the container

2. New content appears (scroll-triggered insert, or the page's own JS re-renders):
     - MutationObserver's callback fires, debounced 200ms
     - each newly-added subtree is run through findContentBlocksGeneric
       (Readability-style scoring -- tag/class/comma/length scoring, ancestor
       propagation, sibling merging; no per-site selectors anywhere)
     - each match gets an IntersectionObserver attached (processElement)

3. The element scrolls near the viewport (rootMargin: 200px, i.e. slightly early):
     - IntersectionObserver fires, unobserves itself (one-shot)
     - its innerText is extracted and handed to queueForScoring
     - detectLanguage() picks "en" or "bn" per item

4. Within a 200ms window, all queued items are flushed as one batch
   (capped at 50 items, capped at 3 concurrent in-flight batches):
     - content_script.js -> chrome.runtime.sendMessage({ type: "PREDICT_BATCH", items })
     - background.js receives it, does the actual fetch (its origin avoids the
       per-site Local Network Access prompt a content script would trigger)
     - POST http://localhost:8000/predict/batch  { items: [{id, text, language}, ...] }

5. FastAPI (main.py):
     - Pydantic validates the whole batch shape before any code runs
     - each item runs through _run_inference(): tokenize -> model forward pass
       (torch.no_grad()) -> softmax over 3 classes + independent sigmoid per
       auxiliary head -> label + scores
     - one JSON response with per-item results, each still carrying its `id`
     - a structured JSON log line is written per item

6. background.js relays the response back to content_script.js via sendResponse

7. content_script.js matches each result's `id` back to its original DOM element
   (elementById map built at queue time) and, for any result labeled "hateful"
   specifically, adds the .hs-blurred class to that element

8. The user sees the covered text. The only way to reveal it is the popup's
   on/off toggle, which reveals everything currently blurred on the page.
```

---

## Known limitations, stated plainly

- **CORS is wide open** (`allow_origins=["*"]`) — fine for local development, but this and
  the hardcoded `http://localhost:8000` endpoints in `background.js`/`popup.js` would need
  to change before any real, non-local deployment.
- **The generic extractor is still a heuristic, not a certainty.** Porting Readability's
  real scoring math (tag/class weighting, ancestor propagation, sibling merging) is a
  meaningfully better generic extractor than a hand-rolled approximation, and removing
  per-site selectors means there's no per-site maintenance burden or per-site markup
  breakage risk anymore — but on a genuinely unusual page layout it can still pick the
  wrong block or merge in a neighboring caption. There's no per-site override left as an
  escape hatch if a specific site scores badly; the fix in that case is tuning the shared
  scoring constants, not adding a one-off selector.
- **No click-to-reveal / false-positive escape hatch beyond the global toggle.** Any model
  makes mistakes; right now the only recovery from a wrongly-blurred post is disabling the
  extension entirely (which un-blurs everything on the page, not just that one post).
- **Batch inference is still a Python loop, not a truly batched forward pass** —
  `predict_batch` calls `_run_inference` once per item rather than tokenizing all texts
  together and running one batched tensor through the model. Correct, but not the most
  compute-efficient version of this endpoint.
