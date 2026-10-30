# How the Hate & Sarcasm Blur Extension Works

A complete walkthrough of `hate-sarcasm-extension/` — what runs, in what order, and **why
each decision was made the way it was**. Where a number appears (a threshold, a timeout, a
speedup), it was measured, and the measurement is shown.

---

## The 60-second version

You install a Chrome extension. You browse normally. When a line of text is **hateful**, it
blurs behind a red wash — unreadable, and the more confident the model, the heavier the blur.
When a line is merely **sarcastic**, it gets a yellow tint and stays fully readable: that
flag is a note about tone, not something to hide. If it gets either one wrong, you press and
hold the line and tell it the truth — and it remembers, immediately and permanently.

Two halves make that happen, talking over HTTP on your own machine:

```
┌──────────────────────────────────┐              ┌──────────────────────────────────┐
│  extension/     (Chrome, JS)     │              │  backend/       (FastAPI, Python)│
│                                  │ ──HTTP POST─▶│                                  │
│  • watches the page for text     │              │  • holds 2 ML models in RAM      │
│  • decides what to send, when    │              │  • scores text → 3 probabilities │
│  • blurs / tints the results     │◀──JSON───────│  • remembers your corrections    │
└──────────────────────────────────┘              └──────────────────────────────────┘
```

The extension is JavaScript and cannot run a PyTorch model. The backend is Python and
cannot see the page. Neither half is optional.

---

## Table of contents

| Part | What it covers |
|---|---|
| [Glossary](#glossary) | Terms used throughout, defined once |
| [Part 1 — The model](#part-1--the-model) | What is actually doing the classifying, and a defect you must know about |
| [Part 2 — The backend](#part-2--the-backend) | The server, its endpoints, and why it exists |
| [Part 3 — The extension](#part-3--the-extension) | Finding text on a live page and marking it |
| [Part 4 — Teaching it](#part-4--teaching-it-corrections) | Correcting mistakes, and how corrections are matched |
| [Part 5 — Performance](#part-5--performance) | What made it ~2× faster without changing any answer |
| [Part 6 — Full lifecycle](#part-6--the-full-lifecycle-end-to-end) | One trip through the whole system |
| [Part 7 — Limitations](#part-7--known-limitations-stated-plainly) | What it does not do well |
| [Appendix](#appendix--reference-tables) | File map, constants, endpoints |

Read top to bottom once; each part assumes the one before it.

---

## Glossary

Terms that recur everywhere. If a sentence later stops making sense, it's probably one of
these.

| Term | Meaning here |
|---|---|
| **Token** | A word-piece. The model doesn't see letters or words, it sees numbered chunks. `শুয়োরের` becomes 5 tokens. |
| **`[CLS]` vector** | 768 numbers summarising a whole sentence, produced by the encoder. Everything downstream reads *this*, not the text. |
| **Logit** | A raw, unbounded model output, before being squashed into a probability. |
| **Softmax** | Turns several logits into probabilities that **sum to 1** — for "pick exactly one of these." |
| **Sigmoid** | Turns one logit into a 0–1 score **independently** — for "yes/no, unrelated to the others." |
| **Head** | A small network on top of the shared encoder answering one narrow question. |
| **Checkpoint** | A `.pt` file of trained weights. Meaningless without matching code. |
| **Line** | This project's unit of text: one visual line/sentence on the page. Everything is scored per line. |
| **Normalization** | Cleaning text to match training (strip URLs, emoji, list numbering). |
| **Correction / taught example** | A label you supplied by hand, stored server-side, overriding the model. |

---

## Part 1 — The model

> **In one sentence:** one shared language encoder feeds two specialist heads and a fusion
> layer, which outputs three competing probabilities — and in the shipped weights, one of
> those two heads is broken.

### The architecture

The class is `LightweightDualHeadModel` (`backend/app/architecture.py`). It is **not** a
plain "text in, label out" classifier:

```
              input text
                  │
                  ▼
      ┌───────────────────────┐
      │  DistilBERT encoder   │   general language understanding,
      │  (6 transformer       │   pretrained before this project
      │   layers)             │
      └───────────┬───────────┘
                  ▼
        [CLS] vector — 768 numbers summarising the sentence
                  │
        ┌─────────┼─────────┐
        ▼         ▼         │
   ┌─────────┐ ┌─────────┐  │
   │hate_head│ │sarc_head│  │  two specialists, 768 → 128 → 1
   │"hateful?"│ │"sarcastic?"│ │
   └────┬────┘ └────┬────┘  │
        │  128-dim  │       │  their *hidden* layers, not just their answers,
        └─────┬─────┴───────┘  are what gets passed on
              ▼
      ┌───────────────────┐
      │    fusion_mlp     │   sees [CLS] + both hidden layers  (768+128+128 → 256 → 3)
      └─────────┬─────────┘
                ▼
   non_hateful   |   hateful   |   sarcastic      ← three probabilities summing to 1.0
```

**Why this shape.** Sarcasm is often the *delivery mechanism* for hate ("oh sure, because
[group] are SO trustworthy"). A single generic classifier tends to learn "sarcastic ⇒
harmless," because most sarcasm in ordinary training data *is* harmless. Giving the model a
dedicated hate branch and a dedicated sarcasm branch, then fusing both, is meant to let it
reason about *tone* and *toxicity* separately instead of making one signal stand in for
both. That premise is the project's central research question.

### Two separate models, not one bilingual one

```python
# backend/app/config.py
CHECKPOINTS = {
    "en": {"backbone": "distilbert-base-uncased"},
    "bn": {"backbone": "distilbert-base-multilingual-cased"},
}
```

English and Bangla each have their **own fully trained copy**, with different backbones.
This is not cosmetic: sending Bangla through the English checkpoint won't error, it will
just return meaningless numbers, because those weights never saw Bangla. The extension's
`detectLanguage()` exists to route each line to the right one.

### What a checkpoint actually is

A `.pt` file is a dictionary: layer name → tensor of trained numbers
(`"hate_head.0.weight"`, `"fusion_mlp.3.bias"`, …). It carries **no** information about
what those numbers *do*. Loading is always two steps:

1. Build `LightweightDualHeadModel(backbone)` — correct *shapes*, untrained values.
2. `load_state_dict(...)` — overwrite every layer by matching name and shape.

> ⚠️ **`architecture.py` must stay structurally identical to the training code — not
> "equivalent," not "tidied up."** `forward()` deliberately calls `hate_head[0]` and
> `hate_head[3]` by index rather than `self.hate_head(x)`, because that is exactly how
> training applied dropout. A cleaner rewrite would load **without any error** and silently
> compute different numbers. This is the most dangerous class of bug in the repo: it doesn't
> crash, it just makes everything subtly wrong.

### Turning raw output into a verdict

```python
with torch.no_grad():
    logits, hate_logit, sarc_logit, cls = model(ids, mask, return_cls=True)
    probs = torch.softmax(logits, dim=-1)[0]   # 3 numbers, summing to 1.0
    label_id = int(probs.argmax())             # the winning class
```

Two squashing functions, deliberately different:

- **Softmax on the 3-class output** — `non_hateful` / `hateful` / `sarcastic` are treated as
  mutually exclusive, so they must compete and sum to 1.
- **Sigmoid on each auxiliary head** — *intended* as two independent yes/no questions, each
  free to score high at the same time.

> ### ⚠️ Correction: the auxiliary heads are broken in every shipped checkpoint
>
> The design above is the intent. It is **not** what the shipped weights do. The training
> loop optimised *both* sigmoid heads against the *sarcasm* label:
>
> ```python
> lh = batch['labels_hate'].to(device)     # loaded, moved to GPU, then never used
> loss = ALPHA*crit_b(hl, ls) + BETA*crit_b(sl, ls) + GAMMA*crit_3(logits, l3)
> #                     ^^ should be lh
> ```
>
> Measured over 3,000 held-out rows:
>
> | | Bangla | English |
> |---|---|---|
> | `corr(hate_head, sarc_head)` | **0.9997** | **0.9999** |
> | `corr(hate_head, y == hateful)` | **−0.36** | **−0.48** |
>
> `is_hateful_score` is therefore *a second sarcasm score that anti-correlates with real
> hate*. **Never** use it for scoring, thresholding, or display.
>
> This is worse than one bad output field. `fusion_mlp` consumes both heads' hidden layers,
> so the 3-class classifier the extension actually reads received the sarcasm signal twice
> and the hate signal never — meaning **RQ1 is unanswered by these checkpoints**.
>
> Use `hate_prob` / `sarc_prob` / `non_hateful_prob` instead; those come from the fused
> softmax and are sound. Full analysis in
> [MODEL_DEFECTS_AND_DETECTION.md](MODEL_DEFECTS_AND_DETECTION.md); the fix is in
> `lightweight-model-corrected.ipynb`.

### `no_grad()` and `eval()`

- `torch.no_grad()` — skip building the graph needed for backpropagation. Nothing is being
  trained; tracking gradients is pure wasted memory and time.
- `model.eval()` (set once at load, in `model_registry.py`) — disables `Dropout`, which
  randomly zeroes activations *during training only*. Forget it and the model still "works"
  but gives a different answer to the same input every time.

---

## Part 2 — The backend

> **In one sentence:** a small FastAPI server that loads both models once and answers
> "score this text" over HTTP.

### Why it exists

A Chrome extension is JavaScript in a sandbox. It cannot `import torch`, cannot open a
`.pt` file, cannot allocate a 500 MB tensor. The backend's whole job is to hold the models
in RAM and expose scoring as an HTTP call.

### Loading once, at startup

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load_all()          # both tokenizers + both checkpoints, once
    feedback_store.load()        # replay saved corrections
    threading.Thread(target=feedback_store.warm_index, args=(_embed,), daemon=True).start()
    yield
```

If loading happened per request, every call would re-read hundreds of MB from disk and
rebuild DistilBERT — turning a ~25 ms request into several seconds, every time. The
registry is a singleton holding the loaded models for the process lifetime.

Note the **daemon thread**: corrections restored from disk need their `[CLS]` vectors
recomputed for similarity matching. That runs in the background, so a large accumulated
file never delays the server accepting traffic. Exact matching works before it finishes.

### The endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Alive? which models loaded? how many corrections stored? |
| `GET /model-info` | Architecture description and backbone per language |
| `POST /predict` | Score one line |
| `POST /predict/batch` | Score up to 50 lines in one round trip |
| `POST /feedback` | Teach one correction |
| `GET /feedback` | List all stored corrections |
| `DELETE /feedback` | Undo one correction |

### Validation order, and why the status codes differ

```python
if not text:                              raise HTTPException(422)  # bad shape
if req.language not in SUPPORTED_LANGUAGES: raise HTTPException(400)  # bad value
try:    model, tokenizer = registry.get(req.language)
except KeyError:                          raise HTTPException(503)  # our fault, not yours
```

422 and 400 are both the *client's* fault (malformed vs. semantically invalid). 503 means
the **server** failed to load a language it advertises — a different failure needing a
different response from the caller.

Pydantic validates before the function body runs at all: `text: str = Field(...,
min_length=1, max_length=10000)`. A malformed request never reaches the model.

The optional **`id` field** exists purely for the extension: when 50 results come back, it
needs to know which DOM element each belongs to without trusting array order. It assigns an
id per queued line and matches on the way back.

### Two browser-security mechanisms, easy to confuse

1. **CORS** — browsers block JS from *reading* a cross-origin response unless the server
   opts in. `CORSMiddleware` handles it (`allow_origins=["*"]` for local dev only).
2. **Private Network Access (PNA)** — a *separate*, newer Chrome check. A request from a
   "more public" origin into `localhost` triggers an extra preflight header,
   `Access-Control-Request-Private-Network: true`, that `CORSMiddleware` doesn't know how
   to answer. Without a reply, Chrome fails the request as a generic
   `TypeError: Failed to fetch` — **with no hint that PNA is the cause**. A small custom
   `PrivateNetworkAccessMiddleware` answers it.

> **Middleware order is load-bearing.** Starlette runs the *last-added* middleware
> outermost. `CORSMiddleware` is added first (innermost) and PNA second (outermost), because
> the PNA header must be attached to the *final* response — including CORS's own
> short-circuited `OPTIONS` preflight, which it returns without calling inward. Reverse the
> order and the header never gets added to preflights at all.

### Text normalization — the train/serve gap

`preprocessing.py` exists because **the model was trained on cleaned text, and the
extension sends raw page text.** It ports Phase 2's cleaning (strip URLs, mentions,
hashtags, HTML entities, emoji; NFC-normalize Bangla) and adds one thing training never
needed:

```python
strip_list_prefix("৭। খাঙ্কিরপোলা")  →  "খাঙ্কিরপোলা"
```

Real pages number their lines. Training text never did, so a leading `৭। ` is
out-of-distribution noise that measurably moves the decision:

| text | verdict |
|---|---|
| `খাঙ্কিরপোলা` | hateful, P = 0.529 |
| `৭। খাঙ্কিরপোলা` | **non_hateful**, P(hate) = 0.454 |

Stripping it moved recall on the reported list from 17.5% → 26.3% at no false-positive
cost. The guard: if stripping would leave no letters behind (`"৭।"` alone, or `"1. 2. 3."`),
the original is kept, so a purely numeric line is never destroyed.

---

## Part 3 — The extension

> **In one sentence:** watch the page forever, extract text line by line, score only what's
> about to be seen, and mark the results.

```
manifest.json      → permissions and what runs where
background.js      → the ONLY code that makes network calls
content_script.js  → injected into every page: watches, extracts, batches, marks
popup.html/js      → on/off switch, backend status, correction count
```

### Why the network call lives in `background.js`

The single least obvious decision in the extension.

A content script's `fetch()` executes as **the visited page's origin**, not the extension's.
So a content script on `https://example.com` calling `http://localhost:8000` looks to Chrome
like *example.com* reaching into your local network — which triggers the Local Network
Access prompt, **once per site**, forever. Unusable for something meant to run silently
everywhere.

A background service worker's origin is always `chrome-extension://<id>`, covered by the
extension's own `host_permissions`, and exempt from that prompt. So the content script never
fetches; it messages the worker, which fetches and relays the result back.

### The problem being solved: pages never stop changing

A static page would be easy — walk the DOM once, score everything, done. Real feed pages
stream new content in forever, with no reload. "Scan once at load" would miss nearly
everything. Four browser mechanisms cooperate:

```
MutationObserver      → "new text was inserted somewhere"
        ▼
TreeWalker            → "here are the text nodes, in document order"
        ▼
line grouping         → "these nodes form one line"
        ▼
IntersectionObserver  → "...and it's about to scroll into view"
        ▼
debounced batch queue → POST /predict/batch
        ▼
blur / tint / leave alone
```

### Step 1 — Noticing new content

```javascript
mutationObserverRef = new MutationObserver(handleMutations);
mutationObserverRef.observe(container, { childList: true, subtree: true });
```

`container` is `document.getElementById("feed") || document.body` — narrower where a feed
container exists, since watching all of `document.body` on a React-heavy site fires
constantly for reasons unrelated to new text.

Mutations are **debounced 200 ms** (`MUTATION_DEBOUNCE_MS`), so a burst of inserts during
fast scrolling is handled once, not once per node.

### Step 2 — Extracting one *line* of text

> **This section was rewritten. An earlier version of the extension ported Mozilla's
> Readability.js scoring — tag weights, class/id regexes, ancestor score propagation,
> repeated-list detection, structural-region splitting. All of it has been deleted.** The
> explanation of why is worth keeping, because it is the main architectural lesson here.

**What was tried first.** Element-level scoring: rank DOM elements and pick "the one right
element" per post. It kept breaking in a specific, repeating way — fixing a real page's edge
case (a Facebook post rendered as ~50 sibling `<div>` lines) broke a previously working case
(a tweet fragmented across sibling `<span>`s), and vice versa, across many iterations.

**Why it kept breaking.** Element-level scoring is *inherently ambiguous about what counts
as one unit*. Ancestor propagation makes a shared container accumulate the sum of every
child's score, so it always outranks any individual post — which is correct for Reader
View's job ("this page is one article") and exactly wrong for a feed ("this page is 50
independent posts").

**What replaced it,** confirmed against a mature production analogue
([FrostCo/AdvancedProfanityFilter](https://github.com/FrostCo/AdvancedProfanityFilter), which
solves the closely analogous problem of finding and hiding bad words in arbitrary live page
text): **walk text nodes directly, not elements.**

```javascript
const walker = document.createTreeWalker(root,
  NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT, { acceptNode(node) { ... } });
```

Filtering collapses to a short skip-list (`SCRIPT`, `STYLE`, `INPUT`, `BUTTON`,
`contenteditable`, …) instead of a multi-rule scoring system, because at the text-node level
a stray `"127"` or `"Reply"` simply never matches anything and is harmless to send.

#### Skipping page furniture — the single biggest accuracy win

The claim above ("a stray `Reply` is harmless") turned out to be **the most costly wrong
assumption in the project.** Chrome text is not harmless: the model scores it, and scores it
badly. On the real test page, 471 of 652 extracted lines were English page furniture, and
**41 of them were flagged** — against 8 false flags on ordinary benign prose.

So navigation, sidebars and footers are now pruned before anything is scored:

```javascript
const SKIP_REGION_TAGS  = new Set(["NAV", "ASIDE", "FOOTER"]);
const SKIP_REGION_ROLES = new Set(["navigation", "complementary", "contentinfo"]);
```

ARIA roles are checked alongside the tags because a large share of real sites build these
regions from `<div role="navigation">` rather than the semantic element.

`FILTER_REJECT` prunes the entire subtree, so a whole `<nav>` costs one check rather than
one per text node inside it. Measured on the real page:

| | before | after |
|---|---|---|
| lines sent to the model | 652 | **185** (71.6% less work) |
| English false positives | 41 | **9** |
| article content lost | — | **0** |

**Why this matters more than any model work.** Every adaptation method tried on top of the
classifier — linear probe, DFR reweighting, AFR, BitFit, partial-layer fine-tuning, cache
models — moved that 41 by single digits at best. The classifier was never the problem: this
text should not have reached it. One DOM rule beat all of them, and made the page 3.5×
cheaper at the same time.

**The safety guard.** A region is only treated as furniture if it does **not** contain the
page's main content landmark:

```javascript
return element.querySelector("main, article, [role='main']") === null;
```

Some templates wrap the whole page in a mis-chosen landmark, and pruning that subtree would
silently drop the very content the extension exists to check. Verified on the real page:
all 39 numbered slur entries and every article chunk still sent. The lines that *did*
disappear were the sidebar's "Popular Posts" widget — truncated copies of other posts,
which is exactly what should be skipped.

**Grouping nodes into lines.** A new line starts when either:

- **the nearest block-level ancestor changes** — this is what merges a tweet's
  `<span>`-fragmented sentence back into one string, since sibling inline elements share a
  block parent; or
- **a `<br>` is encountered** — which is why the walker requests `SHOW_ELEMENT` as well as
  `SHOW_TEXT`. Hand-formatted pages (a Word-paste blog post) often render list entries as
  inline elements separated only by `<br>` under one wrapper, with no per-entry element at
  all. Without seeing `<br>`, all 175 entries would merge into one enormous "line."

**Keeping direct references.** Each line keeps the live DOM nodes it came from, so marking a
result never has to search the page for matching text.

> **A real bug this caused, worth understanding.** State was originally tracked per
> *element*. On the `<br>`-separated page above, many distinct lines share the same wrapper
> `<span>` as their parent — so scoring line 1 marked that span as "scored," and lines 2–50
> were silently skipped as already handled. The fix: track "scored" per **text node** (which
> by construction belongs to exactly one line), and keep elements only for applying CSS,
> since a text node can't take a class. The same fix applies to `linesByAnchor`, which maps
> an anchor element to a **list** of lines rather than one.

`MIN_LINE_LENGTH = 3` is the only length filter — a real slur can be 4 characters, so
anything stricter would discard exactly the content that matters most.

### Step 3 — Only scoring what's about to be seen

```javascript
const viewportObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    viewportObserver.unobserve(entry.target);        // one-shot
    for (const line of linesByAnchor.get(entry.target) || []) queueLineIfDue(line);
  }
}, { rootMargin: "200px" });
```

A feed can hold hundreds of posts in the DOM that the user will never reach. `rootMargin:
"200px"` starts scoring slightly *before* the text is visible, avoiding a flash of readable
content followed by a jarring cover-up.

### Step 4 — Batching

Fast scrolling can trigger dozens of "about to be visible" events per second. One HTTP
request each would hammer the backend for nothing.

Everything queued within `BATCH_DEBOUNCE_MS = 200` flushes as one request, capped at
`BATCH_MAX_ITEMS = 50` (matching the backend's own limit) and `MAX_CONCURRENT_BATCHES = 3`
in flight. On failure, items are **put back on the queue** and retried with exponential
backoff up to 30 s — a dropped message must not silently skip posts forever.

> **A subtle timer bug worth knowing.** A fired `setTimeout` does not clear the variable
> holding its id. `batchTimer` must be nulled *at the top of the callback*, or every later
> "is a flush already scheduled?" check reads the stale id and returns false forever. The
> symptom: the first 50 lines score, and leftovers never get scheduled again.

### Language detection

```javascript
const detectLanguage = (text) => /[ঀ-৿]/.test(text) ? "bn" : "en";
```

Any character in the Bangla Unicode block routes that line to the Bangla checkpoint. Runs
per line, so one batch can legitimately mix both languages; the backend groups them.

### Step 5 — Marking the result

Decisions are made on `hate_prob` / `sarc_prob` — **never** on `is_hateful_score` (see the
Part 1 warning) and **never** on the argmax `label` alone. Two reasons:

1. The 3-class softmax forces hate and sarcasm to compete for probability mass, but the two
   genuinely co-occur. Splitting probability between them can leave *neither* winning the
   argmax even when their sum is decisive.
2. `sarcastic` acts as a catch-all for vulgar text the model wasn't trained on — Bangla
   threats score `sarcastic` at 0.886. Under the old `BLUR_LABELS = new Set(["hateful"])`
   rule, every one of those rendered completely in the clear.

#### Intensity means conviction *above the bar*

```javascript
function intensityAbove(prob, threshold) {
  if (prob <= threshold) return 0;
  return (prob - threshold) / (1 - threshold);
}
```

Mapping raw probability to blur directly would render a 0.51 detection at 51% of max blur
and a 0.90 at 90% — nearly indistinguishable. Rescaled from `[threshold, 1]` onto `[0, 1]`,
they land at **2%** and **80%**. That difference is the entire point of the feature.

Blur radius and tint alpha are written as **CSS custom properties per element**, not bucketed
class names (`hs-blur-1`…`hs-blur-5`), because bucketing would quantize away exactly the
gradient being visualised.

```css
.hs-flagged {
  filter: blur(var(--hs-blur, 8px));
  background-color: rgba(var(--hs-tint-rgb), var(--hs-tint-alpha, 0.3));
  user-select: none;
}
.hs-flagged.hs-hate { --hs-tint-rgb: 220, 38, 38; }   /* red    */
.hs-flagged.hs-sarc { --hs-tint-rgb: 234, 179, 8; }   /* yellow */
```

| | hate | sarcasm (bn) | sarcasm (en) |
|---|---|---|---|
| threshold | 0.50 | **0.85** | **0.95** |
| blur radius | 4 → 14 px | **none** | **none** |
| tint | red | yellow | yellow |
| text stays readable | no | **yes** | **yes** |

**Only hate blurs. Sarcasm is marked with colour alone.** The two flags mean different
things: hate is content the user asked not to see, while sarcasm is a note about *tone* — and
tone can't be judged without reading the words, so blurring the thing being commented on
defeats the comment.

It also keeps the failure mode proportionate. The English sarcasm head is saturated, so some
yellow flags are simply wrong; a wrong tint is a harmless annotation, whereas a wrong blur
hides text the user wanted. Hate has no equivalent problem — zero English false positives on
the real test page — so it keeps the real blur.

Sarcastic lines also stay **selectable and copyable**: the base `.hs-flagged` rule sets
`user-select: none` to stop blurred text being copied out, and `.hs-sarc` restores it, since
nothing is hidden there. The rule uses `filter: none` rather than a 0 px blur so no
compositing layer is created at all — `blur(0px)` still promotes the element and can subtly
change text antialiasing on a line meant to look untouched.

**The sarcasm bar is per-language.** Hate is not — it behaves comparably in both checkpoints
(93 flags on the real test page, all Bangla, zero English false positives). Sarcasm does not,
and the two curves look nothing alike; see [the next section](#why-sarcasm-is-held-to-a-much-stricter-bar).

#### Where these thresholds live, and what they touch

All of them are **defined once and read once**, in `extension/content_script.js`. Nothing in
the backend knows about them — the server returns raw probabilities and takes no view on what
counts as "flagged."

```
content_script.js:89    const HATE_THRESHOLD = 0.5
content_script.js:144   const SARC_THRESHOLD_BY_LANGUAGE = { bn: 0.85, en: 0.95 }
content_script.js:147   const SARC_THRESHOLD_FALLBACK   = 0.95   ← unknown language: stricter
content_script.js:149   function sarcThreshold(language)
                    │
                    ▼  the ONLY place any of them is read  (inside applyFlag)
content_script.js:621   intensityAbove(result.hate_prob, HATE_THRESHOLD)
content_script.js:625   intensityAbove(result.sarc_prob, sarcThreshold(result.language))
                    │
                    ▼  rescales [threshold, 1] → [0, 1]      (helper at :182)
                 hateIntensity / sarcIntensity
                    │
                    ▼  :635   const isHate = hateIntensity >= sarcIntensity   ← winner
              intensity  ──┬──▶ :639      blur px, HATE ONLY (sarcasm gets 0)
                           ├──▶ :640      tint alpha lerp(TINT_MIN_ALPHA, TINT_MAX_ALPHA)
                           ├──▶ :642      tooltip percentage
                           └──▶ :654-658  writes --hs-blur, --hs-tint-alpha,
                                          .hs-flagged + .hs-hate / .hs-sarc
```

**To change a threshold, edit one line.** Everything downstream — blur, tint, colour, tooltip
percentage — derives from `intensity`, so no other constant needs touching.

Two details worth noting:

**The language comes off the *response*, not from re-detecting it client-side** (`:618`). The
server is the authority on which checkpoint actually ran, and the threshold is a property of
that checkpoint's failure mode — so re-running `detectLanguage()` here would risk applying
Bangla's bar to a score the English model produced.

**The winner is decided on rescaled intensity, not raw probability** (`:635`). A line at
`hate = 0.55, sarc = 0.96` has hate intensity `(0.55−0.50)/0.50 = 0.10` and sarcasm intensity
`(0.96−0.95)/0.05 = 0.20`, so it renders **yellow** — but at `sarc = 0.955` it would render
**red**, despite `sarc_prob` being nearly double `hate_prob`. Each signal is judged only
against its own bar. This is easy to misread as a bug when eyeballing raw response JSON.

#### Why sarcasm is held to a much stricter bar

It's the harder problem in principle — sarcasm rides on context, tone and shared knowledge
rather than lexical cues, so a short page line often doesn't contain the evidence to call it
at all. And in these checkpoints it's a catch-all: source leakage means the model learned to
separate its *datasets*, not the two phenomena, so out-of-distribution text drifts there
regardless of content.

##### Bangla — 0.85, a clean knee

Measured against `ood_eval_bn.csv` plus 20 benign Bangla lines:

| sarcasm bar | benign wrongly tinted | threats caught **only** by the sarcasm rule |
|---|---|---|
| 0.50 | 2/41 (4.9%) | 8/49 (16.3%) |
| 0.70 | 1/41 (2.4%) | 7/49 (14.3%) |
| 0.80 | 1/41 (2.4%) | 5/49 (10.2%) |
| **0.85** | **0/41 (0.0%)** | **5/49 (10.2%)** |
| 0.90 | 0/41 (0.0%) | 4/49 (8.2%) |

**0.85 is the knee, and the only step on this curve that is free** — it clears the last benign
false positive (`ট্রেন সকাল আটটায় ছাড়বে`, 0.840) while no genuine threat scores between 0.80
and 0.85. Earlier steps were *not* free: 0.70 → 0.80 gave up two real threats. Benign Bangla
sits well below the bar — `আজকে আবহাওয়া খুব সুন্দর` ("the weather is lovely today") scores
**0.700**.

##### English — 0.95, and the curve has no knee at all

| sarcasm bar | benign English wrongly tinted | UI chrome wrongly tinted | true sarcasm caught |
|---|---|---|---|
| 0.85 | 8/20 (40%) | 11/18 (61%) | 5/5 (100%) |
| 0.90 | 7/20 (35%) | 11/18 (61%) | 5/5 (100%) |
| **0.95** | **7/20 (35%)** | **10/18 (56%)** | **5/5 (100%)** |
| 0.99 | 5/20 (25%) | 9/18 (50%) | 4/5 (80%) |

**The English sarcasm head is saturated.** Ordinary declarative sentences score at the very
top of the range:

```
0.998  "He works as an engineer."      1.000  "Menu"
0.998  "Thanks for sending the report." 0.999  "Posts"
0.997  "It rained all afternoon."       0.999  "2.4k"
```

The distribution is **bimodal at ~0.00 and ~0.99**, so no threshold cleanly separates the
classes. 0.95 is a real but **partial** mitigation — it does remove the 0.85–0.95 band
(`TOP` 0.877, `2d ago` 0.907, `8y ago` 0.946 all stop being tinted) while still catching 100%
of genuine sarcasm — but it cannot reach the saturated majority.

> ⚠️ **Corrections do not scale here either.** Teaching `16M` and `1M` left **18 of 20**
> unseen chrome strings still flagged (`2.4k`, `500K`, `3M`, `r/pics`, `3mo ago`, …). That is
> the length-dependent similarity bar working correctly: strings under 16 characters need
> 0.98 cosine, and `16M` is not that close to `2.4k`. Corrections are the right tool for a
> *specific miss*; this is a *systematic* model failure. The real fixes are the corrected
> checkpoint and skipping non-content page regions — see
> [Limitations](#part-7--known-limitations-stated-plainly).

Hate keeps the lower 0.50 bar in both languages: a missed slur costs more than missed irony,
and hate shows no equivalent saturation.

#### One signal wins outright

Both are *scored* independently, but whichever clears its threshold by more owns the **entire
treatment**: colour, blur radius, tint alpha, tooltip.

An earlier version mixed them — hue from the dominant signal, blur from
`Math.max(hateBlur, sarcBlur)`. Because the hate blur range (4–14 px) sits entirely above the
sarcasm range (0.5–3 px), any trace of hate above threshold dragged the blur into hate
territory while the tint stayed yellow: a line that read "sarcastic" but was blurred as hard
as hate. Reading one signal end to end is what makes the colour mean anything.

Every flagged element carries a `title` — `hateful: 90% — hold to review`, or
`hateful: 100% (you marked this)` when the verdict came from a correction rather than the
model.

### Step 6 — Reviewing one line (press and hold)

Blurred text is `user-select: none`, so a false positive could not even be *selected*, let
alone corrected. The reveal is narrowed rather than removed:

**Press and hold a flagged line for 450 ms** → it un-blurs, loses its tint, becomes
selectable, and keeps a dashed outline in its original colour. A toolbar appears beneath it.

| button | effect |
|---|---|
| **Hateful** | stores the correction, re-blurs **red** at full intensity |
| **Sarcastic** | stores the correction, re-blurs **yellow** at full intensity |
| **Neither** | stores the correction, stays **permanently un-blurred** |

Reclassification is one click: a line the model called sarcastic that is really hate turns
red. "Neither" is the only button that doesn't re-blur — re-hiding text the user just
declared fine would undo the correction in front of them.

**Only three buttons exist because the model is 3-class.** "Not hateful" and "not sarcastic"
are *the same stored label*; offering both would be two buttons that secretly do the same
thing.

#### Why a hold, and not a double-click

The gesture was originally double-click. It is unusable on flagged text inside a link — and
on a feed page, much of it is. `dblclick` fires only **after** two complete click events, so
the browser has already followed the `href`; `preventDefault()` on the `dblclick` is far too
late.

A hold is intercepted at the source: the review opens on a timer during `mousedown`, and the
`click` that follows is swallowed in the **capture phase**, before the link's navigation or
the page's own handlers can act. That click-suppression is the entire reason the gesture is
a hold.

Three guards keep it from misfiring:

- **Movement past 6 px cancels the hold**, so starting a text selection never opens a review.
- **The suppression flag clears on the next `mousedown`**, so a hold whose click never
  arrived (pointer released off-target) can't swallow a later, unrelated click.
- **A hold on an already-revealed line does nothing**, leaving press-and-drag free to select.

Verified in a real DOM against a flagged `<a href>`: short click navigates normally, hold
reveals without navigating, drag does neither, and a subsequent ordinary click elsewhere is
unaffected.

Also load-bearing:

- **Any outside press, or Escape, ends the review** and re-blurs.
- **A press *inside* the revealed line does not** — that's the user dragging to select, and
  re-blurring mid-drag would make selection impossible.
- **A selection, if made, is what gets marked** — not the whole line. Teaching a whole line
  when only two words are the slur teaches a lot of irrelevant context with it.

The toolbar lives in a **closed shadow root**: broad page selectors (`div button {...}`)
routinely restyle or hide injected UI sharing the page's style scope, and a closed root also
keeps page scripts out.

### The on/off toggle, and why re-enabling needs a full rescan

State lives in `chrome.storage.local`, shared between popup and every content script, so
flipping it takes effect on already-open tabs immediately.

- **Off** (`revealAll()`) clears the classes *and* removes those lines' text nodes from
  `scoredNodes`. If it only unblurred visually, re-enabling would see them as already scored
  and skip them forever — leaving genuinely hateful text permanently exposed.
- **On** (`rescanWholePage()`) cannot rely on `IntersectionObserver` firing again, because
  that API only fires on a *change* of intersection state, and lines already sitting in the
  viewport never register one. So re-enabling forces an immediate direct pass.

Two `WeakSet`s track different things on purpose: `observedNodes` ("an observer is already
attached") and `scoredNodes` ("this has actually been sent"). While disabled a line can
become observed without ever becoming scored. Both are `WeakSet`s so entries are collected
when a virtualized list unmounts a node — otherwise a long browsing session leaks.
`blurredLines` is a plain array because it must be iterated on toggle-off; it's cleared on
reveal, so it can't grow without bound either.

---

## Part 4 — Teaching it (corrections)

> **In one sentence:** mark a line once, and every later occurrence is flagged instantly, on
> every page, with no retraining and no reload.

The Bangla checkpoint misses most out-of-distribution slurs (**38.8%** recall on the OOD
fixture vs **91.4%** in-distribution). Retraining is the real fix, but it's offline and
GPU-bound. Corrections are the online half.

```
hold a flagged line  ──▶ toolbar button  ─┐
                                          ├─▶ sendFeedback(text, label)
right-click missed text ─▶ context menu  ─┘         │
                                                    ▼
                                          POST /feedback   (in-memory before it returns)
                                                    ▼
                                  content script re-scores matching lines (~200 ms)
```

Two entry points, one function, so they can never drift: **hold-to-review** for text that
*was* flagged, **right-click** for text the model *missed* (not blurred, so selectable
normally).

### Three matching tiers, cheapest and safest first

| tier | mechanism | cost | catches |
|---|---|---|---|
| 1 | exact, on **normalized** text | O(1) dict, **before tokenization** | `৭। খাঙ্কিরপোলা` ≡ `খাঙ্কিরপোলা` |
| 2 | exact, whitespace removed | O(1) dict | `খাঙ্কির পোলা` ≡ `খাঙ্কিরপোলা` |
| 3 | cosine on the `[CLS]` vector | one `(B,H)@(H,N)` matmul | `শুয়োরের বাচ্চা` → `শুয়োরের বাচ্চারা` |

Tier 1 keys on the *normalized* form, so list numbering collapses for free. It runs before
tokenization, so a taught line costs **zero model inference** — measured **1.1 ms vs
25.4 ms**. Corrections make the backend *cheaper* as it learns, not busier.

Tier 2 exists because Bangla compounds get split and joined inconsistently by real writers,
making spacing the single most common variant — and being still an exact string comparison,
it generalizes with none of tier 3's risk.

### The similarity threshold is calibrated, not guessed

Tier 3 is the one that can do damage: a wrong generalization silently mislabels unrelated
text. Cross-similarity of 36 benign Bangla lines against 49 taught slurs:

| threshold | 0.90 | 0.91 | 0.92 | 0.93 | 0.94 | **0.95** |
|---|---|---|---|---|---|---|
| benign lines wrongly inheriting a slur label | 41.7% | 36.1% | 25.0% | 2.8% | 0.0% | **0.0%** |

A first pass on a handful of hand-picked pairs suggested 0.93 was safe; the full cross matrix
showed it false-links 2.8%. **A small sample was actively misleading here.**

Broken down by length, every near-miss came from a **short taught side**: benign text reaches
0.938 against taught examples of ≤10 chars (`বুশ` ~ `ধোন`) but only 0.80 against examples of
≥16. A one-word string's `[CLS]` vector sits near the generic "short Bangla token" vector, so
cosine stops carrying meaning. Hence a **length-dependent bar**: taught examples under 16
characters require **0.98**. Genuine short-slur variants clear that easily
(`শুয়োরের বাচ্চা` ~ `শুয়োরের বাচ্চারা` = 0.9989, `মাগীর` ~ `মাগির` = 0.9967).

End to end: **7/7** variant forms of 4 taught slurs caught, **0/27** benign lines false-linked.

### Nothing blocks a request

- Disk writes go to a **daemon writer thread** via a queue; the request path never touches
  disk.
- The embedding index is warmed on a **background thread** at startup.
- Storage is **append-only JSONL with tombstone records** for deletes — no rewriting, so a
  process killed mid-append can only lose the last line, which `load()` skips.

### Corrections are training data

`backend/data/feedback.jsonl` carries `text`, `normalized`, `language`, `label`, `label_id`,
`created_at` — everything a retrain needs. This is not throwaway UI state; it's a growing,
human-verified dataset drawn from exactly the distribution the model is weakest on.

> ⚠️ **Two cautions before using it as training data.** It is **non-random** — it samples
> only what the model got wrong *and* the user noticed — so letting it leak into an
> evaluation split will inflate apparent gains. Keep it strictly in `train` and keep
> `ood_eval_bn.csv` as the held-out yardstick. It is also single-annotator, carrying none of
> the inter-annotator agreement the Phase 4 protocol produces.
>
> It also records a trace of real browsing. `data/` is not in `.gitignore`.

### Beyond the taught text — the correction cache

The three tiers above only cover text the matcher can reach. Measured: after teaching `16M`
and `1M`, **18 of 20** unseen chrome strings from the same family were still flagged. A
correction that only covers its own string isn't teaching.

So every correction is also kept as a **key in the encoder's embedding space**, and the
nearest one votes — with the vote decaying as distance grows:

```
vote_c(x) = α · max{ exp(−β · (1 − cos(x, kᵢ))) : label(kᵢ) = c }
P(x)      = softmax( log P_base(x) + vote(x) )
```

This is the cache-model family from [Tip-Adapter](https://arxiv.org/abs/2207.09519), with
one deliberate change. Tip-Adapter **sums** the affinity over every key, which grows without
bound as the store fills — tuned at 78 corrections, it wiped out sarcasm (9/12 → 1/12) by
156. **Max**-pooling depends only on the single closest correction, so adding a thousand
far-away ones changes nothing. Verified identical from 78 keys to 4,992.

| | value | why |
|---|---|---|
| `α` English | 8.0 | largest value costing **zero** genuine detections |
| `α` Bangla | 4.0 | corrections push *towards* flagging; 8.0 started flagging benign Bangla |
| `β` | 8.0 | cos 0.95 → 67% strength; cos 0.50 → 1.8% |

It requires **no training at all**, so it works from correction #1 and takes effect on the
very next line scanned. Deleting a correction removes its key, so undo is equally immediate.

> **A trained linear probe was tried and is off by default.** Measured end-to-end against
> the live server, it made chrome false positives *worse* (37 → 42 of 471) while changing
> nothing else. Set `ADAPTER_MODE=probe` to re-enable it; `cache` is the default.

### And eventually, the model itself

The cache never changes a weight. After **150 new corrections** for a language, the backend
fine-tunes that language's real 135M-parameter checkpoint on them, in a subprocess, mixed
with 6,000 class-stratified rows of the original training data. Nothing is promoted
automatically — a candidate must improve on corrections while degrading none of five
protected metrics. See **[RETRAINING.md](RETRAINING.md)** for the full pipeline.

---

## Part 5 — Performance

> **In one sentence:** three changes made a real page ~2× faster and re-scans ~59× faster,
> with zero changed decisions.

### Where the time actually goes

Profiled on a page-length batch of 20 lines:

| stage | time | share |
|---|---|---|
| normalization (regex) | 0.10 ms | 0.1% |
| tokenization | 0.54 ms | 0.7% |
| **model forward pass** | **78.03 ms** | **99.2%** |

Only the encoder matters. This immediately rules out micro-optimising the Python around it.

### Measured on the real page

The page this project was debugged against — a 175-entry Bangla slur list — yields **652
extracted lines** (181 Bangla, 471 English page chrome), **11.3% of them duplicates**,
median 21 characters with a 361-character outlier.

| | before | after |
|---|---|---|
| cold (first load) | 3169 ms | **1630 ms** (1.94×) |
| re-scan (warm cache) | 2483 ms | **42 ms** (58.6×) |
| **decisions changed** | — | **0 / 652** |

> **Superseded by a much simpler win.** Skipping `<nav>`/`<aside>`/`<footer>`
> ([Step 2](#skipping-page-furniture--the-single-biggest-accuracy-win)) cut the same page
> from 652 lines to **185** — a further 3.5× reduction that no amount of batching could
> reach, because the cheapest inference is the one never run. The optimizations below still
> apply to the 185 that remain.

### The three changes

**1. Length bucketing — the biggest and least obvious win.** A batch pads to its *longest*
member. One 106-token paragraph sitting among 20 nine-token list entries makes all 20 cost
106 tokens of compute:

```
19 short lines + 1 long   → padded to (20,128) → 434 ms
same, split by length     → (20,16) + (1,128)  →  94 ms
```

Sorting by token length before chunking keeps long lines together. Real pages always mix a
long post with short lines, so this fires constantly.

**2. Deduplication.** 11.3% of that page's lines were repeats. Scoring one copy and fanning
the result out is **bit-identical** (max |ΔP| = 0.0).

**3. Result cache** (`inference_cache.py`), keyed on normalized text so `৭। X` and `X` share
an entry. A dict lookup is **331,000×** cheaper than a forward pass.

> **The cache stores raw model output, never a finished verdict.** Two consequences make it
> safe:
> - **Corrections are consulted *before* the cache**, so teaching still overrides instantly.
> - **The `[CLS]` vector is cached alongside the probabilities**, so the similarity tier still
>   runs on a cache hit. Caching only the probabilities would silently disable
>   `feedback_similar` for every previously-seen line — a real behaviour change disguised as
>   an optimization. There is a test pinning exactly this.

### Rejected, with evidence

| candidate | result |
|---|---|
| **int8 dynamic quantization** | **2.1× slower** here (0.47×) — the standard recommendation, and a net loss on this hardware |
| thread tuning | 4 threads already optimal (1→92 ms, 2→81, **4→77**, 6→90, 8→89) |
| fixed-length padding | already dynamic; `padding=True` is 5× better than `max_length` |
| `inference_mode` vs `no_grad` | +2.9%, not worth the churn |

### On "zero accuracy cost"

Dedup and caching are **bit-identical**. Length bucketing is not — it changes padding width,
so float reductions run in a different order, perturbing probabilities by **~2.8e-06**.

That is safe here, and the margin says why: on the real page, the **closest any line came to
a decision boundary was 0.0211** — a safety factor of **~7,500×** — and no line was within
1e-4 of a threshold. It is also not a new class of error: scoring a line alone versus in a
batch already moves it ~8e-07 in the current code.

---

## Part 6 — The full lifecycle, end to end

```
1. Page loads. content_script.js runs at document_idle → start():
     • injects the .hs-flagged CSS
     • finds the container (#feed if present, else document.body)
     • walks it with the TreeWalker, groups text nodes into lines
     • attaches MutationObserver

2. New content appears (scroll insert, or the page's own JS re-renders):
     • MutationObserver fires, debounced 200 ms
     • each new subtree → extractLines() → lines
     • each line's anchor element gets an IntersectionObserver

3. A line scrolls within 200 px of the viewport:
     • IntersectionObserver fires, unobserves itself (one-shot)
     • every line registered to that anchor is queued
     • detectLanguage() tags each "bn" or "en"

4. Within 200 ms, the queue flushes as one batch (≤50 items, ≤3 in flight):
     • content_script → chrome.runtime.sendMessage({type:"PREDICT_BATCH", items})
     • background.js does the fetch (its origin avoids the per-site LNA prompt)
     • POST localhost:8000/predict/batch

5. FastAPI:
     • Pydantic validates the batch shape before any code runs
     • normalize each text (Phase 2 cleaning + strip list numbering)
     • empty-after-normalization → definite non_hateful, no model call
     • taught exactly? → return the correction, no model call
     • dedup → cache lookup → sort by length → batched forward pass
     • similarity tier over the [CLS] vectors
     • one JSON response, each result carrying its client-supplied id

6. background.js relays the response back.

7. content_script matches each result by id and calls applyFlag():
     • hate_prob ≥ 0.50 or sarc_prob ≥ 0.85 → the stronger signal owns
       colour + blur + tint + tooltip
     • neither → clear any existing marks

8. The user sees it. To correct it: hold the line, pick a button — which stores
   the correction server-side and re-scores the affected lines in ~200 ms.
```

---

## Part 7 — Known limitations, stated plainly

- **The auxiliary heads are broken in every shipped checkpoint** (Part 1). This is the
  largest outstanding problem — it invalidates RQ1, not just one response field.
- **Page chrome gets flagged.** On the real test page, **41 of 48 yellow flags were
  navigation and widget text** — "Visitor Counter", "Feedjit", "Posts", "Post a Comment" —
  scoring `sarcastic` at 0.998–1.000. Raising the threshold cannot help; they score 1.000.
  Red is clean (93 flags, all Bangla, zero English false positives). The fix is to skip
  `<nav>`/`<aside>`/`<footer>`/link-only elements before scoring, which would also remove
  ~72% of the work on that page — but it risks skipping real content inside those landmarks,
  so it is not implemented.
- **CORS is wide open** (`allow_origins=["*"]`) and endpoints are hardcoded to
  `http://localhost:8000`. Both need to change before any non-local deployment.
- **Line extraction is a heuristic.** Walking text nodes is far more robust than the
  element-scoring system it replaced, but a line is still "text between block boundaries or
  `<br>`s" — unusual markup can still split a sentence or merge two.
- **Corrections are per-line, not per-concept.** A correction generalizes only as far as the
  three tiers reach. A genuinely novel rephrasing must be taught again. This is a deliberate
  ceiling: the threshold that would catch loose paraphrases is the one that false-links
  25–42% of benign text.
- **Corrections have no review UI.** `GET /feedback` lists them and `DELETE /feedback`
  removes one, but there is no in-extension browser, so a mistaken correction is awkward to
  find. The JSONL is directly editable as a fallback.
- **The similarity index is a linear scan** — one dense matmul against every taught vector.
  Correct at this scale (sub-millisecond at the 5,000 cap), but O(N) per request, not an ANN
  index.
- **The hold gesture is mouse-only.** `mousedown`/`mousemove`/`mouseup` are wired;
  `touchstart`/`touchend` are not, so long-press does not open the review on a touchscreen.

---

## Appendix — reference tables

### File map

| File | Owns |
|---|---|
| `backend/app/config.py` | Checkpoint paths, backbones, labels, every tunable constant |
| `backend/app/architecture.py` | The model class — must match training exactly |
| `backend/app/model_registry.py` | Loading both models once, holding them in memory |
| `backend/app/preprocessing.py` | Train/serve text normalization + list-prefix stripping |
| `backend/app/schemas.py` | Request/response contracts (Pydantic) |
| `backend/app/feedback.py` | User corrections: 3-tier matching, persistence |
| `backend/app/inference_cache.py` | Memoized model outputs (probs + `[CLS]`) |
| `backend/app/logging_config.py` | JSON-lines structured logging |
| `backend/app/main.py` | Endpoints, orchestration, CORS/PNA middleware |
| `extension/manifest.json` | Permissions, what runs where |
| `extension/background.js` | The only network calls; context menus |
| `extension/content_script.js` | Watch, extract, batch, mark, review |
| `extension/popup.html/js` | On/off, backend status, correction count |
| `extension/test-page/index.html` | Self-contained fake feed (`.post` divs) for pipeline testing |

### Tunable constants — exactly where each one lives

Two rules make this table short: **every threshold is defined in exactly one place**, and
**all visual thresholds are client-side** (the backend returns probabilities and takes no
view on what counts as flagged).

> Line numbers are accurate as of this writing but drift with edits. To re-locate any of
> them: `grep -n "NAME" backend/app/*.py extension/*.js`

#### Visual decisions — `extension/content_script.js`

| Constant | Value | Defined | Read at | Why this value |
|---|---|---|---|---|
| `HATE_THRESHOLD` | 0.50 | `:89` | `:621` | A missed slur costs more than missed irony |
| `SARC_THRESHOLD_BY_LANGUAGE` | `{bn: 0.85, en: 0.95}` | `:144` | `:625` via `sarcThreshold()` `:149` | bn has a clean knee; en is saturated and has none |
| `SARC_THRESHOLD_FALLBACK` | 0.95 | `:147` | `:151` | An unmeasured language is where over-flagging is least defensible |
| `HATE_BLUR_MIN_PX` / `MAX` | 4 / 14 | `:167-168` | `:639` | Genuinely unreadable at full conviction |
| `TINT_MIN_ALPHA` / `MAX` | 0.15 / 0.55 | `:172-173` | `:640` | Saturation tracks conviction |
| `MIN_LINE_LENGTH` | 3 | `:217` | `queueLineIfDue` | A real slur can be 4 characters |

#### Pipeline timing — `extension/content_script.js`

| Constant | Value | Defined | Why this value |
|---|---|---|---|
| `MUTATION_DEBOUNCE_MS` | 200 | `:69` | Collapse a scroll burst into one pass |
| `BATCH_DEBOUNCE_MS` | 200 | `:70` | Batch instead of one request per line |
| `BATCH_MAX_ITEMS` | 50 | `:71` | Matches the backend's own cap |
| `VIEWPORT_ROOT_MARGIN` | `"200px"` | `:72` | Score just before the text is visible |
| `MAX_CONCURRENT_BATCHES` | 3 | `:446` | Backpressure; parallel measured 1.29× over serial |
| `MAX_RETRY_DELAY_MS` | 30000 | `:496` | Cap on exponential backoff when the backend is down |
| `LONG_PRESS_MS` | 450 | `:917` | Long enough not to fire on an ordinary click |
| `LONG_PRESS_SLOP_PX` | 6 | `:921` | Survives a shaky hand, cancels on a real drag |

#### Backend — `backend/app/config.py` (all env-overridable)

| Constant | Value | Defined | Applied at | Why this value |
|---|---|---|---|---|
| `MAX_LEN` | 128 | `:16` | `main.py` tokenizer calls | Must match training-time truncation |
| `FEEDBACK_SIM_THRESHOLD` | 0.95 | `:52` | `feedback.py:410` | Past the false-link cliff, with margin |
| `FEEDBACK_SIM_SHORT_CHARS` | 16 | `:65` | `feedback.py:409` | Below this, the stricter bar applies |
| `FEEDBACK_SIM_SHORT_THRESHOLD` | 0.98 | `:66` | `feedback.py:408` | Short strings have near-degenerate embeddings |
| `INFERENCE_CACHE_MAX_ENTRIES` | 4000 | `:72` | `inference_cache.py` | ~3 KB/entry ⇒ ~12 MB |
| `FEEDBACK_MAX_ENTRIES` | 5000 | `:77` | `feedback.py` eviction | Similarity matmul stays sub-millisecond |
| `_MAX_FORWARD_BATCH` | 32 | `main.py:48` | `main.py:435` | Batching gains flatten past ~20 |

The three similarity constants converge in one expression — the per-candidate bar built in
`feedback.py:406-412`:

```python
thresholds = np.array([
    self._short_threshold                                    # 0.98
    if len(self._entries[k].normalized) < self._short_chars  # < 16 chars
    else self._sim_threshold                                 # 0.95
    for k in keys
], dtype=np.float32)
```

Each taught example carries **its own** threshold, so a short slur and a long sentence are
judged on different bars inside a single matmul:

```python
feedback.py:375   sims = _unit(vectors) @ matrix.T                          # (B, N) cosines
feedback.py:382   qualifying = np.where(sims >= thresholds[None, :], sims, -np.inf)
```

The `-np.inf` masking on `:382` is load-bearing rather than stylistic. The winner has to be
the best candidate that clears **its own** bar — a plain `argmax` over `sims` could return a
short candidate sitting *below* its 0.98 threshold and thereby hide a longer, qualifying one
behind it.

#### Endpoints — `extension/background.js`

| Constant | Value | Defined |
|---|---|---|
| `BATCH_ENDPOINT` | `http://localhost:8000/predict/batch` | `:21` |
| `FEEDBACK_ENDPOINT` | `http://localhost:8000/feedback` | `:22` |

These two are the only hardcoded URLs in the extension; `popup.js` separately calls
`/health`. All three must change before any non-local deployment.

### Response fields

| Field | Use it? | Meaning |
|---|---|---|
| `non_hateful_prob` / `hate_prob` / `sarc_prob` | ✅ **yes** | Fused 3-class softmax; sums to 1.0 |
| `label` / `label_id` / `confidence` | ⚠️ display only | The argmax; don't threshold on it |
| `source` | ✅ yes | `model`, `feedback_exact`, or `feedback_similar` |
| `matched_text` / `match_similarity` | ✅ yes | Which correction caused a `feedback_similar` verdict |
| `is_hateful_score` | ❌ **never** | Broken — a second sarcasm score, anti-correlated with hate |
| `is_sarcastic_score` | ⚠️ unreliable | Trained correctly, but redundant with `sarc_prob` |
