# What executes after a correction

*You marked a flagged line. This is everything that runs from that moment on, in order.*

There are **four phases**, at four very different timescales:

| phase | when | changes model weights? |
|---|---|---|
| **1. Store** | 0 – 70 ms | no |
| **2. Re-score the page** | +18 ms | no |
| **3. Accumulate** | until 150 corrections | no |
| **4. Retrain** | ~29 min, then a gate decides | **yes, if it passes** |

Phases 1–3 always happen. Phase 4 happens once every 150 corrections, and may end with
nothing being adopted.

---

## Phase 1 — Store the correction  (0 → 70 ms)

`submitMark()` in the content script fires two things at once.

**On screen, immediately (0 ms):** if you chose *Neither*, the blur is removed right away.
If you chose *Hateful* or *Sarcastic*, the line re-blurs in the new colour. You never wait
for the network.

**On the wire:** `POST /feedback` → [`main.py:614`](../backend/app/main.py#L614). Five steps
run inside that request:

```
1. normalize        strip list numbering, collapse whitespace, lowercase
                    → the SAME cleaning used at training time
2. embed            run the encoder once → a 768-number vector
                    (this is the 70 ms; everything else is noise)
3. write to disk    append one JSON line to backend/data/feedback.jsonl     0.06 ms
4. rebuild cache    re-stack the correction vectors for that language       0.07 ms
5. count            advance the retrain counter, persist it
```

### Step 3 is synchronous on purpose

An earlier version appended on a background daemon thread. Daemon threads are killed at
interpreter exit **without draining their queue**, so corrections were being silently lost
on shutdown. It now writes before the function returns, at a cost of 0.06 ms — noise next to
the 70 ms embedding. A regression test reads the file *without* flushing to make sure this
never comes back.

### Step 4 is what makes it generalize

The cache is rebuilt inline rather than on a background thread, because it is just a
`np.stack` of vectors that already exist. By the time the HTTP response is written, the
correction already influences text you never taught.

**Net effect of phase 1: the correction is durable and in force in about 70 ms.**

---

## Phase 2 — Re-score the affected lines  (+18 ms)

When the response comes back `ok`, the content script calls `rescoreMatchingLines(text)`:

1. re-extract the page's lines;
2. keep those that contain your corrected text, **or are contained by it** (you may have
   selected a fragment, or corrected a line that sits inside a longer one);
3. clear their flags and re-queue them;
4. **flush immediately** — the normal 200 ms batch debounce is skipped, because this is a
   direct response to something you just did.

Only matching lines are cleared. Blanket-clearing every flag on the page would briefly
expose unrelated hateful text that isn't going to be re-scored.

### What the backend does with those re-scored lines

Every line now goes through four tiers, cheapest and safest first:

| tier | matches | catches | cost |
|---|---|---|---|
| **1. exact** | identical after normalizing | the line you corrected | ~0 |
| **2. similar** | cosine ≥ 0.95 on the embedding | spelling variants — `হারামি` vs `হারামী` | one matmul |
| **3. cache** | nearest correction, distance-weighted | text merely *similar in meaning* | 0.08 ms / 50 lines |
| **4. model** | — | everything else | ~40 ms |

Tiers 1 and 2 fix the exact line. **Tier 3 is why lines you never touched also get fixed.**

### Tier 3, concretely

Every correction is a point in the model's meaning-space. New text is scored by its
**nearest** correction, with the vote fading as distance grows:

```
vote(x) = α · max{ exp( −β · (1 − cosine(x, correctionᵢ)) ) }
final   = softmax( log P_model(x) + vote(x) )
```

In plain terms: *close to something you corrected → apply it strongly; far away → leave it
completely alone.*

Measured live — teaching one sentence, then scoring a paraphrase that was **never taught**:

```
"Absolutely stunning work, a real masterpiece of incompetence."   ← taught
"Truly stunning work, a genuine masterpiece of incompetence."     ← never taught

P(sarcastic) on the untaught one:   0.995  →  0.008
```

No training ran. This is a lookup, and it works from correction #1.

> **Why `max` and not a sum?** The published version of this technique (Tip-Adapter) sums
> every correction's vote. We built that first and it breaks: the total grows as the store
> fills, so settings tuned at 78 corrections destroyed sarcasm detection by 156 (9/12 →
> 1/12). `max` depends only on the single nearest correction, so adding a thousand unrelated
> ones changes nothing. Verified identical from 78 keys to 4,992.

**Undo is symmetrical.** Delete the correction, its point disappears, and the paraphrase
above goes straight back to 0.995.

---

## Phase 3 — Accumulate  (until 150)

`retrainer.note_correction(language)` → [`retrain.py:628`](../backend/app/retrain.py#L628).

Two separate counters are kept, both persisted to `backend/data/retrain_state.json` so a
server restart never loses progress:

| counter | resets? |
|---|---|
| `since_last_run` — progress toward the next fine-tune | only on a **promoted** run |
| `total_corrections` — lifetime tally | **never** |

That first rule is the important one. **A rejected run does not reset the counter** — that
is what "if training fails the gate, keep collecting corrections" means mechanically. The
next attempt happens with more data, not with the same data.

Check progress any time:

```bash
curl localhost:8000/retrain
```

---

## Phase 4 — Retrain  (at 150 corrections)

### 4.1 Trigger

The 150th correction calls `trigger()` → [`retrain.py:642`](../backend/app/retrain.py#L642),
which launches a **separate OS process**:

```
python -m app.retrain --language en
```

A subprocess, not a thread, for two reasons: a full fine-tune saturates the CPU for half an
hour, and a crash inside training must not be able to take the API down. **The extension
keeps working normally throughout** — it is still being served by the current model.

If a run is already going for that language, a second one is refused rather than queued.

### 4.2 Assemble the training data

This is the step that decides whether the whole thing works.

```
150 corrections × 8 repeats   ≈ 1,200 examples    ~17% of the gradient
+ 6,000 replay rows                                class-stratified from the
                                                   ORIGINAL Phase 2 training data
= ~7,200 examples
```

**Why replay the original data at all?** Because training on the corrections alone destroys
the model. That is measured, not assumed — we ran it deliberately:

| replay rows | sarcasm recall | outcome |
|---|---|---|
| 300 | 0.750 → **0.333** | **REJECTED** by the gate |
| 6,000 | 0.750 → **0.750** | promoted |

With too little replay, the model learns "when unsure, say non-hateful" — which scores
perfectly on the corrections and destroys everything else. The replay sample is what holds
the rest of the model in place.

Preview the mix without training anything:

```bash
curl -X POST "localhost:8000/retrain/train?language=en&dry_run=true"
```

### 4.3 Train

Starts from **whatever checkpoint is currently in production** (the original, or the last
promoted one), and fine-tunes all ~135M parameters:

| | |
|---|---|
| optimizer | AdamW, lr 2e-5, weight decay 0.01 |
| schedule | linear warmup |
| epochs | 2 |
| batch size | 32 |
| gradient clipping | 1.0 |
| loss | `0.3·hate + 0.3·sarcasm + 0.4·three-class` |

> **One deliberate difference from the original notebook.** The shipped checkpoints were
> trained with the hate head pointed at the *sarcasm* target — a bug that left the two heads
> 0.9997 correlated. The retrain uses the corrected loss, each head against its own
> phenomenon, so a retrain **cannot reintroduce that defect.**

### 4.4 Save the candidate — before judging it

```
backend/models/retrained/en/20260813-025132/
    model.pt        the candidate
    metrics.json    before/after numbers, machine-readable
    report.txt      the human-readable report
```

Written **before** the gate runs, and kept **even if rejected** — so a refusal is
inspectable rather than invisible.

The originals in `backend/models/Lightweight English/` are never touched at any point.

### 4.5 Evaluate

Both the old and new models are scored on a **frozen** regression set
(`backend/data/regression_set.json`), built once from data the models never trained on:

- the Phase 2 **test** splits, class-stratified;
- `ood_eval_bn.csv` Bangla slurs and threats;
- a curated English sarcasm / benign pair.

Frozen because a gate whose yardstick moves between runs is not a gate.

### 4.6 The gate

Promotion requires **both** conditions:

**(a) It must actually learn the corrections** — at least +5 percentage points. A run that
changes nothing still swaps the model users are served; it has to earn that.

**(b) No protected metric may degrade beyond its tolerance:**

| metric | may drop by at most | why it's protected |
|---|---|---|
| macro-F1 | 0.01 | general ability |
| sarcasm recall | 0.02 | the first thing to collapse |
| hate recall | 0.02 | the product's core job |
| Bangla slur recall | 0.02 | the extension's main use case |
| benign false-positive rate | may **rise** by 0.02 | a rate — lower is better |

### 4.7 Promote, or don't

**A real run — real corrections, default settings:**

```
corrections used : 78          training time : 29.1 min

metric                before     after     delta
macro_f1               0.897     0.892    -0.005    ok
sarcasm_recall         0.750     0.750    +0.000    ok
hate_recall            0.912     0.905    -0.007    ok
slur_recall            1.000     1.000    +0.000    ok
benign_fp_rate         0.500     0.500    +0.000    ok
corrections            0.000     1.000    +1.000    ok
VERDICT: PROMOTED
```

| outcome | what happens |
|---|---|
| **passes** | pointer flips to the new checkpoint; counter resets to 0; previous checkpoint remembered for rollback |
| **fails** | production model untouched; **counter not reset**; candidate kept on disk; failure reasons written to the report |

Promotion is a **pointer flip**, not a file overwrite —
`backend/models/retrained/en/production.json`:

```json
{ "current": ".../20260813-025132/model.pt", "previous": null }
```

### 4.8 Load it

The pointer changes on disk; the running server picks it up on next load:

```bash
curl -X POST localhost:8000/retrain/reload
```

This re-reads the checkpoint, **clears the inference cache** (its entries were produced by
the outgoing model — keeping them would serve stale scores), and rebuilds the correction
cache.

### 4.9 Rollback

```bash
curl -X POST "localhost:8000/retrain/rollback?language=en"
curl -X POST  localhost:8000/retrain/reload
```

Rolling back the first promotion returns to the model the project shipped with. A promoted
checkpoint that goes missing, or a corrupt manifest, **also** falls back to the original
rather than failing to start — serving the original model is always safe; failing to boot
is not.

---

## The whole timeline

```
 t = 0 ms      you click Hateful / Sarcastic / Neither
               └─ the line updates on screen instantly

 t = 70 ms     correction embedded, written to disk, cache rebuilt, counter++
               └─ FROM HERE ON the correction is permanent and in force

 t = 88 ms     matching lines on the page re-scored and repainted
               └─ and every similar line, from now on, anywhere

 ...           corrections 2 … 149 accumulate the same way

 correction    fine-tune starts in a separate process
   #150        └─ the extension keeps working normally

 +29 min       candidate saved, old and new models both scored

 +29 min       GATE
               ├─ pass → promote, reset counter, rollback still available
               └─ fail → discard, KEEP COLLECTING, model unchanged
```

---

## Where each thing lives

| what | where | survives restart? | survives retrain? |
|---|---|---|---|
| your corrections | `backend/data/feedback.jsonl` | yes | **yes — never deleted** |
| correction cache | memory, rebuilt from the file | rebuilt at startup | yes |
| retrain counters | `backend/data/retrain_state.json` | yes | reset only on promotion |
| candidates + reports | `backend/models/retrained/<lang>/<timestamp>/` | yes | yes |
| which model is live | `backend/models/retrained/<lang>/production.json` | yes | — |
| the original models | `backend/models/Lightweight */` | yes | **never written to** |

---

## Two questions this design answers

**"When does my correction actually take effect?"**
About 90 ms. Not at the next retrain, not at the next page load. Phases 1–2 are the answer,
and they involve no training at all.

**"So what is the retraining for, then?"**
The cache is a patch applied *on top of* a model that is still wrong underneath — it works,
but every correction it handles is one the model itself still gets wrong. Retraining moves
the fix into the weights, so the cache has less to do. One is immediate and reversible; the
other is permanent and gated. You get both.
