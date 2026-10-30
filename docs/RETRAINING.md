# Shipping a retrained model to the extension

> **Two different things are described in this file.**
>
> * **[The automatic pipeline](#automatic-retraining-from-user-corrections)** — runs itself
>   from corrections made in the extension. Nothing to do by hand.
> * **Everything after it** — the manual path, for when you train on a *new dataset*
>   (new CSVs, a new backbone, the corrected notebook). Start at [§0](#0-which-notebook-which-model-and-do-you-start-over).

---

# Automatic retraining from user corrections

Every correction made in the extension is stored permanently in
`backend/data/feedback.jsonl` as labelled training data. After **150 new corrections for
one language**, the backend fine-tunes that language's real 135M-parameter checkpoint on
them — automatically, in a separate process, without touching the model being served.

## What happens, in order

| Step | Where | Notes |
|---|---|---|
| 1. Correction stored | `feedback.py` | Durable before the HTTP response returns |
| 2. Takes effect **immediately** | `adapter.py` cache | No training; see [instant vs. eventual](#instant-vs-eventual) |
| 3. Counter advances | `retrain.py` | Persisted, so a restart doesn't lose progress |
| 4. At 150 → fine-tune starts | subprocess | Serving latency unaffected |
| 5. Candidate scored against a frozen regression set | `evaluate_checkpoint` | |
| 6. Gate decides | `gate()` | **No automatic promotion** |
| 7a. Pass → promote, reset counter | `promote()` | Pointer flip; rollback stays available |
| 7b. Fail → keep current model, **counter is not reset** | | Corrections keep accumulating |

## The training mix (this is the part that matters)

Training on the 150 corrections alone destroys the model. That is measured, not theoretical
— on this project's own data a starved run collapsed sarcasm recall from 0.750 to **0.333**.
So every run mixes:

```
150 corrections  x8 repeats     ~1,200 examples   (≈17% of the gradient)
+ 6,000 replay rows             class-stratified from the ORIGINAL Phase 2 training data
= ~7,200 examples
```

The replay sample is what prevents catastrophic forgetting. Both numbers are tunable
(`RETRAIN_REPLAY_ROWS`, `RETRAIN_CORRECTION_REPEAT` in `config.py`), and the difference is
stark:

| replay rows | sarcasm recall | verdict |
|---|---|---|
| 300 | 0.750 → **0.333** | REJECTED by the gate |
| 6,000 | 0.750 → **0.750** | PROMOTED |

The run also uses the **corrected** loss — hate head against the hate target — so a retrain
cannot reintroduce the defect that made the two auxiliary heads 0.9997 correlated.

## The gate

A candidate is promoted only if **both** hold:

1. it actually learned the corrections (≥ +5 percentage points), and
2. **no protected metric degraded** beyond its tolerance:

| protected metric | tolerance | why it's protected |
|---|---|---|
| `macro_f1` | 0.01 | general ability, on the untouched Phase 2 test split |
| `sarcasm_recall` | 0.02 | the first thing to collapse under one-sided corrections |
| `hate_recall` | 0.02 | the product's core job |
| `slur_recall` | 0.02 | Bangla slurs/threats from `ood_eval_bn.csv` |
| `benign_fp_rate` | 0.02 | a **rate** — may *rise* by at most this |

The regression set is built once and frozen to `backend/data/regression_set.json`, from data
the models never trained on. A gate whose yardstick moves between runs is not a gate.

A real promoted run looked like this:

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

Every run writes `model.pt`, `metrics.json` and `report.txt` under
`backend/models/retrained/<lang>/<timestamp>/` — **including rejected ones**, so a refusal
is inspectable rather than invisible.

## How long it takes

Measured on CPU (Apple Silicon, no GPU), English, 78 corrections + 6,000 replay rows,
2 epochs: **29 minutes**, including both evaluation passes. Bangla is somewhat slower
(multilingual vocabulary, larger embedding matrix) — budget ~35–40 min. On a GPU this is
a few minutes. It runs in a subprocess, so the API keeps serving throughout.

## Your corrections are never at risk

Nothing in this pipeline deletes `feedback.jsonl`. Corrections remain in force through the
cache regardless of whether a retrain has happened, has failed, or has been rolled back.
Promotion doesn't retire them either — they stay as training data for every future run.

## Rollback

The original checkpoints in `backend/models/` are **never written to**. Promotion writes a
new file and flips a pointer in `production.json`, so:

```bash
curl -X POST "localhost:8000/retrain/rollback?language=en"   # back to the previous checkpoint
curl -X POST "localhost:8000/retrain/reload"                 # load it without a restart
```

Rolling back the first promotion returns to the model the project shipped with. A promoted
checkpoint that goes missing, or a corrupt manifest, also falls back to the original rather
than failing to boot.

## Controls

```bash
curl localhost:8000/retrain                                     # progress, production, history
curl -X POST "localhost:8000/retrain/train?language=en&dry_run=true"   # show the mix, train nothing
curl -X POST "localhost:8000/retrain/train?language=en"         # force a run now
```

## Instant vs. eventual

These are two different mechanisms and it's worth keeping them apart:

| | correction cache | retraining |
|---|---|---|
| when | immediately, correction #1 | every 150 corrections |
| changes weights? | **no** | yes |
| cost | ~0 ms | ~30 min, out of process |
| scope | text near a correction | the whole model |
| undo | delete the correction | `/retrain/rollback` |

The cache is why a correction is visible on the next scan. Retraining is what eventually
makes the model itself right, so the cache has less to do.

---

## 0. Which notebook, which model, and do you start over?

**One notebook trains both shipped checkpoints:**

```
Phase 6 (Architectures)/distilbert-base (Lightweight model)/lightweight-model.ipynb
    cell 5  → English  → lightweight_en/{model.pt, tokenizer.json, tokenizer_config.json}
    cell 6  → Bangla   → lightweight_bn/{model.pt, tokenizer.json, tokenizer_config.json}
```

Those two output folders are what get renamed to `Lightweight English` / `Lightweight Bangla`
and dropped into `backend/models/`. `lightweight-model-corrected.ipynb` is the same notebook
with the training defects fixed — start from that one, not the original.

**No, you do not start from the beginning.** Only Phase 6 re-runs:

| Phase | What it produced | Re-run it? |
|---|---|---|
| 1 — collection | raw corpora | **No** |
| 2 — preprocessing | `bn_train/val/test.csv` (67,009 bn), `en_*` (83,455 en) — the `text_clean` column | **No**, unless you change the cleaning — and if you do, `backend/app/preprocessing.py` must change identically |
| 3 — model comparison | classical / DL / transformer baselines | **No** — that question is already answered |
| 4 — human annotation | 999 rows per language, two-axis | **No** — it's an input, see below |
| 5 — data evaluation | dataset analysis | **No** |
| **6 — architecture** | **the shipped checkpoints** | **Yes — this is the only one** |

The notebook reads six CSVs and nothing else:

```python
PREPROCESSED_DIR = "/kaggle/input/datasets/.../preprocessed-datasets"
en_train = pd.read_csv(f"{PREPROCESSED_DIR}/en_train.csv")   # cols: text_clean, class
bn_train = pd.read_csv(f"{PREPROCESSED_DIR}/bn_train.csv")   # cols: text_clean, label, source
```

So to train on different data, you only have to produce CSVs with those column names. Note
English uses `class` and Bangla uses `label` — an inconsistency the notebook works around.

---

## 0b. If you train from the Phase 4 CSV annotations — read this first

`Phase 4/Bangla Annotation - bangla_hate_speech_1k.csv` and its English counterpart are the
**only two-axis labels in the project**: `hate_type` and `sarcasm_type` are separate columns,
annotated per row. That is exactly the instrument RQ1 needs. Four things will bite you.

### ⚠️ 1. The current auxiliary labels are derived, not annotated — a second defect

`lightweight-model.ipynb` cell 1 builds the two head targets like this:

```python
def add_aux_labels(df, label_col):
    df['is_hateful']   = (df[label_col] == 1).astype(int)
    df['is_sarcastic'] = (df[label_col] == 2).astype(int)
    return df
```

Both come from **the same 3-class column**, so they are mutually exclusive *by construction*:
`label==1` → hate 1 / sarc 0, `label==2` → hate 0 / sarc 1. **No row can ever be both.**

This matters more than it looks, and it revises the earlier diagnosis in
[MODEL_DEFECTS_AND_DETECTION.md](MODEL_DEFECTS_AND_DETECTION.md): fixing the loss bug (which
trained both heads against the sarcasm target) is **necessary but not sufficient**. Even with
the corrected loss, the heads are being taught from labels that can never co-occur, so the
architecture's whole premise — "a text can be hateful *and* sarcastic, and two independent
heads can capture that" — is still not testable. Phase 4's `hate_type`/`sarcasm_type` are the
only labels in the repo that can fix this:

```python
df['is_hateful']   = norm(df.hate_type).isin({'hate','hate_speech','abusive','offensive'}).astype(int)
df['is_sarcastic'] = norm(df.sarcasm_type).isin({'sarcasm','irony'}).astype(int)
```

### ⚠️ 2. …but genuine co-occurrence is very rare

Measured on the annotations after normalizing the label strings:

| | rows both hateful-ish **and** sarcastic |
|---|---|
| Bangla | **7 / 999** |
| English | **9 / 999** |

Seven examples is not enough to learn "sarcastic hate" from. Independent heads become
*trainable* with this data, but the co-occurrence case stays under-evidenced. If RQ1 is the
goal, this is the number to grow — targeted annotation of sarcastic-hateful text would do far
more than any architecture change.

### ⚠️ 3. The label strings are dirty — normalize before use

Case and trailing-space variants are treated as distinct classes if you don't:

```
sarcasm_type:  'Sarcasm' (82) · 'sarcasm ' (63) · 'sarcasm' (60)      ← three spellings, one class
hate_type:     'non_hateful' (236) · 'non hateful ' (187) · 'Non-hateful' (105)
```

```python
def norm(s):
    return None if pd.isna(s) else re.sub(r'_+','_', str(s).strip().lower().replace('-','_').replace(' ','_'))
```

Also: `sarcasm_type` is **23% empty** and `hate_type` **11% empty**. Decide explicitly whether
blank means "none" or "not annotated" — they are not the same, and treating NaN as "none"
silently invents ~230 negative labels per language.

### ⚠️ 4. 999 rows is an evaluation set, not a training set

| | Phase 2 (current training data) | Phase 4 annotations |
|---|---|---|
| Bangla | 67,009 | 999 |
| English | 83,455 | 999 |

Fine-tuning a 135M-parameter model on 999 rows will overfit hard. Three sane options:

1. **Best use — hold it out as the evaluation set.** It is balanced (333/333/333),
   human-verified, and two-axis. The project's core measurement problem is that `bn_test`
   shares sources with training, which is why in-distribution recall reads 91.4% while real
   out-of-distribution recall is 38.8%. Phase 4 fixes that honestly.
2. **Train on Phase 2 + supervise the heads with Phase 4** where the two overlap, so the bulk
   data drives the 3-class head and the annotated subset drives the two binary heads.
3. **Fine-tune the existing checkpoint** on Phase 4 at a low learning rate for 1–2 epochs.
   Cheapest, but with 999 rows expect it to move style more than substance.

Whichever you pick, keep `ood_eval_bn.csv` as a separate held-out fixture — it is the only
thing measuring the failure the extension actually hits.

---

Short answer to the two deployment questions:

- **To make a new model take effect:** drop 3 files into `backend/models/<name>/`, restart the
  server. That's it — unless you changed the backbone, in which case one line in `config.py`
  changes too.
- **Your corrections survive automatically.** They are stored as *text*, never as model
  vectors, so they are model-independent. Nothing to migrate. Details in §3.

---

## 1. What training must produce

Save these three files per language, from the training notebook:

```
backend/models/Lightweight Bangla/
├── model.pt              torch.save(model.state_dict())   ← state_dict ONLY, not the model object
├── tokenizer.json
└── tokenizer_config.json                                   ← tokenizer.save_pretrained(dir)
```

```python
# in the notebook, after training
torch.save(model.state_dict(), "model.pt")   # NOT torch.save(model, ...)
tokenizer.save_pretrained("./")
```

> **`state_dict()`, not the model object.** `torch.save(model)` pickles the class definition
> and will fail or silently misbehave when loaded against `architecture.py`.

**The architecture must be unchanged.** `load_state_dict()` matches by layer name and shape
and runs with `strict=True`, so a mismatch raises loudly — good. But if you *added* a layer
in training, you must make the identical change in `backend/app/architecture.py`, and read
that file's docstring first: it is deliberately written to mirror the training code
line-for-line, including the odd `hate_head[0]` / `hate_head[3]` indexing.

---

## 2. Installing it

**A. Back up your corrections first** (they're gitignored, so nothing else protects them):

```bash
cp backend/data/feedback.jsonl ~/feedback-backup-$(date +%F).jsonl
```

**B. Replace the files** at the paths above.

**C. If you changed the backbone**, change it in one place:

```python
# backend/app/config.py
CHECKPOINTS = {
    "bn": {"backbone": "sagorsarker/bangla-bert-base", ...},   # was distilbert-base-multilingual-cased
}
```

Nothing else needs touching for a dimension change — `architecture.py` reads
`hidden_size` off the encoder config, so 768 → 1024 adapts on its own.

**D. Verify before going through HTTP** — much faster to debug:

```bash
cd backend && source .venv/bin/activate
python verify_checkpoints.py      # loads both checkpoints, runs one real inference each
```

**E. Restart the server.** A restart is mandatory, not optional — see §3.

```bash
uvicorn app.main:app --reload --port 8000
```

---

## 3. Why your corrections survive, and what they need

`feedback.jsonl` stores **only text**:

```json
{"text":"...","normalized":"...","language":"bn","label":"hateful","label_id":1,"created_at":"..."}
```

No embeddings, no probabilities, nothing tied to a particular set of weights. So:

| tier | how it survives a model swap |
|---|---|
| 1 — exact match | Pure string comparison. Unaffected. |
| 2 — whitespace-insensitive | Pure string comparison. Unaffected. |
| 3 — `[CLS]` similarity | Vectors are **recomputed at startup** by `warm_index()`, using whatever model is now loaded. |

That last row is the reason a **restart is required**. Two caches hold old-model data and are
memory-only, so restarting clears both:

- the **similarity index** — old `[CLS]` vectors are meaningless in a new embedding space
- the **inference cache** (`inference_cache.py`) — holds old-model probabilities

Hot-swapping the `.pt` file without a restart leaves both stale and will produce wrong
answers with no error.

> ⚠️ **The one thing that would break corrections: changing `LABELS`.** `feedback.py` replays
> the log using the stored `label_id`, so if `config.py`'s
> `LABELS = {0: "non_hateful", 1: "hateful", 2: "sarcastic"}` ever changes meaning, every
> stored correction silently becomes a *different* label. If you must change the label set,
> rewrite `label_id` in the JSONL to match, using the `label` string as the source of truth.

---

## 4. What you MUST re-tune (do not skip this)

Every threshold in the system was calibrated against the *current* checkpoints. A new model
has a different probability distribution and a different embedding geometry, so **none of
these carry over**:

| what | current | where | why it must change |
|---|---|---|---|
| `HATE_THRESHOLD` | 0.50 | `extension/content_script.js:89` | New model, new distribution |
| `SARC_THRESHOLD_BY_LANGUAGE` | `{bn:0.85, en:0.95}` | `content_script.js:144` | 0.85 was the knee of *this* model's curve |
| `FEEDBACK_SIM_THRESHOLD` | 0.95 | `backend/app/config.py:52` | Cosine geometry is model-specific |
| `FEEDBACK_SIM_SHORT_THRESHOLD` | 0.98 | `config.py:66` | Same |

Re-run the same measurements that produced them:

```bash
# 1. Out-of-distribution recall / false positives on named targets
#    (recall and FP printed per category)
python "Phase 6 (Architectures)/distilbert-base (Lightweight model)/…"  # ood_eval_bn.csv fixture

# 2. Threshold sweep: benign wrongly tinted vs threats caught, per candidate bar
# 3. Similarity calibration: cross-similarity of benign lines vs taught slurs,
#    checking the false-link rate at 0.90 / 0.92 / 0.94 / 0.95 and by text length
```

The numbers to beat, from the current checkpoints:

| metric | current |
|---|---|
| `bn_test` hateful recall (in-distribution) | 91.4% |
| OOD fixture recall (slur + threat) | 38.8% |
| OOD false positives on bare proper nouns | **0.0%** — protect this |
| English benign wrongly tinted sarcastic | 40% at 0.85 |

> **Judge on `ood_eval_bn.csv`, not `bn_test`.** `bn_test` shares sources with training, so it
> measures in-distribution fit and is structurally blind to the failure that matters. The
> 91.4% → 38.8% gap is the whole story.

---

## 5. If you fixed the auxiliary-head bug

`lightweight-model-corrected.ipynb` fixes the loss that trained *both* sigmoid heads against
the sarcasm label. Once a checkpoint trained with it ships:

1. Confirm it worked: `corr(hate_head, sarc_head)` should be **well under 0.95** (currently
   0.9997). The corrected notebook asserts this every epoch.
2. `backend/app/schemas.py` — remove the `deprecated=True` markers and the warning comment on
   `is_hateful_score` / `is_sarcastic_score`.
3. Consider repointing the extension from `hate_prob` to `is_hateful_score`. A dedicated
   binary head is better calibrated for an intensity gradient than a 3-way softmax, which
   forces hate and sarcasm to compete for probability mass even though they co-occur.

---

## 6. Feeding your corrections back into training

`feedback.jsonl` is human-verified data drawn from exactly where the model is weakest — but
it is **not a clean dataset**:

- **Non-random.** It only samples what the model got wrong *and* someone noticed. Letting it
  into an evaluation split will inflate results. Keep it strictly in `train`; keep
  `ood_eval_bn.csv` held out.
- **Single-annotator.** None of the inter-annotator agreement the Phase 4 protocol produces.
- **Contains a browsing trace.** Review it before sharing.

```python
import pandas as pd
df = pd.read_json("backend/data/feedback.jsonl", lines=True)
df = df[~df.get("deleted", False).fillna(False)]        # drop tombstones
df = df.drop_duplicates("normalized", keep="last")       # last correction wins
train_extra = df[["normalized", "label_id", "language"]].rename(columns={"normalized": "text_clean"})
```

---

## 7. Checklist

```
[ ] backed up backend/data/feedback.jsonl
[ ] model.pt saved from state_dict(), not the model object
[ ] tokenizer.json + tokenizer_config.json saved alongside it
[ ] architecture.py matches the training class (if it changed at all)
[ ] config.py backbone string updated (only if the backbone changed)
[ ] LABELS unchanged — or label_id rewritten in the JSONL
[ ] python verify_checkpoints.py passes
[ ] server RESTARTED (clears both stale caches)
[ ] /health shows the same feedback_entries count as before
[ ] thresholds re-tuned against ood_eval_bn.csv
[ ] OOD false positives on proper nouns still 0.0%
```
