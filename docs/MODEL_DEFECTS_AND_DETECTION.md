# Why Bangla detection misses, and what was done about it

Investigation triggered by a reported failure: on a Facebook post consisting of a numbered
list of Bangla slurs, most entries were not blurred, some unrelated lines were, and the
behaviour looked inconsistent between reloads.

The extension's DOM handling turned out to be largely fine. **The model is the problem**, and
it has a specific, identifiable cause.

---

## 1. The headline numbers

Scored against the 57 reported lines, and against a cleaned-up fixture
(`Phase 6 (Architectures)/distilbert-base (Lightweight model)/ood_eval_bn.csv`) that separates
genuine slurs from bare proper nouns:

| measurement | value |
|---|---|
| `bn_test` hateful recall (in-distribution) | **91.4%** |
| reported list, raw, as the extension sent it | **17.5%** |
| reported list, after list-prefix normalization | **26.3%** |
| fixture recall on `slur` + `threat` (normalized) | **38.8%** |
| fixture false-positive rate on `named_target` | **0.0%** |

Two things to read out of this:

- The **91.4% → 38.8% gap** is the whole story. `bn_test` is drawn from the same
  BD_SHS / BIDWESH / BenSarc sources as training, so it measures in-distribution fit and is
  structurally blind to this failure. Nothing in the existing evaluation could have caught it.
- **0.0% false positives on bare proper nouns** is genuinely good and must be protected. A
  model that flags `গোলাম আযম` or `পাকিস্তানি` standalone is over-triggering on identity
  terms — the classic hate-speech failure mode. Any recall improvement bought by regressing
  this number is not an improvement.

---

## 2. Root cause: the hate head was trained on sarcasm labels

`Phase 6 (Architectures)/distilbert-base (Lightweight model)/lightweight-model.ipynb`, cell 4:

```python
lh = batch['labels_hate'].to(device)     # loaded, moved to GPU, then NEVER used
ls = batch['labels_sarc'].to(device)

loss = ALPHA*crit_b(hl, ls) + BETA*crit_b(sl, ls) + GAMMA*crit_3(logits, l3)
#                     ^^ should be lh
```

Both auxiliary heads were optimized against the same target. Measured over 3,000 held-out rows
of the shipped checkpoints:

| | Bangla | English |
|---|---|---|
| `corr(hate_head, sarc_head)` | **0.9997** | **0.9999** |
| mean absolute difference | 0.0037 | 0.0009 |
| `corr(hate_head, y == hateful)` | **−0.36** | **−0.48** |
| `corr(hate_head, y == sarcastic)` | +0.65 | +0.89 |

### Why this is more than a bad output field

It would be easy to dismiss this as "one response field is wrong." It isn't, for two reasons.

**The fused classifier lost half its input.** `fusion_mlp` consumes
`[cls_output, hate_hidden, sarc_hidden]`. Because both branches were driven toward the same
target, the fused 3-class head — the one the extension actually acts on — received the sarcasm
signal twice and the hate signal never. The architecture's entire premise is a dedicated hate
branch feeding the fusion layer. That branch did not exist.

**It invalidates RQ1.** The research question is whether explicit sarcasm modelling via a
dedicated branch improves implicit hate detection. The dual-head model never had a working
hate branch, so no result from these checkpoints can answer it either way.

### Why nothing caught it

`evaluate()` read `logits, _, _ = model(...)` — it discarded both auxiliary heads and scored
only fused macro-F1. There was no metric anywhere in the pipeline whose value would have
changed if the heads collapsed. The corrected notebook scores all three outputs every epoch
and hard-asserts `corr(hate_head, sarc_head) < 0.95`.

---

## 3. Secondary defects

**Class weights computed and discarded.** `cw_bn = compute_class_weight('balanced', ...)` was
computed and printed, then `crit_3 = nn.CrossEntropyLoss()` was constructed with no `weight=`.
Bangla is 49 / 31 / 20 imbalanced, uncorrected.

**Train/serve preprocessing mismatch.** Models were trained on the `text_clean` column produced
by `Phase 2/Preprocessed Data.ipynb` (URLs, mentions, emoji stripped; NFC-normalized). The
extension sent **raw page text**. Now ported to `backend/app/preprocessing.py` and applied at
inference.

**Page list numbering is out-of-distribution noise.** Training text never contained
`৭। ` / `২১.` / `58। ` prefixes. Their presence measurably shifts the decision:

| text | verdict |
|---|---|
| `খাঙ্কিরপোলা` | hateful, P = 0.529 |
| `৭। খাঙ্কিরপোলা` | **non_hateful**, P(hate) = 0.454 |

Stripping the prefix moved recall 17.5% → 26.3% with no change in false-positive rate on
`bn_test` non-hateful rows — the cheapest available win, and now done in `strip_list_prefix`.

**`sarcastic` is acting as a catch-all.** Vulgar threats the model wasn't trained on land in
`sarcastic` with high confidence (`তর বাপের হোগা দিয়া ট্রেন ভইরা দিমু।` → 0.886). This is the
source-leakage problem the top-level README already identifies: BD_SHS and BIDWESH contribute
hate but no sarcasm, BenSarc and BanglaSarc3 contribute sarcasm but no hate, so the model
learned to separate *sources*, not *phenomena*. Under the old `BLUR_LABELS = new Set(["hateful"])`
rule, every one of these rendered in the clear.

**Tokenization.** Not a defect, but a ceiling: UNK rate on Bangla is 0%, yet mBERT's WordPiece
shatters Bangla into near-characters — `শুয়োরের` → `['শ','##ু','##য়','##োর','##ের']`. A
Bangla-specific backbone is the largest remaining lever.

---

## 4. Threshold tuning cannot rescue this

Sweeping the decision rule over the reported list vs. `bn_test` non-hateful rows:

| rule | recall on slurs | FP on benign |
|---|---|---|
| `argmax == hateful` (original) | 17.5% | 6.6% |
| `P(hate) ≥ 0.4` | 26.3% | 7.2% |
| `P(hate) ≥ 0.3` | 33.3% | 8.6% |
| `P(hate) + P(sarc) ≥ 0.4` | 52.6% | **21.0%** |
| `P(hate) + P(sarc) ≥ 0.3` | 57.9% | **24.6%** |

Buying ~53% recall costs a 21% false-positive rate — one benign line in five blurred. There is
no operating point on this model that is simultaneously useful and acceptable. **Retraining is
required**; thresholds only redistribute the existing error.

---

## 5. What changed

### Backend
- **`app/preprocessing.py`** (new) — ports the Phase 2 cleaning functions and adds
  `strip_list_prefix`. Applied in both `/predict` and `/predict/batch`.
- **`app/schemas.py`** — added `non_hateful_prob` / `hate_prob` / `sarc_prob` (the fused
  softmax, the only trustworthy per-class scores). `is_hateful_score` / `is_sarcastic_score`
  are marked `deprecated=True` with the reason inline; they remain in the response only so
  existing clients don't break on a missing key.
- **`app/main.py`** — lines that normalize to nothing (a bare list marker, pure punctuation)
  return a definite non-hateful verdict rather than an error, so the extension always gets
  exactly one result per item it sent.
- **`tests/test_preprocessing.py`** (new) — pins the normalization contract, including a
  regression guard that `খাঙ্কিরপোলা` and `৭। খাঙ্কিরপোলা` reach identical verdicts.

### Extension
- Independent thresholds on `hate_prob` / `sarc_prob` instead of switching on argmax `label`.
- Intensity-scaled blur (4→14 px hate, 0.5→3 px sarcasm) and tint alpha (0.15→0.55), rescaled
  from `[threshold, 1]` so conviction is actually visible.
- Red for hate, yellow for sarcasm; the **stronger signal owns the whole treatment** (colour,
  blur, alpha, tooltip), so a line never renders yellow while blurred at hate strength.
  `title` exposes the exact percentage on hover.
- Right-click corrections (see below).

### User corrections — the online complement to retraining
- **`app/feedback.py`** (new) — three-tier override store: exact on normalized text (O(1),
  before tokenization), exact ignoring whitespace, then `[CLS]` cosine similarity. Taught
  lines cost **zero model inference**: 1.1 ms vs 25.4 ms measured.
- Similarity thresholds calibrated against `ood_eval_bn.csv`, not guessed: 0.95 general,
  0.98 for taught examples under 16 characters (short strings produce near-degenerate
  embeddings — benign text reaches 0.938 against a ≤10-char taught example). Measured 7/7
  variant forms caught, 0/27 benign lines false-linked.
- `POST/GET/DELETE /feedback`; disk writes on a daemon thread, embedding index warmed on a
  background thread at startup, so nothing blocks a request.
- **This does not fix the model** — it is a targeted patch over a specific miss, and its
  reach is deliberately capped at close variants. It matters here for a second reason: every
  correction lands in `backend/data/feedback.jsonl` with the fields a retrain needs, so the
  corrections accumulate into human-verified training data drawn from exactly the
  distribution the model is weakest on.

### Research
- **`lightweight-model-corrected.ipynb`** — fixes all three training defects, evaluates the
  auxiliary heads, asserts head independence, and runs the out-of-distribution eval.
- **`ood_eval_bn.csv`** — 79 rows in four categories: `slur` (26) and `threat` (23) must be
  flagged; `named_target` (21) must **not** be; `coded` (9) is reported but excluded from
  scoring, because hatefulness there is genuinely context-dependent and there is no defensible
  standalone ground truth.

---

## 6. Recommended order of work

1. **Re-run `lightweight-model-corrected.ipynb` with `MODEL_BN` unchanged.** This isolates the
   loss fix from everything else and gives a clean before/after against the shipped checkpoint.
   Confirm `head_corr` drops well below 0.95 and `hate_head_ap` becomes meaningfully higher
   than chance.
2. **Re-run with `MODEL_BN = "sagorsarker/bangla-bert-base"`.** `hidden_size` is 768, so
   `architecture.py` and the serving stack need no change beyond the backbone string in
   `config.py`. Compare on the OOD fixture, not just `bn_test`.
3. **Address source leakage.** This is the deepest issue and the one with the most research
   value. As long as sarcasm and hate never co-occur in any single source, the model can
   separate them by style alone and will keep routing unfamiliar vulgar text to `sarcastic`.
   The Phase 4 two-dimensional annotation is exactly the right instrument — the priority is
   getting texts labelled on **both** axes at once.
4. **Re-tune thresholds on the retrained model**, then revisit whether the extension should
   read `is_hateful_score` instead of `hate_prob`. Once the heads are trained correctly, a
   dedicated binary head is better calibrated for an intensity gradient than a 3-way softmax,
   which forces hate and sarcasm to compete for probability mass even though they co-occur.
5. **Fold `backend/data/feedback.jsonl` into the training set** at whichever retrain comes
   next. Two cautions worth stating before it is used as data rather than as a runtime patch:
   it is *non-random* — it samples only what the model already got wrong and the user
   happened to notice — so it will inflate apparent gains if it is allowed to leak into an
   evaluation split. Keep it strictly in `train`, and keep `ood_eval_bn.csv` as the held-out
   yardstick. It is also single-annotator, so it carries none of the inter-annotator
   agreement the Phase 4 protocol is designed to produce.
