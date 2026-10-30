# The whole system, in plain language

*A reviewer's walkthrough. No prior knowledge of the code assumed.*

---

## Table of contents

1. [What the product does](#1-what-the-product-does)
2. [The four moving parts](#2-the-four-moving-parts)
3. [The journey of one line of text](#3-the-journey-of-one-line-of-text)
4. **[What happens when you hold for 450 ms and make a correction](#4-what-happens-when-you-hold-for-450-ms-and-make-a-correction)** ← the question this doc was written for
5. [The four layers of learning](#5-the-four-layers-of-learning)
6. [The safety story](#6-the-safety-story-what-cannot-go-wrong)
7. [Numbers a reviewer might ask for](#7-numbers-a-reviewer-might-ask-for)
8. [Honest limitations](#8-honest-limitations)
9. [Likely reviewer questions, answered](#9-likely-reviewer-questions-answered)

---

## 1. What the product does

A Chrome extension reads the text on whatever page you're viewing, sends it to a small local
server, and **blurs** anything the model calls hateful (red) or **tints** anything it calls
sarcastic (yellow). You can press and hold on any flagged line to read it, and tell the
system it got the answer wrong.

The two models are the capstone's own research output: DistilBERT dual-head classifiers,
one English, one Bangla, ~135 million parameters each.

> **The one-sentence version:** the page is broken into lines, only lines about to appear on
> screen are scored, results are cached, and every correction you make is remembered
> permanently — taking effect instantly, and eventually retraining the model itself.

---

## 2. The four moving parts

```
┌──────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  content_script.js   │     │    background.js     │     │   FastAPI backend   │
│  (runs on the page)  │────▶│ (the extension's     │────▶│   (localhost:8000)  │
│                      │     │  network layer)      │     │                     │
│ • finds text         │     │                      │     │ • 2 DistilBERT      │
│ • decides what to    │     │ Exists only because  │     │   models            │
│   send               │     │ a page-side fetch to │     │ • correction store  │
│ • blurs / tints      │     │ localhost triggers a │     │ • result cache      │
│ • press-and-hold UI  │     │ Chrome permission    │     │ • retrain pipeline  │
└──────────────────────┘     │ prompt               │     └─────────────────────┘
                             └──────────────────────┘                │
                                                                     ▼
                                                          ┌─────────────────────┐
                                                          │ feedback.jsonl      │
                                                          │ every correction,   │
                                                          │ kept forever        │
                                                          └─────────────────────┘
```

**Why a separate server instead of running the model in the browser?** The checkpoints are
~500 MB each. Loading them once into a resident Python process and answering over HTTP is
the difference between ~40 ms and "not possible".

---

## 3. The journey of one line of text

**Step 1 — Find the text.** A `TreeWalker` walks the page's text nodes. Text is grouped into
*lines*: consecutive nodes under the same block-level parent, split on `<br>`. This is what
correctly reassembles a sentence that a site chopped into four `<span>`s.

**Step 2 — Throw away page furniture.** `<nav>`, `<aside>`, `<footer>` and their ARIA
equivalents are skipped entirely. This is the single most important accuracy decision in the
project, and it isn't a model change at all:

| on the real test page | before | after |
|---|---|---|
| lines sent to the model | 652 | **185** |
| wrong flags on English | 41 | **9** |
| real article content lost | — | **0** |

The reason it matters so much: menu labels, tag clouds, timestamps and usernames are
*nothing like* the sentences the model was trained on, and it flags them at a far higher
rate than real prose. Every attempt to fix this inside the model moved the number by single
digits. Not asking the model was worth 32 false positives and 71% of the compute.

**Step 3 — Wait until it's nearly on screen.** An `IntersectionObserver` holds each line
until it's within 200 px of the viewport. Scoring a 5,000-line page you never scroll is
wasted work.

**Step 4 — Batch.** Lines are collected for 200 ms and sent up to 50 at a time.

**Step 5 — Score and decide.** The backend returns three probabilities. The extension
decides:

```
blur red      if P(hateful)   ≥ 0.50
tint yellow   if P(sarcastic) ≥ 0.95   (English)  /  0.85 (Bangla)
```

Only **one** colour is applied — whichever signal is stronger. Sarcasm tints but never
blurs, because a false yellow should be a hint, not a wall.

> **Why is the English sarcasm bar so high?** Because that head is broken. On these
> checkpoints, "He works as an engineer." scores 0.998 sarcastic. The scores pile up at
> ~0.00 and ~0.99 with nothing in between, so no threshold separates the classes cleanly.
> 0.95 is damage control, and the documentation says so rather than pretending otherwise.

---

## 4. What happens when you hold for 450 ms and make a correction

*This is the reviewer's question, answered end to end.*

### 4.1 The gesture

You press the mouse down on a blurred line and hold. A timer starts.

- **Release before 450 ms** → nothing happens. It was a click.
- **Move more than 6 px** → cancelled. You were selecting text or dragging.
- **Hold past 450 ms** → the review opens.

> **Why a hold and not a double-click?** It used to be a double-click. On a feed, flagged
> text is very often inside a link, and `dblclick` only fires *after* two complete clicks —
> by which time the browser has already navigated away. A hold is detected during
> `mousedown`, and the click that follows is swallowed in the capture phase before the link
> can act. This was a real bug, found on a real page, and the gesture was changed because of
> it.

### 4.2 What you see

The line un-blurs (the whole line, including every fragment it was split across), becomes
selectable, and a small toolbar appears saying e.g. *"flagged sarcasm 97% —"* with three
buttons:

| button | meaning |
|---|---|
| **Hateful** | red |
| **Sarcastic** | yellow |
| **Neither** | not a problem, leave it alone |

The toolbar lives in a **closed shadow DOM**, so the page's own CSS and JavaScript cannot
style it, read it, or click it.

You may also **select just a few words** before choosing. If you do, the correction applies
to your selection rather than the whole line — more precise, because on a long sentence
where only two words are the slur, teaching the whole sentence teaches a lot of irrelevant
context along with it.

### 4.3 What happens the instant you click

```
you click "Neither"
      │
      ├──▶  the line un-blurs on screen                              ~0 ms   (immediate)
      │
      ├──▶  POST /feedback  ──────────────────────────────────────▶  ~70 ms
      │        1. normalize the text (same cleaning as training)
      │        2. embed it → a 768-number vector
      │        3. WRITE TO DISK, synchronously                       0.06 ms
      │        4. rebuild the correction cache                       0.07 ms
      │        5. advance the retrain counter
      │
      └──▶  re-score every line on the page that matches           ~18 ms
                (debounce skipped — this is a direct user action)
```

**Total: roughly a tenth of a second from click to the page being correct.**

Three details a reviewer is likely to probe:

- **The disk write is synchronous, on purpose.** An earlier version wrote on a background
  daemon thread, and corrections were being *silently lost* when the process exited with
  work still queued. It costs 0.06 ms. There is a regression test that reads the file
  without flushing to make sure this never comes back.
- **Nothing waits for training.** The correction is in force before the HTTP response is
  written.
- **Only matching lines are re-scored**, not the whole page. Blanket-clearing every flag
  would briefly expose unrelated hateful text.

### 4.4 What happens on the *next* line of text

This is the part that matters, and it's the difference between a lookup table and learning.

When any line is scored from now on, the backend tries four things in order — cheapest and
safest first:

| # | tier | what it catches | cost |
|---|---|---|---|
| 1 | **exact match** | the identical text, ignoring case/spacing/list numbering | ~0 |
| 2 | **similarity match** | near-identical text — spelling variants, "হারামি" vs "হারামী" | one matmul |
| 3 | **correction cache** | text that is merely *similar in meaning*, never taught | 0.08 ms per 50 lines |
| 4 | the model | everything else | ~40 ms |

Tiers 1 and 2 are why the exact line you corrected is fixed. **Tier 3 is why lines you never
touched get fixed too.**

**Measured, live:** teaching one sentence moved an unseen near-paraphrase from
`P(sarcastic) = 0.995` to `0.008` — with no training step of any kind.

### 4.5 If you change your mind

Delete the correction and its key disappears from the cache. The near-paraphrase above went
straight back to 0.995. Undo is as immediate as the correction was.

### 4.6 And then, at 150 corrections

The counter advances by one. When a language reaches **150 new corrections**, the backend
starts a real fine-tune of that language's 135M-parameter model in a **separate process**
(so the extension keeps working normally throughout). See
[§5, layer 4](#layer-4-retraining-the-actual-model).

---

## 5. The four layers of learning

The system remembers a correction in four different ways, at four different timescales.
This is the core of the design, so it's worth seeing them together:

| layer | when it acts | changes model weights? | scope | how to undo |
|---|---|---|---|---|
| 1. exact match | instantly | no | that exact text | delete the correction |
| 2. similarity match | instantly | no | near-identical text | delete the correction |
| 3. **correction cache** | instantly | no | text similar in *meaning* | delete the correction |
| 4. **retraining** | every 150 corrections | **yes** | the whole model | `/retrain/rollback` |

### Layer 3: the correction cache

Every correction is kept as a point in the model's "meaning space". When new text arrives,
the **nearest** correction votes on it, and the vote fades with distance:

```
vote(x) = α · max{ exp(−β · distance(x, correction ᵢ)) }
```

Plain English: *"if this new text is very close to something you already corrected, apply
that correction strongly; if it's far away, leave it completely alone."*

**Why `max` and not a sum?** This method comes from a published technique (Tip-Adapter) that
**adds up** every correction's vote. We implemented that first and it fails: the total grows
as the store fills, so settings tuned at 78 corrections destroyed sarcasm detection by 156
(9/12 → 1/12 correct). Taking the **maximum** instead depends only on the single nearest
correction, so adding a thousand irrelevant ones changes nothing. Verified byte-identical
from 78 corrections to 4,992.

This layer needs **no training at all** — which is why it works from correction #1.

> **A trained alternative was tried and rejected.** A small trained classifier ("linear
> probe") was built, measured against the live server, and found to make chrome false
> positives *worse* (37 → 42). It is still in the code but **off by default**, with the
> measurement recorded next to the switch.

### Layer 4: retraining the actual model

Every 150 corrections, one language's real checkpoint is fine-tuned.

**The training data is not just the corrections.** Training on 150 corrections alone
destroys the model — and that isn't a guess, it was measured:

| replay data mixed in | sarcasm recall after | outcome |
|---|---|---|
| 300 rows | 0.750 → **0.333** | **rejected** |
| 6,000 rows | 0.750 → **0.750** | promoted |

So each run mixes ~1,200 correction examples with **6,000 class-balanced rows of the
original training data**. That's what stops the model forgetting everything it knew.

**Nothing is promoted automatically.** The new model is scored against a frozen test set and
must clear a gate:

1. it must actually *learn* the corrections (≥ +5 points), **and**
2. **no protected metric may degrade** beyond its tolerance:

| protected metric | may drop by at most |
|---|---|
| overall macro-F1 | 0.01 |
| sarcasm recall | 0.02 |
| hate recall | 0.02 |
| Bangla slur recall | 0.02 |
| false positives on benign text | may *rise* by 0.02 |

A real run, on real corrections, default settings:

```
corrections    0.000 → 1.000   (+1.000)  ok      training time: 29 minutes
macro_f1       0.897 → 0.892   (−0.005)  ok
sarcasm_recall 0.750 → 0.750   (+0.000)  ok
hate_recall    0.912 → 0.905   (−0.007)  ok
slur_recall    1.000 → 1.000   (+0.000)  ok
benign_fp_rate 0.500 → 0.500   (+0.000)  ok
VERDICT: PROMOTED
```

If it fails, the production model is untouched, **the counter is not reset**, and
corrections keep accumulating for the next attempt. Every run — including rejected ones —
writes its model, metrics and a plain-text report to disk, so a refusal is inspectable
rather than invisible.

---

## 6. The safety story: what *cannot* go wrong

| worry | why it can't happen |
|---|---|
| "Retraining ruins the model" | The gate blocks it, and the original checkpoint is never overwritten |
| "We can't go back" | Rollback is a pointer flip: `POST /retrain/rollback` then `/retrain/reload` |
| "A corrupted checkpoint bricks the server" | A missing or unreadable checkpoint falls back to the original model rather than failing to start |
| "My corrections get lost in a retrain" | `feedback.jsonl` is append-only and never deleted by any pipeline step |
| "Corrections vanish on a crash" | Written to disk synchronously, before the response returns |
| "Training freezes the extension" | It runs in a separate OS process |
| "The 450 ms hold opens a link instead" | The click is swallowed in the capture phase |
| "The page can tamper with the toolbar" | Closed shadow DOM |
| "The reviewer's browsing gets committed to git" | `backend/data/` is gitignored |

---

## 7. Numbers a reviewer might ask for

**Speed** (real page, 652 raw lines):

| | |
|---|---|
| Model forward pass, share of total time | 99.2% |
| Cold first load | 1,630 ms |
| Full page re-score, warm | **18 ms** |
| `POST /feedback` round trip | **70 ms** |
| Correction cache applied to 50 lines | 0.08 ms |
| Full fine-tune (CPU, no GPU) | ~29 min English, ~35–40 min Bangla |

**Accuracy on the real page** (185 lines after furniture skipping):

| | base model | shipped config |
|---|---|---|
| English false positives | 9 | **7** |
| genuine sarcasm caught | 9/12 | **9/12** (unchanged) |
| genuine hate caught | 6/8 | **6/8** (unchanged) |
| Bangla slurs caught | 26/49 | **28/49** |
| Bangla benign wrongly flagged | 0/41 | 1/41 |

The cache was deliberately tuned to the strongest setting that costs **zero** genuine
detections.

**Tests:** 93 automated tests, covering the cache's size-invariance, the gate's rejection
of every failure mode, promotion/rollback, and correction durability.

---

## 8. Honest limitations

These are real and should be stated before a reviewer finds them.

1. **The auxiliary heads are broken in the shipped checkpoints.** The original training loop
   optimised the hate head against the *sarcasm* target. The two heads correlate at 0.9997,
   and the hate head *anti*-correlates with actual hate. The product works around this by
   using only the 3-class output. A corrected notebook exists; the fix has not yet been
   retrained into the shipped models.

2. **English sarcasm is saturated and cannot be fixed by thresholding.** ~35% of ordinary
   benign English sentences still score above 0.95.

3. **Source leakage.** The models partly learned to tell their *training corpora* apart
   rather than the two phenomena, so `sarcastic` behaves as a catch-all for anything
   unfamiliar. Fixing this needs the two-axis Phase 4 annotations, not more tuning.

4. **Corrections are a biased sample** — only what the model got wrong *and* a human
   noticed. They belong in training data, never in an evaluation split. The gate's
   regression set is kept strictly separate for this reason.

5. **Single annotator.** The corrections carry none of the inter-annotator agreement the
   Phase 4 protocol produces.

---

## 9. Likely reviewer questions, answered

**"If I hold 450 ms and mark a line, when does the model actually change?"**
Two different things happen. The *behaviour* changes in about 90 ms — the correction is
stored and the page is re-scored before you've let go of the mouse. The *model weights*
change only after 150 corrections, take ~29 minutes, and only if the result passes the
quality gate. Both are deliberate: you get an instant fix without waiting, and a permanent
fix without risk.

**"Does one correction only fix that one line?"**
No. That was the first version's flaw, and it was measured: after teaching two examples,
18 of 20 similar unseen strings were still wrongly flagged. The correction cache fixes that
— a taught sentence pulled an unseen paraphrase from 0.995 to 0.008.

**"What if I make a mistake and mark something wrongly?"**
Delete the correction; its effect disappears immediately. If it has already been absorbed
into a retrained model, roll that checkpoint back.

**"What stops 150 corrections from wrecking a model trained on 83,000 examples?"**
Three things: corrections are only ~17% of each training batch, the rest being replayed
original data; the gate rejects any candidate that degrades a protected metric; and the old
checkpoint is kept for rollback. We demonstrated the failure deliberately — with too little
replay data, sarcasm recall collapsed to 0.333 and the gate refused to promote it.

**"Why did skipping `<nav>` help more than all the machine learning?"**
Because the classifier was never the problem. Menu labels and timestamps aren't sentences,
and asking a sentence classifier about them produces noise. We tried six adaptation methods
from the literature (linear probing, DFR, AFR, BitFit, partial fine-tuning, cache models);
the best moved false positives by single digits. One DOM rule removed 32 of them and made
the page 3.5× cheaper. **It is worth knowing which problems are not model problems.**

**"Is the 450 ms number tuned?"**
It's a standard long-press interval — long enough not to fire on an ordinary click, short
enough not to feel stuck. The 6 px movement tolerance is the more carefully chosen number:
small enough that a shaky hand still triggers a hold, large enough that starting a text
selection reliably cancels it.

**"How do I see what the system is doing right now?"**

```bash
curl localhost:8000/adapter    # how many corrections are active, per language
curl localhost:8000/retrain    # progress to next fine-tune, which model is live, run history
curl localhost:8000/feedback   # every correction stored
```

---

## Where to read more

| document | covers |
|---|---|
| [HOW_IT_WORKS.md](HOW_IT_WORKS.md) | the full technical walkthrough, with code |
| [RETRAINING.md](RETRAINING.md) | the retraining pipeline, and training on a new dataset |
| [MODEL_DEFECTS_AND_DETECTION.md](MODEL_DEFECTS_AND_DETECTION.md) | the auxiliary-head defect in detail |
| [../SETUP.md](../SETUP.md) | running it on a new machine |
