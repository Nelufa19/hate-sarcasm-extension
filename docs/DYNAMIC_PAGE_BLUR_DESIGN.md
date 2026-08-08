# Design Doc: Detecting and Blurring Hate Speech on Live, Dynamic Pages

## The problem, stated precisely

Everything built so far (`/predict` + the popup) is **pull-based and manual**: the user selects text, clicks a button, gets one answer. What you're describing now is different in kind, not just in size: **the extension should scan a page automatically, find hateful/sarcastic-hateful content on its own, and visually cover it before the user reads it — and keep doing that as the page changes.**

The hard part is the second half of that sentence. A static page (view source, all text present at load) would be almost easy — walk the DOM once, send every text node to `/predict`, blur the bad ones, done. But real pages you'd actually want this on — Twitter/X, Facebook, Reddit, YouTube comments, news comment sections — are **dynamic**: content is added continuously as the user scrolls (infinite scroll), as a single-page app (SPA) navigates without a full page reload, and sometimes edited or removed in place. "Scan once at page load" would miss almost everything a user actually sees.

So the real design question is: **how do you keep noticing new content forever, without re-scanning the whole page forever, without hammering your own API, and without janking the page's scroll performance?** That's four sub-problems. This doc addresses each one.

---

## Part 1 — How do you know when new content appears? (`MutationObserver`)

### The API

`MutationObserver` is a browser API built exactly for this: you point it at a DOM node, and it calls your callback whenever children are added/removed/changed underneath it — without you polling.

```javascript
const observer = new MutationObserver((mutationsList) => {
  for (const mutation of mutationsList) {
    for (const node of mutation.addedNodes) {
      if (node.nodeType === Node.ELEMENT_NODE) {
        handleNewElement(node);
      }
    }
  }
});

observer.observe(document.body, {
  childList: true,   // watch for added/removed children
  subtree: true,     // watch the whole tree under body, not just direct children
});
```

This is how the extension "knows" a new tweet/post/comment just got inserted by the page's own JavaScript (Twitter's React app, Reddit's infinite scroll, etc.) — it never has to ask the page anything; the browser tells it.

### The trap: watching too broadly

Searches confirm this is the single most common mistake: `subtree: true` on `document.body` on a page like Twitter can fire **thousands of mutation callbacks per second** during active scrolling, because React re-renders touch huge swaths of the DOM for reasons that have nothing to do with new content (style recalculation, focus changes, tiny attribute updates). Naively processing every mutation will visibly freeze the page.

**Mitigations, in order of importance:**
1. **Filter mutation types.** Only care about `mutation.addedNodes` with `nodeType === Node.ELEMENT_NODE` — ignore text-node tweaks and attribute-only mutations you don't need.
2. **Debounce the handler**, not the observer itself (`MutationObserver` batches natively already — the browser coalesces rapid mutations into fewer callback firings — but a burst of legitimate new-post insertions during fast scrolling can still arrive faster than you want to process them one at a time). A `setTimeout`-based debounce collecting mutations for ~150–300ms before processing the batch as one unit is the standard pattern.
3. **Target the narrowest container you can, not `document.body`.** On most feed-style sites there's an identifiable feed/timeline container (`<div role="feed">`, a specific class name) — observing *that* instead of the whole page cuts irrelevant mutation noise dramatically. This does mean the content script needs a **per-site selector** (see Part 4), which is real complexity, not a nice-to-have.
4. **Track already-processed nodes with a `WeakSet`**, not a regular array or a DOM attribute check alone — `WeakSet` lets processed nodes be garbage-collected once removed from the DOM (e.g. scrolled far away and unmounted by the page's virtualized list), preventing an ever-growing memory leak over a long browsing session.

```javascript
const processedNodes = new WeakSet();

function handleNewElement(node) {
  if (processedNodes.has(node)) return;
  processedNodes.add(node);
  queueForScanning(node);
}
```

5. **Always call `observer.disconnect()`** when the extension is toggled off or the tab navigates away (SPA route change) — an orphaned observer with no corresponding cleanup is a classic memory leak in long-lived single-page apps, since the observer keeps a live reference to DOM subtrees that would otherwise be freed.

---

## Part 2 — Don't scan things the user can't even see (`IntersectionObserver`)

### Why this matters specifically for your use case

A feed page might have hundreds of posts in the DOM at once (React/Twitter keep a large virtualized window loaded even off-screen for smooth scrolling). Sending all of them to `/predict` the instant they're inserted — most of which the user will never scroll to — wastes API calls, wastes your backend's compute, and adds latency to the posts the user actually cares about right now.

`IntersectionObserver` solves the complementary problem to `MutationObserver`: it tells you when an element **enters the viewport** (or a margin around it), regardless of how it got into the DOM.

```javascript
const viewportObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        scanElement(entry.target);
        viewportObserver.unobserve(entry.target); // scan once, then stop watching it
      }
    }
  },
  { rootMargin: "200px" } // start scanning slightly *before* it's actually visible
);
```

`rootMargin: "200px"` is the key tuning knob — it grows the "viewport" by 200px on every side, so content is scanned and (if needed) already blurred *before* the user's eyes actually reach it, avoiding a visible "flash of unblurred content" followed by a jarring cover-up a moment later.

### The combined pipeline

```
MutationObserver
    │  "a new post-like element was inserted somewhere on the page"
    ▼
IntersectionObserver (attached to that new element)
    │  "...and now it's about to scroll into view"
    ▼
extract text → queue for /predict → render blur if hateful
```

This two-observer combination is the actual pattern found in real production content-scanning extensions (translation extensions, ad blockers with content analysis, and the hate-speech extensions surveyed) — `MutationObserver` answers "what's new," `IntersectionObserver` answers "what's worth spending an API call on right now."

---

## Part 3 — Don't hammer your own FastAPI server (batching + rate limiting)

### The problem

A user scrolling quickly through a busy feed could trigger dozens of "newly visible" events per second. Firing one `POST /predict` per post, un-throttled, will: overload a single-process `uvicorn` dev server instantly, and even against a properly scaled backend, burns API quota/compute for zero user benefit if they scroll past in half a second anyway.

### Client-side mitigations, combined

1. **Debounce the queue, then flush in batches.** Instead of firing a request the instant a post becomes visible, collect visible-and-unscanned posts for ~150–250ms, then send them together. This requires the backend to accept a **list** of texts per request, not just one — a real change needed from the current single-text `/predict` (see Part 5).
2. **Cap concurrency.** Even with batching, don't fire unlimited batches at once — track a small number of "in-flight" requests (e.g. max 2–4 concurrent batch calls) and queue the rest, so a sudden burst of scrolling doesn't open 50 simultaneous connections to your own server.
3. **Client-side result caching.** If the same text appears twice (retweets/reposts, or a post re-rendered by a virtualized list scrolling back into view), keyed by a hash of the text, skip the API call and reuse the cached verdict. This is a real, common win on feeds where content reflows.
4. **Token-bucket style self-throttling** as a hard backstop: track a rolling count of requests in the last N seconds and drop (or delay) new scan requests past a ceiling, so a pathological page (or a bug in your own mutation-handling) can't runaway-loop your backend.

### Why this isn't optional

Without steps 1–2 specifically, this feature would functionally be a self-inflicted denial-of-service against your own FastAPI server the moment it's used on a real feed — this is the single most important non-obvious engineering point in this whole design, more likely to actually break in practice than any modeling concern.

---

## Part 4 — Extracting "one unit of text" from an arbitrary page

### Why this is harder than it sounds

`/predict` wants "a piece of text." A live webpage doesn't hand you clean units of text — a single tweet/post is a `<div>` containing nested spans, emoji images, mention links, timestamps, and UI chrome (like/reply buttons) all mixed together. Naively taking `element.innerText` on too broad a container captures noise (button labels, timestamps) alongside the actual message; too narrow, and you fragment one sentence across multiple sibling `<span>` elements and analyze meaningless fragments.

### Two realistic strategies

**Strategy A — generic heuristic extraction (works everywhere, imperfect):**
Walk the DOM under a MutationObserver-flagged new node, find the largest contiguous block of visible text (heuristics: reasonable character count, not inside a `<button>`/`<a class="ui-*">`, not matching common "timestamp" or "X likes" patterns), and treat that as one unit. This is what any "must work on any random webpage" extension has to fall back to — real accuracy will be inconsistent, but it never needs per-site maintenance.

**Strategy B — per-site selectors (accurate, needs maintenance):**
Maintain a small config mapping known domains to CSS selectors for "this is a post/comment container" (e.g., Twitter/X's `article[data-testid="tweet"]`, Reddit's comment `<div>` class). This is dramatically more accurate — you get exactly the post text, cleanly — but it breaks whenever the target site redesigns its frontend (a real, recurring maintenance cost that every scraping-adjacent browser extension eventually deals with), and it only covers sites you've explicitly added.

```javascript
const SITE_SELECTORS = {
  "twitter.com": '[data-testid="tweetText"]',
  "x.com": '[data-testid="tweetText"]',
  "reddit.com": '[data-testid="comment"]',
  // fallback: generic heuristic extraction for everything else
};
```

**Recommendation for your MVP:** ship Strategy B for one or two platforms you actually demo against (pick whichever you'll show in an interview/demo — Twitter/X is the most recognizable), and fall back to Strategy A everywhere else so the extension "does something reasonable" on arbitrary pages without claiming full accuracy there. Don't try to build a universal, perfectly accurate extractor for every possible site layout — that's an open-ended problem real companies dedicate teams to, not a junior-project scope.

---

## Part 5 — What actually changes in your existing FastAPI backend

Your current `/predict` (see `CODE_WALKTHROUGH.md`) takes one `text` string and returns one prediction. Automatic page-scanning needs a **batch endpoint** so the client can send the debounced group of newly-visible posts in one HTTP round trip instead of one-request-per-post (which multiplies network overhead by however many posts appear at once).

```python
class PredictBatchRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=50)

class PredictBatchResponse(BaseModel):
    results: list[PredictResponse]

@app.post("/predict/batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    return PredictBatchResponse(results=[predict(item) for item in req.items])
```

This is a genuinely good `max_length=50` cap to add — it stops a client bug (or a malicious request) from asking the server to run 10,000 inferences in one HTTP call and blocking the whole process. The real, higher-performance version of this endpoint would tokenize all texts together and run **one batched forward pass** through the model instead of a Python loop calling `predict()` N times (a GPU/CPU batch of 32 texts is far more efficient than 32 sequential single-item calls) — worth naming as a known future optimization, not necessary for a working MVP.

---

## Part 6 — Actually covering the content (CSS)

### Two real approaches, and why one is usually better here

**`filter: blur(Npx)` directly on the content element:**
```css
.hs-blurred {
  filter: blur(8px);
  user-select: none;
  transition: filter 0.2s ease;
}
.hs-blurred.hs-revealed {
  filter: none;
}
```
Simple, cheap, works everywhere. The downside: it blurs the *element itself*, so if the element also contains an avatar image or a "reply" button nested inside, those get blurred too — usually undesirable when you specifically want to hide only the offending *text*, not surrounding UI.

**A separate overlay with `backdrop-filter: blur(Npx)`:**
```css
.hs-overlay {
  position: absolute;
  inset: 0;
  backdrop-filter: blur(8px);
  background: rgba(0, 0, 0, 0.15);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}
```
This blurs *what's behind* the overlay without altering the original element's own styling, and lets you place a "⚠ Possibly hateful content — click to reveal" label on top cleanly. The tradeoff, confirmed by research: `backdrop-filter` is more GPU-intensive than plain `filter` since the browser has to render what's behind it first, then blur, then composite — noticeable if you're blurring dozens of feed items simultaneously during fast scrolling. For a feed with many blurred items, prefer plain `filter: blur()` on the smallest reasonable wrapper around just the text; reserve `backdrop-filter` overlays for cases where you specifically need the "click to reveal" label rendered on top.

### Click-to-reveal behavior

```javascript
overlay.addEventListener("click", () => {
  overlay.remove();
  textElement.classList.add("hs-revealed");
});
```

This matches the standard "spoiler/content-warning" UX pattern users already know from Twitter's own sensitive-content blur and Reddit's spoiler tags — reuse a familiar interaction instead of inventing a new one.

### Don't forget the false-positive escape hatch

Any model makes mistakes (your own capstone's Phase 5 error analysis exists precisely because baseline models misclassify sarcasm and hate). The overlay/reveal pattern *is* the escape hatch — the user is never fully blocked from the content, just given a warning + one click. Never auto-delete or hide content with no way to see it; that turns a model mistake into actual censorship of a real user's speech, which is both a bad UX call and a legitimate ethical concern for a tool operating on other people's text.

---

## Part 7 — Putting the whole pipeline together

```
Page loads / SPA navigates
        │
        ▼
content_script.js starts:
  1. Look up SITE_SELECTORS for this domain (or fall back to generic mode)
  2. Attach MutationObserver to the feed container (not document.body)
        │
        ▼
New post/comment appears in DOM (scroll-triggered, or SPA route change)
        │
        ▼
MutationObserver callback fires (debounced ~200ms, batches mutations)
        │
        ▼
For each newly-added, not-yet-processed node (tracked via WeakSet):
  attach an IntersectionObserver (rootMargin: 200px)
        │
        ▼
Node scrolls near the viewport → IntersectionObserver fires
        │
        ▼
Extract text (per-site selector or generic heuristic)
        │
        ▼
Add to a debounced batch queue (~200ms window, ≤50 items)
        │
        ▼
POST /predict/batch  ──over HTTP──▶  FastAPI server (already-loaded models)
        │
        ▼
Response: per-item label + confidence
        │
        ▼
For items labeled "hateful" (and, per your framework, "sarcastic" carrying
hateful intent — this is exactly the RQ1 distinction your capstone studied):
  inject a blur wrapper / overlay around just that text node
        │
        ▼
User sees a covered post with a "click to reveal" label
```

---

## Part 8 — Honest scope check before you build this

This is meaningfully harder than what you already shipped (`/predict` + manual popup), and it's worth being explicit about why, going into it:

- **The FastAPI/model layer barely changes** (one new batch endpoint). The hard, new work is almost entirely in `content_script.js` — DOM observation, extraction, batching, rendering — which is a different skillset (frontend/DOM performance) than what you've built so far (backend/ML serving).
- **Per-site selectors are real, ongoing maintenance**, not a one-time task — treat "supports Twitter/X" as a specific, scoped, demoable claim, not "works on the whole internet."
- **This is the part of your project most likely to visually impress in a demo** (an actual live-blurring extension is a strong portfolio piece) but also the part most likely to have rough edges under real-world page complexity (ads, A/B-tested layouts, React re-renders) — budget real debugging time against a live site, not just against a local test HTML page.

### Suggested build order (smallest working thing first)
1. Static test page you control (plain HTML with a few `<div class="post">` blocks) — prove MutationObserver + IntersectionObserver + blur/reveal works at all, no rate-limit concerns yet.
2. Add the debounced batch queue and `/predict/batch` endpoint — prove batching works against your real backend.
3. Point it at one real site (Twitter/X) with a hardcoded selector — this is where you'll hit the real mess (nested spans, virtualization, SPA navigation).
4. Only after that works reliably: consider a second site, or a generic fallback extractor for arbitrary pages.

Don't start at step 3 — you'll be debugging DOM-observation logic and Twitter's actual markup simultaneously, which is much harder to reason about than isolating one variable at a time.

---

Sources consulted for this design (browser extension architecture patterns, MutationObserver/IntersectionObserver performance guidance, and CSS blur/reveal techniques):
- [Mutation Observers to use in Extensions — Chromium extensions group](https://groups.google.com/a/chromium.org/g/chromium-extensions/c/PN-sVWAAzUk)
- [Behind the Curtain: Using the MutationObserver for Performance Optimization](https://fsjs.dev/behind-the-curtain-mutationobserver-performance-optimization/)
- [Manifest - content scripts | Chrome Extensions | Chrome for Developers](https://developer.chrome.com/docs/extensions/reference/manifest/content-scripts)
- [Intersection Observer API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API)
- [How I Turned a Chrome Extension Into a Real-Time Page Translation Tool](https://medium.com/@brendancjz/how-i-turned-a-chrome-extension-into-a-real-time-page-translation-tool-429467ebee40)
- [Revealing Blurred Element On Hover/Click - SPOILER ALERT!](https://www.cssscript.com/revealing-blurred-element-hoverclick-spoiler-alert/)
- [How to use the CSS backdrop-filter property - LogRocket Blog](https://blog.logrocket.com/use-css-backdrop-filter-property/)
- [API Rate Limiting in Chrome Extensions: The Complete 2025 Implementation Guide](https://bestchromeextensions.com/2025/01/18/chrome-extension-api-rate-limiting/)
- [GitHub - swadhadwivedi/chrome-hate-text-detector](https://github.com/swadhadwivedi/chrome-hate-text-detector)
