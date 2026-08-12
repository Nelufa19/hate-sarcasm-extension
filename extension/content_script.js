/**
 * Content script: watches the page for new text, scans lines that scroll
 * into view, and blurs any classified as hateful.
 *
 * There is no warning label and no casual click-to-reveal: a blurred line
 * stays blurred, and turning the extension off from the popup reveals
 * everything currently blurred on the page.
 *
 * The one exception is reviewing a single line -- press and hold on it and it
 * un-blurs, becomes selectable, and offers "Hateful / Sarcastic / Neither".
 * That exists because the strict rule made mistakes uncorrectable: to say
 * "this was wrongly flagged" you have to be able to read it first, and
 * blurred text also carries `user-select: none` so it can't even be
 * selected. The reveal is one line at a time and re-blurs as soon as the
 * review ends, unless the user marks the line clean.
 *
 * Pipeline:
 *   MutationObserver               -> "new text was inserted somewhere"
 *   TreeWalker (SHOW_TEXT|ELEMENT) -> "here are the text nodes, and any
 *                                      <br> tags between them, in order"
 *   line grouping                  -> "these adjacent nodes form one
 *                                      line/sentence" (split on block-tag
 *                                      boundaries or a <br>)
 *   IntersectionObserver           -> "...and it's about to scroll into view"
 *   extract text -> debounced batch queue -> POST /predict/batch
 *   result -> blur (or leave alone)
 *
 * ---- Why this replaced a Readability.js-derived scoring system ----
 * An earlier version of this file ported real Readability.js scoring (tag
 * weights, class/id regex scoring, ancestor score propagation, repeated-list
 * detection, structural-region splitting, inline-vs-block distinctions) to
 * pick "the one right DOM element" per post before sending it for
 * classification. That approach kept breaking: fixing one real page's edge
 * case (a Facebook post rendered as ~50 sibling <div> lines) broke a
 * previously-working case (a tweet's sentence fragmented across sibling
 * <span>s), and vice versa, across many iterations -- because element-level
 * scoring is inherently ambiguous about what counts as "one unit."
 *
 * The fix, confirmed against a real, mature, production analogue
 * (FrostCo/AdvancedProfanityFilter, github.com/FrostCo/AdvancedProfanityFilter
 * -- an extension that already solves the closely analogous problem of
 * "find bad words/phrases in arbitrary live page text and visually suppress
 * them"): walk TEXT NODES directly via `document.createTreeWalker(root,
 * NodeFilter.SHOW_TEXT)`, not elements. Filtering becomes trivial (skip a
 * short list of non-content tags) instead of a multi-rule scoring system,
 * because a stray "127" or "Reply" at the text-node level just never
 * matches anything and is harmless to send for classification.
 *
 * Critically, that same real extension NEVER re-searches the DOM for text
 * after processing -- it keeps the live node reference from the TreeWalker
 * and acts on that exact node directly. This project adapts that principle
 * for an async classifier: each queued "line" keeps direct references to
 * every DOM element that contributed to it, so blurring a hateful result
 * never needs to search-and-match a string back into the page.
 *
 * The one problem this does NOT solve for free -- confirmed true even in
 * the real reference extension, which treats it as a separate, harder,
 * opt-in feature -- is "a sentence split across sibling inline elements"
 * (e.g. a tweet's text as 4 sibling <span>s). This file's fix: walk
 * consecutive inline-tag siblings under the same block-level parent and
 * merge them into one line before classifying, using an offset-free
 * approach (concatenate their text; since a flagged line gets its whole
 * set of contributing elements blurred together, no sub-string range
 * tracking is needed, unlike a grammar checker marking one typo).
 */
(function () {
  "use strict";

  const MUTATION_DEBOUNCE_MS = 200;
  const BATCH_DEBOUNCE_MS = 200;
  const BATCH_MAX_ITEMS = 50;
  const VIEWPORT_ROOT_MARGIN = "200px";

  // ---- scoring thresholds & intensity mapping ---------------------------
  // Decisions are made on the fused 3-class softmax (hate_prob / sarc_prob),
  // NOT on the response's is_hateful_score / is_sarcastic_score. Those two
  // fields come from the model's auxiliary sigmoid heads, which are broken in
  // every currently-shipped checkpoint: the training loop optimized the hate
  // head against the sarcasm target, so is_hateful_score is effectively a
  // second sarcasm score (corr with is_sarcastic_score = 0.9997) that
  // *anti*-correlates with real hate. See backend/app/schemas.py.
  //
  // Thresholds are deliberately independent of argmax. A line at
  // hate=0.45 / non=0.47 argmaxes to non_hateful but is far more suspicious
  // than the label alone suggests, and treating the two signals separately
  // is also what lets a single line be both blurred (hate) and tinted
  // (sarcasm) -- the 3-class softmax forces those to compete, but the
  // underlying phenomena genuinely co-occur.
  const HATE_THRESHOLD = 0.5;
  // Sarcasm is held to a much stricter bar than hate.
  //
  // It is the harder problem in principle -- sarcasm is carried by context,
  // tone and shared knowledge rather than by lexical cues, so a short line of
  // page text often doesn't contain the evidence needed to call it at all.
  //
  // And in these checkpoints specifically, `sarcastic` behaves as a catch-all
  // for anything unfamiliar: source leakage means the model learned to
  // separate its *datasets* rather than the two phenomena, so out-of-
  // distribution text drifts into the sarcasm class regardless of content.
  // Benign Bangla lands there directly -- "আজকে আবহাওয়া খুব সুন্দর" ("the
  // weather is lovely today") scores sarcastic at 0.700.
  //
  // Measured against ood_eval_bn.csv plus 20 benign Bangla lines:
  //
  //   bar    benign wrongly tinted    threats caught ONLY by the sarcasm rule
  //   0.50        2/41 (4.9%)                    8/49 (16.3%)
  //   0.70        1/41 (2.4%)                    7/49 (14.3%)
  //   0.80        1/41 (2.4%)                    5/49 (10.2%)
  //   0.85        0/41 (0.0%)                    5/49 (10.2%)   <- here
  //   0.90        0/41 (0.0%)                    4/49 ( 8.2%)
  //
  // 0.85 is the knee of this curve, and the only step on it that costs
  // nothing: it clears the last benign false positive ("ট্রেন সকাল আটটায়
  // ছাড়বে", 0.840) while no genuine threat scores between 0.80 and 0.85, so
  // recall is identical to 0.80. Going further to 0.90 does start costing
  // real detections again.
  //
  // The steps below it were not free -- 0.70 -> 0.80 gave up two genuine
  // vulgar threats scoring in that band -- which is why yellow is tuned for
  // precision here: a tint the user cannot trust is worse than no tint, and
  // anything this bar misses is recoverable through the correction mechanism.
  //
  // Hate keeps the much lower 0.5 bar throughout: a missed slur costs more
  // than a missed piece of irony.
  const SARC_THRESHOLD = 0.85;

  // Blur radius in px at the threshold and at probability 1.0. Hate gets a
  // real, unreadable blur; sarcasm gets a much lighter one -- enough to
  // signal "this was flagged" without suppressing text that is often merely
  // ironic rather than harmful.
  const HATE_BLUR_MIN_PX = 4;
  const HATE_BLUR_MAX_PX = 14;
  const SARC_BLUR_MIN_PX = 0.5;
  const SARC_BLUR_MAX_PX = 3;

  // Tint opacity range, applied the same way: stronger conviction reads as a
  // more saturated highlight.
  const TINT_MIN_ALPHA = 0.15;
  const TINT_MAX_ALPHA = 0.55;

  /**
   * Rescale a probability from [threshold, 1] onto [0, 1] so intensity is
   * proportional to conviction *above the bar*, not to the raw probability.
   * Without this, a 0.90 and a 0.51 detection would look nearly identical
   * (0.90 vs 0.51 of max blur) instead of maximally vs minimally intense --
   * the whole point of the visualization is discriminating between them.
   */
  function intensityAbove(prob, threshold) {
    if (prob <= threshold) return 0;
    return (prob - threshold) / (1 - threshold);
  }

  function lerp(min, max, t) {
    return min + (max - min) * t;
  }

  // ---- text-node extraction -----------------------------------------
  // Tags whose text is never real content: scripts/styles never render,
  // form controls' visible "text" is UI chrome (a button's label, an
  // input's placeholder), and editable regions are the user's own
  // in-progress input, not page content to classify.
  const SKIP_TAGS = new Set([
    "SCRIPT", "STYLE", "NOSCRIPT", "INPUT", "TEXTAREA", "SELECT", "BUTTON",
    "IFRAME", "svg", "SVG", "OPTION",
  ]);

  // Inline elements never introduce a line break on their own (that's the
  // literal CSS/HTML definition of "inline") -- text split across sibling
  // inline elements is one logical line, not independent units. Consecutive
  // text nodes/inline-element text under the same nearest block-level
  // ancestor get merged into one line below.
  const INLINE_TAGS = new Set([
    "SPAN", "A", "B", "I", "EM", "STRONG", "U", "SMALL", "MARK", "SUB", "SUP",
    "LABEL", "CODE", "TIME", "ABBR", "CITE", "Q", "BDI", "BDO",
  ]);

  // A genuinely hateful single word (a slur) can be as short as 3-4
  // characters -- confirmed on a real page where entries like "৪। হারামি"
  // (9 chars) matter. This floor only guards against empty/near-empty
  // fragments; it's applied once, to the merged LINE, not per text node
  // (a line built from several very short inline fragments should still
  // count once concatenated).
  const MIN_LINE_LENGTH = 3;

  function isSkippableAncestor(element, root) {
    let node = element;
    while (node && node !== root.parentElement) {
      if (SKIP_TAGS.has(node.tagName)) return true;
      // isContentEditable also covers inherited editability (a child of a
      // contenteditable container without its own attribute), but check the
      // attribute directly too as a defensive fallback for engines/contexts
      // where the isContentEditable getter isn't reliably populated.
      if (node.isContentEditable || node.getAttribute("contenteditable") === "true") {
        return true;
      }
      node = node.parentElement;
    }
    return false;
  }

  // Walks every text node under `root` (root itself included) and returns
  // them grouped into "lines": arrays of consecutive text nodes that
  // belong together. A new line starts whenever EITHER (a) a text node's
  // nearest block-level ancestor differs from the previous node's -- this
  // is what correctly merges a tweet's <span>-fragmented sentence back
  // into one string, since sibling inline elements share the same block
  // parent -- OR (b) a <br> element is encountered between two text nodes
  // that otherwise share the same block ancestor. (b) matters because
  // older/hand-formatted markup (a Word-paste blog post, for example) often
  // renders many list entries as sibling inline elements *separated by
  // <br> tags* under one shared wrapper, rather than using separate block
  // elements per entry -- confirmed on a real page. Walking SHOW_TEXT +
  // SHOW_ELEMENT (not SHOW_TEXT alone) is what lets this function actually
  // see those <br> elements in document order alongside the text nodes.
  function extractLines(root) {
    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT,
      {
        acceptNode(node) {
          if (node.nodeType === 1 /* ELEMENT_NODE */) {
            if (node.tagName === "BR") return NodeFilter.FILTER_ACCEPT;
            if (SKIP_TAGS.has(node.tagName) || node.isContentEditable) {
              return NodeFilter.FILTER_REJECT;
            }
            return NodeFilter.FILTER_SKIP; // don't stop on it, but do visit its children
          }
          // text node
          if (!node.textContent || node.textContent.trim() === "") {
            return NodeFilter.FILTER_SKIP;
          }
          if (isSkippableAncestor(node.parentElement, root)) {
            return NodeFilter.FILTER_REJECT;
          }
          return NodeFilter.FILTER_ACCEPT;
        },
      }
    );

    const lines = [];
    let currentLine = null;
    let currentBlockAncestor = null;

    let node = walker.nextNode();
    while (node) {
      if (node.nodeType === 1 /* ELEMENT_NODE, i.e. a <br> */) {
        currentLine = null;
        currentBlockAncestor = null;
        node = walker.nextNode();
        continue;
      }

      const blockAncestor = nearestBlockAncestor(node.parentElement, root);

      if (!currentLine || blockAncestor !== currentBlockAncestor) {
        currentLine = { blockAncestor, nodes: [], text: "" };
        lines.push(currentLine);
        currentBlockAncestor = blockAncestor;
      }

      currentLine.nodes.push(node);
      currentLine.text += node.textContent;

      node = walker.nextNode();
    }

    return lines;
  }

  // Walks up from `element` until it finds a non-inline (block-level)
  // element, or hits `root`. Used as the grouping key for "which line does
  // this text node belong to" -- two text nodes with the same nearest
  // block ancestor are on the same line.
  function nearestBlockAncestor(element, root) {
    let node = element;
    while (node && node !== root.parentElement) {
      if (!INLINE_TAGS.has(node.tagName)) return node;
      node = node.parentElement;
    }
    return element;
  }

  // The DOM elements a line's blur should be applied to: the nearest
  // Element for each contributing text node (a Text node itself can't take
  // a CSS class). Deduplicated, since multiple text nodes in a line
  // commonly share the same immediate parent span.
  function elementsForLine(line) {
    const elements = new Set();
    for (const node of line.nodes) {
      if (node.parentElement) elements.add(node.parentElement);
    }
    return [...elements];
  }

  function detectLanguage(text) {
    // Cheap heuristic: presence of Bangla Unicode block chars (U+0980-U+09FF).
    return /[ঀ-৿]/.test(text) ? "bn" : "en";
  }

  // ---- node tracking -------------------------------------------------
  // WeakSet so entries are garbage-collected once a node is removed from
  // the DOM (e.g. scrolled away and unmounted by a virtualized list).
  //
  // Keyed by TEXT NODE, not by parent Element. Confirmed on a real page
  // (hand-formatted list: one wrapper <span> holding many entries separated
  // only by <br>, no per-entry sub-span) that several DISTINCT lines can
  // share the very same contributing element -- keying "scored" by element
  // there means scoring line 1 marks that shared span as scored, so lines
  // 2-5 read back as "already scored" and are silently skipped even though
  // their own text was never sent. A text node, by construction of
  // extractLines, belongs to exactly one line, so it's the correct atomic
  // unit here; elements remain the right unit only for applying a blur
  // (a Text node can't take a CSS class).
  //
  // "observed" = currently attached to viewportObserver (don't attach twice).
  // "scored"   = has actually been sent for scoring at least once. These are
  // deliberately separate: while disabled, nodes can become "observed"
  // (IntersectionObserver fires, but queueForScoring no-ops) without ever
  // becoming "scored" -- re-enabling needs to catch those up directly,
  // since IntersectionObserver won't fire again for elements whose
  // intersection state hasn't changed since it last fired.
  const observedNodes = new WeakSet();
  const scoredNodes = new WeakSet();

  // element -> the line texts that element contributed to. Needed by the
  // right-click "mark this" flow to answer "what line did the user just click
  // on?" without re-deriving line grouping from a raw event target -- and an
  // element can back several lines (the <br>-separated wrapper case), hence a
  // list. WeakMap so detached nodes are collectable.
  const lineTextsByElement = new WeakMap();

  // element -> { text, kind, percent } for currently-flagged elements, so a
  // hold-to-review can answer "what was this flagged as, and over what text?"
  const flagInfoByElement = new WeakMap();

  // Both live up here with the rest of the module state rather than next to
  // the code that reads them, so all per-page tracking is declared in one
  // place alongside observedNodes/scoredNodes.

  // Currently-blurred lines, so a live toggle-off can reveal all of them at
  // once. Each entry is the same { elements, textNodes } shape queued items
  // carry -- both are needed on reveal: elements to unblur, textNodes to
  // un-mark from scoredNodes (see the comment on scoredNodes above for why
  // an element can't be used for that when it's shared across lines). A
  // regular array (not WeakSet) because we actively iterate it; cleared on
  // reveal, so it can't grow unboundedly.
  let blurredLines = [];

  // ---- enabled/disabled state --------------------------------------------
  // Defaults to true so the extension works immediately on first install,
  // before the user has ever opened the popup.
  let isEnabled = true;

  chrome.storage.local.get({ enabled: true }, ({ enabled }) => {
    isEnabled = enabled;
    if (isEnabled) {
      rescanWholePage(); // page may already have content loaded before we got here
    } else {
      revealAll();
    }
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local" || !("enabled" in changes)) return;
    isEnabled = changes.enabled.newValue;
    if (isEnabled) {
      // IntersectionObserver only fires on a *change* of intersection state,
      // so posts already sitting in the viewport when we were disabled
      // never re-fire on their own -- force a fresh look at the whole page.
      rescanWholePage();
    } else {
      revealAll();
    }
  });

  function rescanWholePage() {
    // Catch-up path for re-enabling: score everything already on the page
    // that hasn't been scored yet, immediately -- don't wait on
    // IntersectionObserver, since it only fires on a *change* of
    // intersection state and won't re-fire for elements that were already
    // sitting in the viewport before we were disabled.
    scanRoot(findFeedContainer(), /* forceRescore */ true);
  }

  function revealAll() {
    for (const { elements, textNodes } of blurredLines) {
      for (const element of elements) clearFlag(element);
      // This line's visual state (now unblurred) no longer reflects a real
      // scoring decision, so its own text nodes must be eligible for
      // scoring again on re-enable -- otherwise rescanWholePage() would see
      // them in scoredNodes and wrongly skip them forever, leaving them
      // unblurred even though the model would still flag them as hateful.
      // Unmarking only this line's own textNodes (not blanket-clearing the
      // shared element) matters when several lines share one contributing
      // element but only some of them were ever flagged hateful.
      for (const node of textNodes) scoredNodes.delete(node);
    }
    blurredLines = [];
  }

  // ---- batching --------------------------------------------------------
  // { id, text, language, elements, textNodes } per queued line --
  // `elements` is the set of DOM elements this line's text actually came
  // from (used to apply a blur; a direct reference so a hateful result
  // never needs to search the page for matching text). `textNodes` is the
  // line's own text nodes, used for scoredNodes bookkeeping -- see the
  // comment on `scoredNodes` above for why elements are the wrong
  // granularity for that.
  let pendingQueue = [];
  let batchTimer = null;
  let inFlightBatches = 0;
  const MAX_CONCURRENT_BATCHES = 3;

  function queueForScoring(elements, textNodes, text) {
    if (!isEnabled) return;

    for (const node of textNodes) scoredNodes.add(node);
    for (const element of elements) {
      const known = lineTextsByElement.get(element);
      if (known) {
        if (!known.includes(text)) known.push(text);
      } else {
        lineTextsByElement.set(element, [text]);
      }
    }

    const id = `n${Date.now()}_${Math.random().toString(36).slice(2)}`;
    pendingQueue.push({ id, text, language: detectLanguage(text), elements, textNodes });

    if (batchTimer) clearTimeout(batchTimer);
    batchTimer = setTimeout(flushQueue, BATCH_DEBOUNCE_MS);
  }

  // True once the extension context is confirmed gone for this page (the
  // background worker's connection was severed -- typically because the
  // extension was reloaded/updated in chrome://extensions while this tab
  // stayed open, so this already-injected content script keeps running but
  // chrome.runtime is no longer usable). This is NOT the normal MV3
  // "service worker went idle" case -- Chrome auto-wakes an idle worker and
  // queues the message for it; that case surfaces as a normal fetch/network
  // failure from background.js, handled below by requeueing. This flag is
  // specifically for the unrecoverable case, confirmed by chrome.runtime
  // itself being missing (not just a failed call) -- only a real page
  // reload fixes it, so once detected, stop retrying and disconnect the
  // observers instead of retrying into the void forever.
  let extensionContextInvalidated = false;

  function requeue(batch) {
    // Put failed items back so a transient failure (background worker
    // waking from idle, one dropped message) doesn't silently and
    // permanently skip those posts -- they'll be retried on the next
    // flush instead of just vanishing from scoredNodes' perspective.
    pendingQueue.unshift(...batch);
  }

  // Consecutive-failure backoff: if the backend is actually down (not just
  // a one-off dropped message), retrying every BATCH_DEBOUNCE_MS forever
  // would hammer it indefinitely. Back off exponentially per consecutive
  // failure, capped at 30s, and reset back to the normal fast cadence the
  // moment a request succeeds again.
  let consecutiveFailures = 0;
  const MAX_RETRY_DELAY_MS = 30000;

  function nextRetryDelayMs() {
    return Math.min(BATCH_DEBOUNCE_MS * 2 ** consecutiveFailures, MAX_RETRY_DELAY_MS);
  }

  async function flushQueue() {
    // A fired setTimeout doesn't clear the variable holding its own id --
    // batchTimer must be nulled out here, the moment this callback actually
    // runs, or every later "is a flush already scheduled?" check
    // (`!batchTimer`, at the bottom of this function) permanently reads
    // false after the first flush ever fires, since batchTimer still holds
    // that first, already-fired timer's stale id. Confirmed as a real bug:
    // on a page with many candidate lines, the first flush handles up to
    // BATCH_MAX_ITEMS of them, but leftover items already in pendingQueue
    // afterward were silently never scheduled for a follow-up flush at all.
    batchTimer = null;

    if (extensionContextInvalidated) return;
    if (pendingQueue.length === 0) return;
    if (inFlightBatches >= MAX_CONCURRENT_BATCHES) {
      // backpressure: try again shortly instead of piling up requests
      batchTimer = setTimeout(flushQueue, BATCH_DEBOUNCE_MS);
      return;
    }

    if (!chrome.runtime || !chrome.runtime.sendMessage) {
      // Extension context invalidated (see comment above) -- chrome.runtime
      // itself is gone, not just a failed call. Retrying won't help until
      // the page is reloaded, so stop the whole pipeline cleanly instead of
      // throwing an uncaught TypeError on every future attempt.
      extensionContextInvalidated = true;
      console.warn(
        "[hate-sarcasm-extension] extension context invalidated (likely reloaded/updated) " +
          "-- reload this page to resume scanning."
      );
      mutationObserverRef && mutationObserverRef.disconnect();
      viewportObserver.disconnect();
      return;
    }

    const batch = pendingQueue.splice(0, BATCH_MAX_ITEMS);
    const itemById = new Map(batch.map((item) => [item.id, item]));

    inFlightBatches++;
    try {
      // The actual fetch happens in the background service worker, not
      // here -- a content script's network requests execute as the
      // visited page's own origin, which triggers Chrome's Local Network
      // Access permission prompt (per-site) when reaching into
      // localhost. The background worker's origin is chrome-extension://,
      // already covered by host_permissions, no prompt. See background.js.
      const response = await chrome.runtime.sendMessage({
        type: "PREDICT_BATCH",
        items: batch.map(({ id, text, language }) => ({ id, text, language })),
      });

      if (!response || !response.ok) {
        // Stringify explicitly -- some log capture surfaces (e.g. browser
        // automation tooling reading the console) render a bare object
        // argument as the unhelpful "[object Object]" instead of expanding
        // it, so the failure reason must be baked into the message text
        // itself, not passed as a separate console.warn argument.
        const reason = !response
          ? "no response from background worker"
          : response.error
          ? `network error: ${response.error}`
          : `HTTP ${response.status}`;
        console.warn(`[hate-sarcasm-extension] batch request failed: ${reason}`);
        for (const item of batch) {
          for (const node of item.textNodes) scoredNodes.delete(node);
        }
        requeue(batch);
        consecutiveFailures++;
        return;
      }

      consecutiveFailures = 0;
      if (!isEnabled) return; // disabled while this batch was in flight

      for (const result of response.data.results) {
        const item = itemById.get(result.id);
        if (item) applyFlag(item, result);
      }
    } catch (err) {
      const message = err && err.message ? err.message : String(err);
      console.warn(`[hate-sarcasm-extension] batch request error: ${message}`);
      for (const item of batch) {
        for (const node of item.textNodes) scoredNodes.delete(node);
      }
      requeue(batch);
      consecutiveFailures++;
    } finally {
      inFlightBatches--;
      if (pendingQueue.length > 0 && !batchTimer && !extensionContextInvalidated) {
        batchTimer = setTimeout(flushQueue, nextRetryDelayMs());
      }
    }
  }

  // ---- flagging (no reveal-on-click; only the popup toggle reveals) ------
  /**
   * Apply intensity-scaled treatment to one line's elements.
   *
   * ONE signal wins outright. Hate and sarcasm are still *scored*
   * independently (the 3-class softmax makes them compete for probability
   * mass even though the phenomena co-occur, so thresholding them separately
   * catches lines the argmax label would miss), but whichever clears its
   * threshold by more then owns the entire treatment: colour, blur radius,
   * tint strength and tooltip all come from that one signal.
   *
   * The earlier version mixed them -- hue from the dominant signal, blur from
   * `Math.max(hateBlur, sarcBlur)` across both. Because the hate blur range
   * (4-14px) sits entirely above the sarcasm range (0.5-3px), any trace of
   * hate above threshold dragged the blur into hate territory while the tint
   * stayed yellow: a line reading as "sarcastic" but blurred as hard as hate.
   * Reading a single signal end-to-end is what makes the colour actually mean
   * something.
   *
   * Blur radius and tint alpha are pushed as CSS custom properties rather
   * than baked into class names, because bucketing into discrete classes
   * ("hs-blur-1".."hs-blur-5") would quantize exactly the gradient this
   * feature exists to show.
   */
  function applyFlag(item, result) {
    const hateIntensity = intensityAbove(result.hate_prob, HATE_THRESHOLD);
    const sarcIntensity = intensityAbove(result.sarc_prob, SARC_THRESHOLD);

    if (hateIntensity === 0 && sarcIntensity === 0) {
      // Not just "nothing to do": a line can arrive here having been flagged
      // before (a user correction re-scores it as non-hateful), and it must
      // visibly clear rather than keep a stale blur.
      for (const element of item.elements) clearFlag(element);
      return;
    }

    const isHate = hateIntensity >= sarcIntensity;
    const intensity = isHate ? hateIntensity : sarcIntensity;
    const blurPx = isHate
      ? lerp(HATE_BLUR_MIN_PX, HATE_BLUR_MAX_PX, intensity)
      : lerp(SARC_BLUR_MIN_PX, SARC_BLUR_MAX_PX, intensity);
    const alpha = lerp(TINT_MIN_ALPHA, TINT_MAX_ALPHA, intensity);

    const percent = Math.round((isHate ? result.hate_prob : result.sarc_prob) * 100);
    const kind = isHate ? "hateful" : "sarcastic";
    // A verdict that came from a user correction says so on hover, so an
    // override is never mistaken for the model having changed its mind.
    const taught =
      result.source === "feedback_exact"
        ? " (you marked this)"
        : result.source === "feedback_similar"
        ? ` (matches something you marked, ${Math.round(result.match_similarity * 100)}%)`
        : "";

    for (const element of item.elements) {
      element.style.setProperty("--hs-blur", `${blurPx.toFixed(2)}px`);
      element.style.setProperty("--hs-tint-alpha", alpha.toFixed(3));
      element.classList.add("hs-flagged");
      element.classList.toggle("hs-hate", isHate);
      element.classList.toggle("hs-sarc", !isHate);
      // Readable on hover even while blurred -- the intensity is the point,
      // so it should be inspectable without turning the extension off.
      element.title = `${kind}: ${percent}%${taught} — hold to review`;
      // Remembered so a hold can answer "what was this flagged as,
      // and what text did the flag apply to?" without re-deriving either.
      flagInfoByElement.set(element, { text: item.text, kind, percent });
    }

    blurredLines.push({ elements: item.elements, textNodes: item.textNodes });
  }

  function clearFlag(element) {
    element.classList.remove("hs-flagged", "hs-hate", "hs-sarc", "hs-revealed");
    element.style.removeProperty("--hs-blur");
    element.style.removeProperty("--hs-tint-alpha");
    element.removeAttribute("title");
  }

  function injectStyles() {
    if (document.getElementById("hs-blur-style")) return;
    const style = document.createElement("style");
    style.id = "hs-blur-style";
    // --hs-blur / --hs-tint-alpha are set per-element in applyFlag; the
    // fallbacks here only matter if a class somehow lands without them.
    style.textContent = `
      .hs-flagged {
        filter: blur(var(--hs-blur, 8px));
        background-color: rgba(var(--hs-tint-rgb), var(--hs-tint-alpha, 0.3));
        border-radius: 3px;
        user-select: none;
        transition: filter 0.15s ease, background-color 0.15s ease;
      }
      /* Red = hateful, yellow = sarcastic. Saturation of the wash tracks
         the model's confidence via --hs-tint-alpha. */
      .hs-flagged.hs-hate { --hs-tint-rgb: 220, 38, 38; }
      .hs-flagged.hs-sarc { --hs-tint-rgb: 234, 179, 8; }

      /* Under review (held): fully legible and selectable, so the
         user can actually judge it and mark it. The dashed outline keeps the
         line identifiable as the one being reviewed now that the blur and
         wash are gone -- without it, a revealed line is indistinguishable
         from ordinary page text. !important because these three properties
         are exactly the ones the .hs-flagged rule above sets, and specificity
         alone would be fragile against page styles. */
      .hs-flagged.hs-revealed {
        filter: none !important;
        background-color: transparent !important;
        user-select: text !important;
        outline: 2px dashed rgba(var(--hs-tint-rgb), 0.9);
        outline-offset: 2px;
      }
    `;
    document.head.appendChild(style);
  }

  // ---- press and hold to review a flagged line --------------------------
  //
  // This is a deliberate narrowing of the "no click-to-reveal" rule stated at
  // the top of this file. That rule exists so hateful text can't be casually
  // un-hidden -- but it made correcting a mistake impossible: to say "this was
  // wrongly flagged" you first have to be able to READ it, and blurred text
  // also carries `user-select: none`, so it can't even be selected.
  //
  // The reveal is therefore scoped as tightly as possible: one line at a time,
  // temporary, and it re-blurs the moment the review ends unless the user
  // explicitly says the line is clean. Nothing reveals in bulk, and nothing
  // stays revealed by accident.

  // The line under review: { elements, text, info } or null.
  let reviewing = null;

  const MARK_ACTIONS = [
    { label: "hateful", text: "Hateful", accent: "#dc2626" },
    { label: "sarcastic", text: "Sarcastic", accent: "#eab308" },
    { label: "non_hateful", text: "Neither", accent: "#6b7280" },
  ];

  // The toolbar lives in a shadow root so the host page's CSS cannot restyle
  // or hide it -- on a real feed page, broad selectors ("div button { ... }")
  // routinely break injected UI that shares the page's style scope.
  let toolbarHost = null;
  let toolbarShadow = null;

  function buildToolbar() {
    if (toolbarHost) return;

    toolbarHost = document.createElement("div");
    toolbarHost.style.cssText =
      "position:absolute;z-index:2147483647;display:none;";
    toolbarShadow = toolbarHost.attachShadow({ mode: "closed" });

    const style = document.createElement("style");
    style.textContent = `
      .bar {
        display: flex; align-items: center; gap: 6px;
        padding: 6px 8px;
        background: #1a1d23; color: #e6e6e6;
        border: 1px solid #3a3d44; border-radius: 8px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        font: 500 12px/1.2 system-ui, sans-serif;
        white-space: nowrap;
      }
      .what { color: #aeb2b8; margin-right: 2px; }
      button {
        font: 600 12px/1 system-ui, sans-serif;
        color: #fff; background: #33363d;
        border: 1px solid transparent; border-radius: 5px;
        padding: 5px 9px; cursor: pointer;
      }
      button:hover { border-color: currentColor; }
    `;

    const bar = document.createElement("div");
    bar.className = "bar";

    const what = document.createElement("span");
    what.className = "what";
    bar.appendChild(what);

    for (const action of MARK_ACTIONS) {
      const button = document.createElement("button");
      button.textContent = action.text;
      button.style.color = action.accent;
      // mousedown, not click: the document-level mousedown handler below ends
      // the review on any outside press, and click would fire after it.
      button.addEventListener("mousedown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        submitMark(action.label);
      });
      bar.appendChild(button);
    }

    toolbarShadow.append(style, bar);
    document.body.appendChild(toolbarHost);
    toolbarShadow.__what = what;
  }

  function positionToolbar(anchor) {
    const rect = anchor.getBoundingClientRect();
    toolbarHost.style.top = `${rect.bottom + window.scrollY + 6}px`;
    toolbarHost.style.left = `${rect.left + window.scrollX}px`;
  }

  function beginReview(element) {
    const info = flagInfoByElement.get(element);
    if (!info) return;

    endReview(/* reblur */ true); // only ever one line under review

    // Reveal every element of the line, not just the one clicked -- a line
    // split across sibling inline elements would otherwise reveal in pieces.
    const elements = [];
    for (const entry of blurredLines) {
      if (entry.elements.includes(element)) elements.push(...entry.elements);
    }
    if (elements.length === 0) elements.push(element);

    for (const el of elements) el.classList.add("hs-revealed");
    reviewing = { elements, text: info.text, info };

    buildToolbar();
    toolbarShadow.__what.textContent = `flagged ${info.kind} ${info.percent}% —`;
    toolbarHost.style.display = "block";
    positionToolbar(element);

    window.addEventListener("scroll", repositionToolbar, true);
    window.addEventListener("resize", repositionToolbar);
  }

  function repositionToolbar() {
    if (reviewing) positionToolbar(reviewing.elements[0]);
  }

  function endReview(reblur) {
    if (!reviewing) return;
    if (reblur) {
      for (const el of reviewing.elements) el.classList.remove("hs-revealed");
    }
    reviewing = null;
    if (toolbarHost) toolbarHost.style.display = "none";
    window.removeEventListener("scroll", repositionToolbar, true);
    window.removeEventListener("resize", repositionToolbar);
  }

  /**
   * What the mark applies to: the user's selection if they made one inside
   * the line under review, otherwise the whole line. Selecting a fragment is
   * the more precise correction -- on a line where only a few words are the
   * slur, teaching the whole line teaches a lot of irrelevant context along
   * with it -- so it wins when present.
   */
  function markTarget() {
    const selected = (window.getSelection() || "").toString().trim();
    if (!selected) return reviewing.text;
    // Guard against a stale selection left somewhere else on the page.
    return squash(reviewing.text).includes(squash(selected)) ? selected : reviewing.text;
  }

  function submitMark(label) {
    if (!reviewing) return;
    const target = markTarget();
    const elements = reviewing.elements;

    // "Neither" leaves the text readable: re-blurring the thing the user just
    // told us is fine would undo the correction in front of them. The other
    // two re-blur, in the newly chosen colour, via the re-score below.
    endReview(/* reblur */ label !== "non_hateful");
    if (label === "non_hateful") {
      for (const el of elements) clearFlag(el);
    }

    sendFeedback(target, label);
  }

  function isInsideReview(target) {
    if (!reviewing) return false;
    let node = target;
    while (node && node !== document.body) {
      if (node.nodeType === 1 && reviewing.elements.includes(node)) return true;
      node = node.parentElement;
    }
    return false;
  }

  function closestFlagged(target) {
    let node = target;
    while (node && node !== document.body) {
      if (node.nodeType === 1 && node.classList.contains("hs-flagged")) return node;
      node = node.parentElement;
    }
    return null;
  }

  // ---- the gesture: press and hold -------------------------------------
  //
  // This was originally double-click, which is unusable on flagged text that
  // happens to be inside a link -- and on a feed page a great deal of it is.
  // `dblclick` fires only AFTER two full click events, so by the time the
  // handler runs the browser has already followed the href; calling
  // preventDefault() on the dblclick is far too late to stop it.
  //
  // A hold is intercepted at the source instead: the review opens on a timer
  // during mousedown, and the click that eventually follows is swallowed in
  // the capture phase before the page's own handler or the link's default
  // navigation can act on it.
  const LONG_PRESS_MS = 450;
  // Slop before a hold is treated as a drag instead. Small enough that a
  // deliberate hold survives a shaky hand, large enough that starting a text
  // selection reliably cancels the review.
  const LONG_PRESS_SLOP_PX = 6;

  let pressTimer = null;
  let pressOrigin = null;
  let suppressNextClick = false;

  function cancelPress() {
    if (pressTimer) clearTimeout(pressTimer);
    pressTimer = null;
    pressOrigin = null;
  }

  document.addEventListener(
    "mousedown",
    (event) => {
      // Any press starts a fresh interaction, so a suppression flag left over
      // from a hold whose click never arrived (pointer released off-target,
      // say) can never leak into a later, unrelated click.
      suppressNextClick = false;

      // Presses outside the toolbar and outside the reviewed line abandon the
      // review and re-blur, so a revealed line can't be left exposed by
      // walking away. A press INSIDE it is not "outside" -- that is the user
      // dragging to select the text they were given the chance to read, and
      // re-blurring mid-drag would make selecting it impossible.
      if (reviewing && event.target !== toolbarHost && !isInsideReview(event.target)) {
        endReview(true);
      }

      if (!isEnabled || event.button !== 0) return;
      const flagged = closestFlagged(event.target);
      // Already revealed: no hold, so press-and-drag selects text normally.
      if (!flagged || flagged.classList.contains("hs-revealed")) return;

      pressOrigin = { x: event.clientX, y: event.clientY };
      pressTimer = setTimeout(() => {
        pressTimer = null;
        suppressNextClick = true;
        beginReview(flagged);
      }, LONG_PRESS_MS);
    },
    true
  );

  document.addEventListener(
    "mousemove",
    (event) => {
      if (!pressOrigin) return;
      if (
        Math.abs(event.clientX - pressOrigin.x) > LONG_PRESS_SLOP_PX ||
        Math.abs(event.clientY - pressOrigin.y) > LONG_PRESS_SLOP_PX
      ) {
        cancelPress();
      }
    },
    true
  );

  document.addEventListener("mouseup", cancelPress, true);

  // Capture phase, so the link's navigation and the page's own click handlers
  // are both cut off. This is the whole reason the gesture is a hold.
  document.addEventListener(
    "click",
    (event) => {
      if (!suppressNextClick) return;
      suppressNextClick = false;
      event.preventDefault();
      event.stopPropagation();
    },
    true
  );
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") endReview(true);
  });

  // ---- teaching the model (right-click -> "mark this ...") ---------------
  // A correction is stored server-side and takes effect on the very next
  // /predict call, so nothing here waits for a retrain or a page reload: the
  // affected lines are simply re-scored in place, ~200ms later.

  // The last right-clicked target, captured in the capture phase so it is
  // recorded even if the page stops the event from bubbling. Chrome fires
  // contextmenu before the menu item's click reaches the background worker,
  // so this is always current by the time a menu action arrives.
  let lastContextTarget = null;
  document.addEventListener(
    "contextmenu",
    (event) => {
      lastContextTarget = event.target;
    },
    true
  );

  /**
   * Comparison key for "is this the same text?": case-folded with all
   * whitespace removed. The backend does the authoritative matching; this is
   * only used to decide which on-page lines to re-score after a correction,
   * where being slightly generous is right -- a missed line just gets picked
   * up by the normal scan a moment later.
   */
  function squash(text) {
    return text.replace(/\s+/g, "").toLowerCase();
  }

  /**
   * What the user meant to mark. A selection is unambiguous, so it wins.
   * Otherwise fall back to the line under the cursor -- which is the whole
   * point of allowing a bare right-click: already-blurred text can't be
   * selected (`user-select: none`), so a false positive would otherwise be
   * impossible to correct.
   */
  function resolveContextText() {
    const selected = (window.getSelection() || "").toString().trim();
    if (selected) return selected;

    let node = lastContextTarget;
    while (node && node !== document.body) {
      if (node.nodeType === 1) {
        const texts = lineTextsByElement.get(node);
        if (texts && texts.length === 1) return texts[0];
        if (texts && texts.length > 1) {
          // Several lines share this element. Prefer the one that is actually
          // rendered inside the clicked node; fall back to the first.
          const own = squash(node.textContent || "");
          return texts.find((t) => own.includes(squash(t))) || texts[0];
        }
      }
      node = node.parentElement;
    }
    return "";
  }

  /**
   * Re-score every on-page line matching `text` right now, instead of waiting
   * for the user to reload. Clearing scoredNodes is what makes them eligible
   * again -- without it the WeakSet would report them as already handled and
   * the correction would appear to do nothing until a refresh.
   */
  function rescoreMatchingLines(text) {
    const needle = squash(text);
    if (!needle) return;

    const lines = extractLines(findFeedContainer());
    for (const line of lines) {
      const hay = squash(line.text.trim());
      // `includes` in both directions: the user may have selected only part
      // of a line, or marked a line that is itself contained in a longer one.
      if (!hay || (!hay.includes(needle) && !needle.includes(hay))) continue;

      // Clear only the matching lines. Blanket-clearing every flagged line on
      // the page would unblur unrelated hateful text that isn't going to be
      // re-scored here, leaving it exposed until the next full scan.
      for (const element of elementsForLine(line)) clearFlag(element);
      for (const node of line.nodes) scoredNodes.delete(node);
      queueLineIfDue(line);
    }

    // Anything already queued should go out now rather than after the normal
    // debounce -- this is a direct response to a user action.
    if (batchTimer) clearTimeout(batchTimer);
    batchTimer = setTimeout(flushQueue, 0);
  }

  /**
   * Store one correction and immediately re-score the lines it affects.
   * Shared by both entry points -- the right-click menu and the review
   * toolbar -- so the two can never drift apart in what they actually do.
   */
  function sendFeedback(text, label) {
    return chrome.runtime
      .sendMessage({
        type: "SEND_FEEDBACK",
        text,
        language: detectLanguage(text),
        label,
      })
      .then((response) => {
        if (response && response.ok) {
          rescoreMatchingLines(text);
        } else {
          const reason = response && response.error ? response.error : "backend unreachable";
          console.warn(`[hate-sarcasm-extension] could not save correction: ${reason}`);
        }
        return response || { ok: false, error: "no response" };
      })
      .catch((err) => {
        console.warn(`[hate-sarcasm-extension] could not save correction: ${err}`);
        return { ok: false, error: String(err) };
      });
  }

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type !== "CONTEXT_FEEDBACK") return false;

    const text = resolveContextText();
    if (!text) {
      sendResponse({ ok: false, error: "Could not tell which text you meant" });
      return false;
    }

    sendFeedback(text, message.label).then(sendResponse);
    return true; // async sendResponse
  });

  // ---- viewport-gated scanning -----------------------------------------
  // Each line is anchored to its first contributing element for
  // IntersectionObserver purposes -- when that element scrolls near the
  // viewport, the rest of the line (its siblings/same-parent nodes) is
  // virtually always visible too, since they're part of the same rendered
  // line of text.
  //
  // IMPORTANT: an anchor element is NOT unique per line. Confirmed on a real
  // page (hand-formatted list: one wrapper <span> holding many entries
  // separated only by <br>, with no per-entry sub-span) -- every entry's
  // first text node shares that SAME wrapper <span> as its parentElement.
  // A single `WeakMap<anchor, line>` silently drops all but the last line
  // registered per anchor (each `.set()` overwrites the previous line), and
  // `observedNodes.has(anchor)` then also short-circuits `observeLine` for
  // every one of those dropped lines on top of that -- so only ONE of
  // several distinct lines sharing an anchor was ever queued for scoring.
  // This was confirmed as the actual cause of specific known-hateful lines
  // (e.g. "৫৮। বাপের আগে বাল ফালায়") never getting blurred despite the
  // backend correctly classifying them. Fix: track a LIST of lines per
  // anchor, and observe the anchor element only once.
  const linesByAnchor = new WeakMap();

  function queueLineIfDue(line) {
    const text = line.text.trim();
    if (text.length < MIN_LINE_LENGTH) return;
    const elements = elementsForLine(line);
    if (elements.length === 0) return;
    queueForScoring(elements, line.nodes, text);
  }

  const viewportObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        viewportObserver.unobserve(entry.target);
        const lines = linesByAnchor.get(entry.target);
        if (lines) for (const line of lines) queueLineIfDue(line);
      }
    },
    { rootMargin: VIEWPORT_ROOT_MARGIN }
  );

  function observeLine(line, forceRescore) {
    const anchor = line.nodes[0].parentElement;
    if (!anchor) return;

    if (forceRescore) {
      // Catch-up path for re-enabling: IntersectionObserver only fires on a
      // *change* of intersection state, so lines already sitting in the
      // viewport before we were disabled never re-fire on their own --
      // score immediately instead of waiting for a visibility change.
      const alreadyScored = line.nodes.every((node) => scoredNodes.has(node));
      if (!alreadyScored) queueLineIfDue(line);
      return;
    }

    let lines = linesByAnchor.get(anchor);
    if (lines) {
      // Anchor already registered (shared by an earlier line) -- append
      // this line to it instead of dropping it; only skip re-observing the
      // already-observed DOM element itself.
      lines.push(line);
      return;
    }
    lines = [line];
    linesByAnchor.set(anchor, lines);

    if (observedNodes.has(anchor)) return;
    observedNodes.add(anchor);
    viewportObserver.observe(anchor);
  }

  function scanRoot(root, forceRescore) {
    const lines = extractLines(root);
    for (const line of lines) observeLine(line, forceRescore);
  }

  // ---- mutation watching (debounced) ------------------------------------
  let pendingMutationNodes = [];
  let mutationTimer = null;

  function handleMutations(mutationsList) {
    for (const mutation of mutationsList) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType === Node.ELEMENT_NODE) {
          pendingMutationNodes.push(node);
        }
      }
    }

    if (mutationTimer) clearTimeout(mutationTimer);
    mutationTimer = setTimeout(() => {
      const nodes = pendingMutationNodes;
      pendingMutationNodes = [];
      nodes.forEach((node) => scanRoot(node, false));
    }, MUTATION_DEBOUNCE_MS);
  }

  function findFeedContainer() {
    // Narrower than document.body where possible -- cuts mutation noise
    // dramatically on sites with a clear feed/timeline container.
    return document.getElementById("feed") || document.body;
  }

  let mutationObserverRef = null;

  function start() {
    injectStyles();

    const container = findFeedContainer();
    scanRoot(container, false); // initial pass over whatever's already on the page

    mutationObserverRef = new MutationObserver(handleMutations);
    mutationObserverRef.observe(container, { childList: true, subtree: true });

    window.addEventListener("beforeunload", () => {
      mutationObserverRef.disconnect();
      viewportObserver.disconnect();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
