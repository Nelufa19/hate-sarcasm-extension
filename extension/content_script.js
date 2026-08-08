/**
 * Content script: watches the page for new text, scans lines that scroll
 * into view, and blurs any classified as hateful.
 * No warning label, no click-to-reveal -- the only way to see blurred text
 * again is turning the extension off from the popup, which reveals
 * everything currently blurred on the page.
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

  // Labels that should be blurred. Only "hateful" -- sarcastic-but-not-hateful
  // text is left alone.
  const BLUR_LABELS = new Set(["hateful"]);

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
      for (const element of elements) element.classList.remove("hs-blurred");
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
        if (item && BLUR_LABELS.has(result.label)) {
          applyBlur(item);
        }
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

  // ---- blur (no reveal-on-click; only the popup toggle reveals content) --
  function applyBlur({ elements, textNodes }) {
    for (const element of elements) element.classList.add("hs-blurred");
    blurredLines.push({ elements, textNodes });
  }

  function injectStyles() {
    if (document.getElementById("hs-blur-style")) return;
    const style = document.createElement("style");
    style.id = "hs-blur-style";
    style.textContent = `
      .hs-blurred {
        filter: blur(8px);
        user-select: none;
        transition: filter 0.15s ease;
      }
    `;
    document.head.appendChild(style);
  }

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
