/**
 * MV3 background service worker. Owns the only network call this
 * extension makes (POST /predict/batch to the local FastAPI server).
 *
 * Why this can't live in content_script.js: a content script's fetch()
 * executes as the *visited page's own origin*, not the extension's origin
 * (Chromium's documented content-script network model). That means a
 * content script on https://some-public-site.com reaching into
 * http://localhost:8000 looks, to Chrome, like that public site making a
 * request into the user's local network -- which triggers Chrome's Local
 * Network Access (LNA) permission prompt ("wants to access other apps and
 * services on this device"), repeated per-site, since the "site" making
 * the request is different every time.
 *
 * A background service worker's origin is always chrome-extension://<id>,
 * which is covered by this extension's declared host_permissions and is
 * not subject to the same-origin/LNA check content scripts get. Moving
 * the fetch here removes the per-page prompt entirely.
 */

const BATCH_ENDPOINT = "http://localhost:8000/predict/batch";
const FEEDBACK_ENDPOINT = "http://localhost:8000/feedback";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "PREDICT_BATCH") {
    postJson(BATCH_ENDPOINT, { items: message.items }).then(sendResponse);
    return true; // keep the message channel open for the async sendResponse
  }

  if (message.type === "SEND_FEEDBACK") {
    postJson(FEEDBACK_ENDPOINT, {
      text: message.text,
      language: message.language,
      label: message.label,
    }).then(sendResponse);
    return true;
  }

  return false; // not for us
});

async function postJson(url, body) {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) return { ok: false, status: res.status };
    return { ok: true, data: await res.json() };
  } catch (err) {
    return { ok: false, error: String(err) };
  }
}

// ---- right-click corrections ----------------------------------------------
// Teaching the backend one example. The correction is live for the very next
// prediction -- no retrain, no reload -- so the content script just re-scores
// the affected lines in place once this returns.
//
// contexts: ["all"] rather than ["selection"] on purpose: blurred text carries
// `user-select: none`, so a false positive can never be selected. Without a
// no-selection path there would be no way to un-blur anything.
const MENU_ITEMS = [
  { id: "hs-mark-hateful", title: "Mark as hateful", label: "hateful" },
  { id: "hs-mark-sarcastic", title: "Mark as sarcastic", label: "sarcastic" },
  // The model is 3-class, so "not hateful" and "not sarcastic" are the same
  // stored label -- one item, named for both, rather than two that would
  // secretly do the same thing.
  { id: "hs-mark-clean", title: "Mark as neither (not hateful or sarcastic)", label: "non_hateful" },
];

chrome.runtime.onInstalled.addListener(() => {
  // removeAll first so a reload during development doesn't throw on
  // duplicate menu ids.
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "hs-root",
      title: "Hate & Sarcasm Blur",
      contexts: ["all"],
    });
    for (const item of MENU_ITEMS) {
      chrome.contextMenus.create({
        id: item.id,
        parentId: "hs-root",
        title: item.title,
        contexts: ["all"],
      });
    }
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  const item = MENU_ITEMS.find((m) => m.id === info.menuItemId);
  if (!item || !tab || tab.id === undefined) return;

  // The content script resolves WHICH text was meant (selection, or the line
  // under the cursor) -- that needs DOM and line-grouping knowledge this
  // worker doesn't have -- then calls back into SEND_FEEDBACK above.
  chrome.tabs
    .sendMessage(tab.id, { type: "CONTEXT_FEEDBACK", label: item.label })
    .catch((err) => {
      console.warn(`[hate-sarcasm-extension] could not reach page: ${err}`);
    });
});
