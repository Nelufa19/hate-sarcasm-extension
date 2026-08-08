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

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== "PREDICT_BATCH") return false; // not for us

  fetch(BATCH_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items: message.items }),
  })
    .then(async (res) => {
      if (!res.ok) {
        sendResponse({ ok: false, status: res.status });
        return;
      }
      const data = await res.json();
      sendResponse({ ok: true, data });
    })
    .catch((err) => {
      sendResponse({ ok: false, error: String(err) });
    });

  return true; // keep the message channel open for the async sendResponse above
});
