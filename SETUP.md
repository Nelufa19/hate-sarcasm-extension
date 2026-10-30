# Setup — running this on a new machine

Two things must run: a **local Python server** (holds the ML models) and a **Chrome
extension** (reads the page). Both, or nothing works.

Takes about 10 minutes, most of it `pip install`.

---

## Part A — Before you zip it (sender only)

Do these three things, or the zip is 1.2 GB and leaks your browsing history.

```bash
cd hate-sarcasm-extension

rm -rf backend/.venv          # 492 MB, and it will NOT work on another machine
rm -rf backend/data           # your saved corrections = a trace of pages you visited
find . -name .DS_Store -delete
```

**Keep `backend/models/`** — the two `.pt` files are 771 MB and the project is useless
without them. That's the bulk of the ~780 MB zip. If your transfer can't take that, send the
zip without `backend/models/` and share those two files separately; they go back in at the
same paths.

Sanity check before sending:

```bash
du -sh .                      # expect ~780M
ls backend/models/*/model.pt  # expect exactly 2 files
```

---

## Part B — Backend (receiver)

**Requires Python 3.9 or newer.** Check with `python3 --version`.

```bash
cd hate-sarcasm-extension/backend

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # ~5 min, downloads PyTorch (~200 MB)

uvicorn app.main:app --reload --port 8000
```

Leave that terminal open. It is the server.

**It's working when you see this** (first start takes ~10 s while both models load):

```
Model ready: 'en'
Model ready: 'bn'
Startup complete: models=['en', 'bn'] feedback_entries=0
Application startup complete.
```

Verify in a second terminal:

```bash
curl http://localhost:8000/health
# {"status":"ok","models_loaded":["en","bn"],"feedback_entries":0}
```

---

## Part C — Chrome extension (receiver)

No build step. It loads as-is.

1. Open `chrome://extensions`
2. Turn on **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the **`hate-sarcasm-extension/extension`** folder
   — the folder containing `manifest.json`, **not** the project root
5. Pin it: click the puzzle-piece icon in the toolbar → pin **Hate & Sarcasm Blur**

---

## Part D — Check it actually works

Open the bundled test page — it needs no internet:

```
file:///<path-to-project>/hate-sarcasm-extension/extension/test-page/index.html
```

You should see hateful lines **blurred with a red wash**, and sarcastic lines **tinted
yellow but still readable**. Click the extension icon: it should say `online (en, bn)`.

Then try a real page, e.g. the Bangla list this was built against:
`https://allzinone.blogspot.com/2014/12/list-of-175-abusive-language-bangla.html`

---

## How to use it

| Action | How |
|---|---|
| Read a flagged line | **Press and hold** it (~0.5 s) — it un-blurs and a toolbar appears |
| Correct a wrong flag | Hold it, then pick **Hateful** / **Sarcastic** / **Neither** |
| Flag something it missed | **Right-click** it → *Hate & Sarcasm Blur* → mark it |
| Turn it off | Click the toolbar icon, flip the switch |

Corrections save to `backend/data/feedback.jsonl` and apply **immediately** — no restart, no
retraining. They persist across restarts.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Popup says **"unreachable"** | The server isn't running. Go back to Part B. |
| Nothing ever blurs | Reload the page after loading the extension — content scripts only inject on page load. |
| `address already in use` | Something else is on port 8000. Use `--port 8001`, then change both URLs at the top of `extension/background.js` and the one in `extension/popup.js`. |
| `ModuleNotFoundError` | The venv isn't active. Re-run `source .venv/bin/activate`. |
| `No such file: models/.../model.pt` | The `.pt` files didn't survive the transfer. See Part A. |
| Chrome asks about "other apps on this device" | Shouldn't happen — the network call is in the background worker for exactly this reason. If it does, the extension was loaded from the wrong folder (see Part C step 4). |
| Extension stops working after editing its code | Click the reload ↻ on its card in `chrome://extensions`, then reload the page. |

---

## Notes

- **Everything is local.** No text leaves the machine; the extension only ever talks to
  `localhost:8000`.
- **The extension runs on every site** (`<all_urls>`) and only reads visible text.
- **CORS is wide open** (`allow_origins=["*"]`) and the port is hardcoded — fine for local
  development, must change before any real deployment.
- **Windows/Linux are untested.** Everything here was verified on macOS. The code is
  platform-independent, but the setup commands above are the only part likely to differ.
- Run the test suite any time with `pytest -q` from `backend/` (60 tests, ~7 s).
