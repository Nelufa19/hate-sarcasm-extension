---
title: Hate Sarcasm Detection API
emoji: 🛡️
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
---

# Hate & Sarcasm Detection API

FastAPI inference server for a dual-head DistilBERT model (English + Bangla)
that classifies text as `non_hateful`, `hateful`, or `sarcastic`. Backs the
"Hate & Sarcasm Blur" browser extension.

## Endpoints

- `GET /health` — liveness + which language models are loaded
- `GET /model-info` — architecture + backbone details
- `POST /predict` — single-text classification
- `POST /predict/batch` — batch classification (up to 50 items)

See `/docs` for interactive request/response schemas.
