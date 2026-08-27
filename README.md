# complaint-suggestion-classifier

PMC citizen-message classifier (complaint vs. suggestion), fine-tuned on MuRIL, with a
plain HTML/CSS/JS frontend. The caller (frontend or an integrating backend) specifies
the language variant explicitly — there is no language auto-detection.

## Structure

- `backend/` — FastAPI service (`api.py`), exposes `POST /predict`
- `frontend/` — static HTML/CSS/JS UI, calls the backend
- `deploy.sh` / `stop.sh` — set up a venv, install deps, and run both services
- `guardrails.py` — pre-model off-topic filter (routes non-civic text to human review)

## Model weights

Not included in this repo (exceed GitHub's file size limits) — handed over separately
as a zip. Expected layout, as a sibling of `backend/` and `frontend/`:

```
muril-pmc-classifier/   (config.json, model.safetensors, tokenizer files)
```

## Run

```bash
./deploy.sh   # starts backend on :8000, frontend on :8080
./stop.sh
```

## Authentication

`POST /predict` requires an API key sent as the `X-API-Key` header. `deploy.sh`
generates one automatically on first run (via `openssl rand -hex 32`), prints it, and
saves it to `.api_key` (gitignored, reused across restarts). To use your own instead:

```bash
API_KEY=<your-key> ./deploy.sh
```

The frontend prompts for the key once and saves it in the browser's `localStorage` —
it's never hardcoded into the page source. An existing backend integrating with this
service should send the same key as `X-API-Key` on every request.
