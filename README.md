# complaint-suggestion-classifier

PMC citizen-message classifier (complaint vs. suggestion), fine-tuned on MuRIL, with
IndicLID language auto-detection and a plain HTML/CSS/JS frontend.

## Structure

- `backend/` — FastAPI service (`api.py`), exposes `POST /predict`
- `frontend/` — static HTML/CSS/JS UI, calls the backend
- `deploy.sh` / `stop.sh` — set up a venv, install deps, and run both services
- `guardrails.py` — pre-model off-topic filter (routes non-civic text to human review)

## Model weights

Not included in this repo (exceed GitHub's file size limits) — handed over separately
as a zip. Expected layout, as siblings of `backend/` and `frontend/`:

```
muril-pmc-classifier/   (config.json, model.safetensors, tokenizer files)
indiclid-ftn/            (model_baseline_roman.bin)
indiclid-ftr/            (model_baseline_roman.bin)
```

## Run

```bash
./deploy.sh   # starts backend on :8000, frontend on :8080
./stop.sh
```
