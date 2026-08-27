"""
FastAPI backend for the fine-tuned MuRIL complaint/suggestion classifier.
Loads the model once at startup and exposes POST /predict for classification requests.
Requires the API_KEY env var; callers must send it as the X-API-Key header.

Run: API_KEY=<key> uvicorn api:app --host 0.0.0.0 --port 8000   (from this directory)
"""
import os
import sys
import time
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from guardrails import is_offtopic  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = str(PROJECT_ROOT / "muril-pmc-classifier")

MAX_LEN = 160
LABEL2ID = {"complaint": 0, "suggestion": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Calibrated on golden-1000 (302 real suggestion examples, up from golden-298's 18) via a
# per-language threshold sweep. English/Marathi-Dev independently converged on the same
# optimal threshold (0.85); Hindi-Dev's own optimum (0.80) and Hinglish's (n=5, too few to
# trust) both default to that shared value. Marathi-Roman diverges hard in the opposite
# direction (own optimum 0.05) -- with far less training data its probability outputs run
# systematically lower even when correct, so a high threshold filters out true suggestions
# rather than false ones. See kaggle_notebook/muril_finetune_kaggle_v2.ipynb for the sweep.
SUGGESTION_THRESHOLDS = {
    "Marathi-Roman": 0.05,
    "default": 0.85,  # English, Marathi-Dev, Hindi-Dev, Hinglish
}


def get_suggestion_threshold(language_variant):
    return SUGGESTION_THRESHOLDS.get(language_variant, SUGGESTION_THRESHOLDS["default"])

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError(
        "API_KEY environment variable must be set (deploy.sh generates one automatically)."
    )

_api_key_header = APIKeyHeader(name="X-API-Key")


def require_api_key(key: str = Depends(_api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


app = FastAPI(title="PMC Complaint/Suggestion Classifier")

# The UI is a separately hosted static site (deployment/frontend), not served by this
# process, so it calls this API cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model from {MODEL_DIR} onto {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
print("Model loaded.")


class PredictRequest(BaseModel):
    text: str
    # Optional: "English", "Marathi-Dev", "Marathi-Roman", "Hindi-Dev", or "Hinglish".
    # Omitted or anything else falls back to the shared default threshold.
    language_variant: str | None = None


class PredictResponse(BaseModel):
    text: str
    raw_label: str
    raw_confidence: float
    suggestion_probability: float
    suggestion_threshold_used: float
    language_variant: str
    final_label: str
    latency_ms: float


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest):
    start = time.time()
    text = req.text.strip()
    language_variant = req.language_variant or "Unknown"

    # Guardrail: the model is binary (complaint/suggestion) and was never trained with a
    # third option, so it will confidently force a label onto input that isn't a civic
    # complaint/suggestion at all (greetings, "test", gibberish, unrelated chatter).
    # Catch that here and route to a human instead of guessing.
    if is_offtopic(text):
        latency_ms = (time.time() - start) * 1000
        print(f"[predict] lang={language_variant!r} GUARDRAIL: off-topic/unclassifiable -> human_review | {text[:200]!r}")
        return PredictResponse(
            text=text,
            raw_label="human_review",
            raw_confidence=0.0,
            suggestion_probability=0.0,
            suggestion_threshold_used=0.0,
            language_variant=language_variant,
            final_label="human_review",
            latency_ms=latency_ms,
        )

    threshold = get_suggestion_threshold(language_variant)

    with torch.no_grad():
        enc = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]

    suggestion_probability = float(probs[LABEL2ID["suggestion"]].item())
    pred_id = LABEL2ID["suggestion"] if suggestion_probability >= threshold else LABEL2ID["complaint"]

    raw_label = ID2LABEL[pred_id]
    raw_confidence = float(probs[pred_id].item())
    latency_ms = (time.time() - start) * 1000

    print(f"[predict] lang={language_variant!r} threshold={threshold} "
          f"raw={raw_label}({raw_confidence:.2f}, P(suggestion)={suggestion_probability:.2f}) | {text[:200]!r}")

    return PredictResponse(
        text=text,
        raw_label=raw_label,
        raw_confidence=raw_confidence,
        suggestion_probability=suggestion_probability,
        suggestion_threshold_used=threshold,
        language_variant=language_variant,
        final_label=raw_label,
        latency_ms=latency_ms,
    )
