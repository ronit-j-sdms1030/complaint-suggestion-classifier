"""
FastAPI backend for the fine-tuned MuRIL complaint/suggestion classifier.
Loads the model once at startup and exposes POST /predict for classification requests.

Run: uvicorn api:app --host 0.0.0.0 --port 8000   (from this directory)
"""
import re
import sys
import time
from pathlib import Path

import fasttext
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from guardrails import is_offtopic  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = str(PROJECT_ROOT / "muril-pmc-classifier")

# Same IndicLID FastText language-ID pipeline used to label every row in this project's
# training/eval data, so "auto" detection here matches what the model was calibrated
# against rather than guessing with different logic.
INDICLID_LABEL_MAP = {
    "eng_Latn": "English", "hin_Deva": "Hindi-Dev", "mar_Deva": "Marathi-Dev",
    "hin_Latn": "Hinglish", "mar_Latn": "Marathi-Roman",
}


def _char_percent_check(s):
    special = len(re.findall(r'[@_!#$%^&*()<>?/\|}{~:]', s))
    spaces = len(re.findall(r'\s', s))
    newlines = len(re.findall(r'\n', s))
    total_chars = len(s) - (special + spaces + newlines)
    en_chars = len(re.findall(r'[a-zA-Z0-9]', s))
    return (en_chars / total_chars) if total_chars else 0


print("Loading IndicLID language-ID models...")
_indiclid_native = fasttext.load_model(str(PROJECT_ROOT / "indiclid-ftn" / "model_baseline_roman.bin"))
_indiclid_roman = fasttext.load_model(str(PROJECT_ROOT / "indiclid-ftr" / "model_baseline_roman.bin"))
print("IndicLID models loaded.")


def detect_language(text):
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return "Unknown"
    model = _indiclid_roman if _char_percent_check(cleaned) > 0.5 else _indiclid_native
    # fasttext's predict() hits a numpy 2.x incompatibility when passed a bare string
    # (works fine with a list -- same pattern build_finetune_dataset.py uses).
    raw_label = model.predict([cleaned])[0][0][0].replace("__label__", "")
    return INDICLID_LABEL_MAP.get(raw_label, "Other/Unknown")
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
    language_variant: str | None = None  # e.g. "English", "Marathi-Dev", "Marathi-Roman", ...
                                          # unknown/omitted -> shared default threshold


class PredictResponse(BaseModel):
    text: str
    raw_label: str
    raw_confidence: float
    suggestion_probability: float
    suggestion_threshold_used: float
    language_variant: str
    language_auto_detected: bool
    final_label: str
    latency_ms: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()
    text = req.text.strip()

    language_auto_detected = not req.language_variant
    language_variant = req.language_variant or detect_language(text)

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
            language_auto_detected=language_auto_detected,
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

    detected_note = " (auto-detected)" if language_auto_detected else " (manual)"
    print(f"[predict] lang={language_variant!r}{detected_note} threshold={threshold} "
          f"raw={raw_label}({raw_confidence:.2f}, P(suggestion)={suggestion_probability:.2f}) | {text[:200]!r}")

    return PredictResponse(
        text=text,
        raw_label=raw_label,
        raw_confidence=raw_confidence,
        suggestion_probability=suggestion_probability,
        suggestion_threshold_used=threshold,
        language_variant=language_variant,
        language_auto_detected=language_auto_detected,
        final_label=raw_label,
        latency_ms=latency_ms,
    )
