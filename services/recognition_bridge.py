"""Atomic local JSON bridge between recognition and the Streamlit app."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LATEST_PREDICTION_PATH = PROJECT_ROOT / "runtime" / "latest_prediction.json"
REQUIRED_FIELDS = {
    "intent",
    "text",
    "confidence",
    "critical",
    "prediction_id",
    "timestamp",
}


def validate_prediction(payload) -> dict:
    """Return a normalized prediction or raise ValueError for invalid data."""
    if not isinstance(payload, dict):
        raise ValueError("Prediction must be a JSON object.")
    missing = REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError("Prediction is missing required fields.")

    for field in ("intent", "text", "prediction_id", "timestamp"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"Prediction field {field!r} must be non-empty text.")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("Prediction confidence must be numeric.")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("Prediction confidence must be between zero and one.")
    if not isinstance(payload["critical"], bool):
        raise ValueError("Prediction critical field must be boolean.")

    return {
        "intent": payload["intent"].strip(),
        "text": payload["text"].strip(),
        "confidence": float(confidence),
        "critical": payload["critical"],
        "prediction_id": payload["prediction_id"].strip(),
        "timestamp": payload["timestamp"].strip(),
    }


def publish_prediction(result: dict, path=None) -> dict:
    """Add event metadata and atomically publish the latest prediction."""
    destination = Path(path) if path is not None else LATEST_PREDICTION_PATH
    payload = validate_prediction(
        {
            **result,
            "prediction_id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.name}.{payload['prediction_id']}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as temporary_file:
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return payload


def read_latest_prediction(path=None):
    """Return the latest valid prediction, or None when unavailable/invalid."""
    source = Path(path) if path is not None else LATEST_PREDICTION_PATH
    try:
        with source.open("r", encoding="utf-8") as prediction_file:
            return validate_prediction(json.load(prediction_file))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def read_new_prediction(last_prediction_id=None, path=None):
    """Return only a prediction not already consumed by this UI session."""
    prediction = read_latest_prediction(path=path)
    if prediction is None or prediction["prediction_id"] == last_prediction_id:
        return None
    return prediction
