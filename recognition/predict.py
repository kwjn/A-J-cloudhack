"""Standalone OpenCV predictor for local recognition debugging only.

The deployed application runs camera capture and inference inside Streamlit via
``LiveSignPredictor`` and does not launch this script. Dataset recording remains
available separately through ``collect_data.py``.
"""

from pathlib import Path
import sys


RECOGNITION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RECOGNITION_DIR.parent
for import_path in (RECOGNITION_DIR, PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


import cv2
import joblib
import numpy as np

try:
    from .collect_data import sample_rejection_reason
    from .intents import INTENTS
    from .train import (
        FEATURE_DIM,
        FEATURE_VERSION,
        FIXED_FRAMES,
        resample_sequence,
    )
except ImportError:  # Keep direct ``python recognition/predict.py`` support.
    from collect_data import sample_rejection_reason
    from intents import INTENTS
    from train import FEATURE_DIM, FEATURE_VERSION, FIXED_FRAMES, resample_sequence


from services.recognition_bridge import publish_prediction


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "sgsl_classifier.joblib"
)
EXPECTED_CLASSES = tuple(sorted(INTENTS))


def load_model_artifact(model_path=MODEL_PATH) -> dict:
    """Load and validate the final model and preprocessing contract."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Final classifier not found: {model_path}")

    try:
        artifact = joblib.load(model_path)
    except Exception as error:
        raise RuntimeError(f"Could not load final classifier: {model_path}") from error

    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dictionary.")

    required_keys = {
        "model",
        "preprocessing",
        "fixed_frame_count",
        "feature_dimension",
        "feature_version",
        "class_labels",
    }
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise ValueError(
            "Model artifact is missing: " + ", ".join(sorted(missing_keys))
        )

    if artifact["feature_dimension"] != FEATURE_DIM:
        raise ValueError(
            f"Model expects feature dimension {artifact['feature_dimension']}; "
            f"live extraction provides {FEATURE_DIM}."
        )
    if artifact["feature_version"] != FEATURE_VERSION:
        raise ValueError(
            f"Model feature version {artifact['feature_version']!r} is incompatible "
            f"with {FEATURE_VERSION!r}."
        )

    preprocessing = artifact["preprocessing"]
    if not isinstance(preprocessing, dict):
        raise ValueError("Model preprocessing metadata must be a dictionary.")
    if preprocessing.get("method") != "linear_interpolation":
        raise ValueError("Model does not specify linear-interpolation preprocessing.")
    if preprocessing.get("flatten") is not True:
        raise ValueError("Model does not specify flattened temporal features.")
    fixed_frames = preprocessing.get("fixed_frames")
    if fixed_frames != artifact["fixed_frame_count"] or not isinstance(
        fixed_frames, int
    ) or fixed_frames != FIXED_FRAMES:
        raise ValueError("Model fixed-frame preprocessing metadata is invalid.")

    model = artifact["model"]
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("Model must provide predict() and predict_proba().")
    model_classes = tuple(str(label) for label in model.classes_)
    artifact_classes = tuple(str(label) for label in artifact["class_labels"])
    if artifact_classes != model_classes:
        raise ValueError("Artifact and classifier class ordering do not match.")
    if set(model_classes) != set(EXPECTED_CLASSES):
        raise ValueError(
            f"Final classifier classes must be {EXPECTED_CLASSES}; "
            f"found {model_classes}."
        )
    if any(label not in INTENTS for label in model_classes):
        raise ValueError("Classifier contains a class missing from intents.py.")

    return artifact


def preprocess_sequence(artifact: dict, sequence) -> np.ndarray:
    """Apply the exact shared training resampling and flattening steps."""
    features = np.asarray(sequence, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Sequence must be 2D, got shape {features.shape}.")
    if features.shape[0] == 0:
        raise ValueError("Sequence is empty.")
    if features.shape[1] != artifact["feature_dimension"]:
        raise ValueError(
            f"Sequence has {features.shape[1]} features per frame; "
            f"expected {artifact['feature_dimension']}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Sequence contains NaN or infinity.")

    fixed_frames = artifact["fixed_frame_count"]
    resampled = resample_sequence(features, fixed_frames=fixed_frames)
    return resampled.reshape(1, fixed_frames * artifact["feature_dimension"])


def classify_sequence(artifact: dict, sequence) -> dict:
    """Return intent metadata and model confidence for one sequence."""
    model_input = preprocess_sequence(artifact, sequence)
    model = artifact["model"]
    predicted_intent = str(model.predict(model_input)[0])
    model_classes = [str(label) for label in model.classes_]
    if predicted_intent not in model_classes:
        raise RuntimeError("Classifier returned an unknown class.")
    probabilities = model.predict_proba(model_input)[0]
    confidence = float(probabilities[model_classes.index(predicted_intent)])

    metadata = INTENTS[predicted_intent]
    return {
        "intent": predicted_intent,
        "text": metadata["english"],
        "confidence": confidence,
        "critical": metadata["critical"],
    }


def print_capture_shape_diagnostics(artifact, feature_frames, valid_mask) -> None:
    """Print compact temporal and feature-presence diagnostics."""
    sequence = np.asarray(feature_frames, dtype=np.float32)
    captured_count = len(feature_frames)
    valid_count = int(sum(valid_mask))
    print("\n--- LIVE SEQUENCE DEBUG ---")
    print(f"Captured frames: {captured_count}")
    print(f"Valid frames: {valid_count}/{captured_count}")
    print(f"Sequence shape before resampling: {sequence.shape}")

    if sequence.ndim != 2 or sequence.shape[0] == 0:
        print("Sequence shape after resampling: unavailable")
        print("--- END LIVE SEQUENCE DEBUG ---")
        return

    resampled = resample_sequence(
        sequence,
        fixed_frames=artifact["fixed_frame_count"],
    )
    zero_count = int(np.count_nonzero(sequence == 0))
    zero_proportion = zero_count / sequence.size
    left_absent = int(np.all(sequence[:, :63] == 0, axis=1).sum())
    right_absent = int(np.all(sequence[:, 63:126] == 0, axis=1).sum())
    body_absent = int(np.all(sequence[:, 126:144] == 0, axis=1).sum())

    print(f"Sequence shape after resampling: {resampled.shape}")
    print(
        f"Zero values: {zero_count}/{sequence.size} "
        f"({zero_proportion:.2%})"
    )
    print(f"Left-hand slot all-zero frames: {left_absent}/{captured_count}")
    print(f"Right-hand slot all-zero frames: {right_absent}/{captured_count}")
    print(f"Body slot all-zero frames: {body_absent}/{captured_count}")


def print_prediction_result(artifact, feature_frames, result) -> None:
    """Print metadata and the three highest class probabilities."""
    model_input = preprocess_sequence(artifact, feature_frames)
    model = artifact["model"]
    model_classes = [str(label) for label in model.classes_]
    probabilities = model.predict_proba(model_input)[0]

    print(f"classifier.classes_: {model_classes}")
    print(f"PREDICTION: {result['intent']}")
    print(f"CONFIDENCE: {result['confidence']:.4f}")
    print(f"TEXT: {result['text']}")
    print(f"CRITICAL: {str(result['critical']).lower()}")
    print("Top predictions:")
    ranked_predictions = sorted(
        zip(model_classes, probabilities),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    for class_label, probability in ranked_predictions[:3]:
        print(f"{class_label:<18} {float(probability):.6f}")
    print("--- END LIVE SEQUENCE DEBUG ---")


def finish_capture(artifact, feature_frames, valid_mask):
    """Validate and classify a completed manual capture."""
    print_capture_shape_diagnostics(artifact, feature_frames, valid_mask)
    rejection_reason = sample_rejection_reason(valid_mask)
    if rejection_reason is not None:
        print(f"Classification skipped: {rejection_reason}")
        print("--- END LIVE SEQUENCE DEBUG ---")
        return None, rejection_reason
    try:
        result = classify_sequence(artifact, feature_frames)
        print_prediction_result(artifact, feature_frames, result)
        try:
            result = publish_prediction(result)
            print(f"Published prediction ID: {result['prediction_id']}")
        except OSError as error:
            print(f"Warning: could not publish prediction: {error}")
        return result, None
    except (ValueError, RuntimeError) as error:
        print(f"Classification failed: {error}")
        print("--- END LIVE SEQUENCE DEBUG ---")
        return None, str(error)


def draw_status_lines(frame, lines) -> None:
    """Draw readable live-prediction state on a camera frame."""
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, 35 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def show_live_prediction() -> None:
    """Run the reusable predictor as a local OpenCV CLI debug tool."""
    try:
        from .live_predictor import LiveSignPredictor
    except ImportError:
        from live_predictor import LiveSignPredictor

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open the default webcam.")

    try:
        with LiveSignPredictor() as predictor:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                annotated_frame, _ = predictor.process_frame(frame)
                cv2.imshow("SgSL Two-Intent Live Prediction", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    predictor.start_recording()
                elif key in (ord("r"), ord("R")):
                    predictor.discard_recording()
                elif key in (ord("q"), ord("Q")):
                    if predictor.recording:
                        print("Unclassified sequence discarded on quit.")
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_live_prediction()
