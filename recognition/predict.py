"""Live manual-capture prediction for the final eight-intent model."""

from pathlib import Path
from types import SimpleNamespace
import time

import json
import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from camera import (
    MAX_POSES,
    MODEL_PATH as HAND_MODEL_PATH,
    POSE_MODEL_PATH,
    associate_hands,
    build_pose_observations,
    draw_body_region,
    draw_hand,
    draw_upper_body,
    prepare_results_for_features,
)
from collect_data import (
    MAX_CAPTURED_FRAMES,
    sample_rejection_reason,
)
from face_identity import SessionFaceVerifier
from intents import INTENTS
from landmarks import extract_features
from signer_session import SessionState, SignerSessionController
from train import FEATURE_DIM, FEATURE_VERSION, FIXED_FRAMES, resample_sequence


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

        result_path = Path(__file__).resolve().parent / "latest_result.json"
        with result_path.open("w", encoding="utf-8") as file:
            json.dump(result, file)
            
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
    """Open the webcam and classify deliberately captured sign sequences."""
    artifact = load_model_artifact()
    print("Loaded final classes:", ", ".join(artifact["class_labels"]))

    if not HAND_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Hand Landmarker model not found: {HAND_MODEL_PATH}")
    if not POSE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pose Landmarker model not found: {POSE_MODEL_PATH}")

    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=MAX_POSES,
    )
    signer_session = SignerSessionController(SessionFaceVerifier())
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open the default webcam.")

    recording = False
    feature_frames = []
    valid_mask = []
    last_result = None
    status_message = "SPACE starts when signer is LOCKED"

    try:
        with (
            vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
            vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
        ):
            last_timestamp_ms = -1
            tracker = signer_session.tracker

            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )
                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000,
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms
                hand_results = hand_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
                pose_results = pose_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                current_time = time.monotonic()
                observations = build_pose_observations(pose_results)
                session_result = signer_session.update(
                    frame,
                    observations,
                    current_time,
                )
                active_signer = session_result.active_signer
                accepted_hands = associate_hands(
                    hand_results,
                    active_signer,
                    tracker,
                    current_time,
                )
                adapted_hands = prepare_results_for_features(accepted_hands)
                adapted_pose = (
                    SimpleNamespace(
                        pose_landmarks=SimpleNamespace(
                            landmark=active_signer["landmarks"]
                        )
                    )
                    if active_signer is not None
                    else None
                )
                frame_features = extract_features(adapted_hands, adapted_pose)
                if frame_features.shape != (artifact["feature_dimension"],):
                    raise RuntimeError(
                        f"Live feature shape {frame_features.shape} does not match "
                        f"model dimension {artifact['feature_dimension']}."
                    )

                if recording:
                    feature_frames.append(
                        frame_features
                        if active_signer is not None
                        else np.zeros(FEATURE_DIM, dtype=np.float32)
                    )
                    valid_mask.append(active_signer is not None)
                    if len(feature_frames) >= MAX_CAPTURED_FRAMES:
                        last_result, reason = finish_capture(
                            artifact,
                            feature_frames,
                            valid_mask,
                        )
                        recording = False
                        status_message = (
                            f"REJECTED: {reason}"
                            if reason
                            else f"PREDICTION: {last_result['intent']}"
                        )
                        if not last_result:
                            print(status_message)
                        feature_frames = []
                        valid_mask = []

                display_signer = (
                    active_signer or session_result.acquisition_candidate
                )
                if display_signer is not None:
                    draw_upper_body(frame, display_signer["landmarks"])
                    body_region = (
                        tracker.body_region
                        if tracker.locked
                        else display_signer["body_region"]
                    )
                    draw_body_region(frame, body_region, (0, 255, 255))
                for hand in accepted_hands:
                    draw_hand(
                        frame,
                        hand["landmarks"],
                        hand["handedness"],
                        hand["side"],
                    )

                valid_count = sum(valid_mask)
                prediction_text = (
                    last_result["intent"] if last_result else "--"
                )
                confidence_text = (
                    f"{last_result['confidence']:.3f}" if last_result else "--"
                )
                draw_status_lines(
                    frame,
                    (
                        f"Signer state: {session_result.state.value}",
                        "Capture: " + ("RECORDING" if recording else "READY"),
                        f"Frames: {len(feature_frames)} "
                        f"(valid {valid_count}, missing {len(valid_mask) - valid_count})",
                        f"Last prediction: {prediction_text}",
                        f"Confidence: {confidence_text}",
                        status_message,
                        "SPACE start/stop | R discard | Q quit",
                    ),
                )
                cv2.imshow("SgSL Two-Intent Live Prediction", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if recording:
                        last_result, reason = finish_capture(
                            artifact,
                            feature_frames,
                            valid_mask,
                        )
                        recording = False
                        status_message = (
                            f"REJECTED: {reason}"
                            if reason
                            else f"PREDICTION: {last_result['intent']}"
                        )
                        if not last_result:
                            print(status_message)
                        feature_frames = []
                        valid_mask = []
                    elif (
                        session_result.state is SessionState.LOCKED
                        and active_signer is not None
                    ):
                        recording = True
                        feature_frames = []
                        valid_mask = []
                        status_message = "Recording prediction sequence"
                        print("Recording prediction sequence...")
                    else:
                        status_message = "WAIT: signer must be LOCKED"
                        print(status_message)
                elif key in (ord("r"), ord("R")):
                    recording = False
                    feature_frames = []
                    valid_mask = []
                    status_message = "Current sequence discarded"
                    print(status_message)
                elif key in (ord("q"), ord("Q")):
                    if recording:
                        print("Unclassified sequence discarded on quit.")
                    break
    finally:
        signer_session.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_live_prediction()
