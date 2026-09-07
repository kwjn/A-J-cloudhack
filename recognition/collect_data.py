"""Record labelled temporal feature sequences for supported SgSL intents."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from camera import (
    MAX_POSES,
    MODEL_PATH,
    POSE_MODEL_PATH,
    associate_hands,
    build_pose_observations,
    draw_body_region,
    draw_hand,
    draw_upper_body,
    prepare_results_for_features,
)
from face_identity import SessionFaceVerifier
from intents import INTENTS
from landmarks import extract_features
from signer_session import SessionState, SignerSessionController


DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
FEATURE_VERSION = "v1"
FEATURE_DIM = 144
MIN_CAPTURED_FRAMES = 15
MAX_CAPTURED_FRAMES = 300
MIN_VALID_FRAME_PROPORTION = 0.80


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record one temporal SgSL training sample at a time."
    )
    parser.add_argument(
        "--intent",
        required=True,
        choices=tuple(INTENTS),
        help="Semantic intent ID. Valid values: " + ", ".join(INTENTS),
    )
    return parser.parse_args()


def prepare_dataset_directories(data_directory=DATA_DIRECTORY):
    """Create one dataset directory for every supported semantic intent."""
    for intent in INTENTS:
        (Path(data_directory) / intent).mkdir(parents=True, exist_ok=True)


def next_sample_number(intent, data_directory=DATA_DIRECTORY):
    """Return the next unused numeric sample suffix for an intent."""
    intent_directory = Path(data_directory) / intent
    numbers = []
    for path in intent_directory.glob(f"{intent}_*.npz"):
        suffix = path.stem.removeprefix(f"{intent}_")
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers, default=0) + 1


def sample_rejection_reason(valid_mask):
    """Return why a recording is unusable, or None when it may be saved."""
    frame_count = len(valid_mask)
    if frame_count < MIN_CAPTURED_FRAMES:
        return (
            f"only {frame_count} frames captured; minimum is "
            f"{MIN_CAPTURED_FRAMES}"
        )
    valid_proportion = sum(valid_mask) / frame_count
    if valid_proportion < MIN_VALID_FRAME_PROPORTION:
        return (
            f"only {valid_proportion:.0%} valid signer frames; minimum is "
            f"{MIN_VALID_FRAME_PROPORTION:.0%}"
        )
    return None


def save_sample(
    intent,
    feature_frames,
    valid_mask,
    sample_number,
    data_directory=DATA_DIRECTORY,
):
    """Validate and save one frames-by-features temporal sample."""
    rejection_reason = sample_rejection_reason(valid_mask)
    if rejection_reason is not None:
        return None, rejection_reason

    features = np.asarray(feature_frames, dtype=np.float32)
    if features.shape != (len(valid_mask), FEATURE_DIM):
        return None, f"unexpected feature shape {features.shape}"

    valid_mask_array = np.asarray(valid_mask, dtype=np.bool_)
    frame_count = len(valid_mask)
    valid_frame_count = int(valid_mask_array.sum())
    invalid_frame_count = frame_count - valid_frame_count
    valid_frame_proportion = valid_frame_count / frame_count
    sample_id = f"{intent}_{sample_number:04d}"
    output_path = Path(data_directory) / intent / f"{sample_id}.npz"
    if output_path.exists():
        return None, f"sample already exists: {output_path.name}"

    np.savez_compressed(
        output_path,
        intent=np.array(intent),
        features=features,
        frame_count=np.array(frame_count, dtype=np.int64),
        valid_frame_count=np.array(valid_frame_count, dtype=np.int64),
        invalid_frame_count=np.array(invalid_frame_count, dtype=np.int64),
        valid_frame_proportion=np.array(valid_frame_proportion, dtype=np.float32),
        valid_mask=valid_mask_array,
        sample_id=np.array(sample_id),
        timestamp=np.array(datetime.now(timezone.utc).isoformat()),
        feature_version=np.array(FEATURE_VERSION),
        feature_dim=np.array(FEATURE_DIM, dtype=np.int64),
    )
    return output_path, None


def draw_status_lines(frame, lines):
    """Draw readable collection status text on the current frame."""
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


def collect_data(intent, data_directory=DATA_DIRECTORY):
    """Open the webcam and deliberately record labelled temporal samples."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Hand Landmarker model not found: {MODEL_PATH}")
    if not POSE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pose Landmarker model not found: {POSE_MODEL_PATH}")

    prepare_dataset_directories(data_directory)
    intent_directory = Path(data_directory) / intent
    sample_number = next_sample_number(intent, data_directory)
    saved_sample_count = len(list(intent_directory.glob(f"{intent}_*.npz")))

    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
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
                hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

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

                if recording:
                    feature_frames.append(
                        frame_features
                        if active_signer is not None
                        else np.zeros(FEATURE_DIM, dtype=np.float32)
                    )
                    valid_mask.append(active_signer is not None)

                    if len(feature_frames) >= MAX_CAPTURED_FRAMES:
                        output_path, reason = save_sample(
                            intent,
                            feature_frames,
                            valid_mask,
                            sample_number,
                            data_directory,
                        )
                        recording = False
                        if output_path is None:
                            status_message = f"REJECTED: {reason}"
                            print(status_message)
                        else:
                            status_message = f"SAVED: {output_path.name}"
                            print(status_message)
                            saved_sample_count += 1
                            sample_number = next_sample_number(intent, data_directory)
                        feature_frames = []
                        valid_mask = []

                display_signer = active_signer or session_result.acquisition_candidate
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
                invalid_count = len(valid_mask) - valid_count
                draw_status_lines(
                    frame,
                    (
                        f"Intent: {intent}",
                        f"Signer: {session_result.state.value}",
                        "Recorder: " + ("RECORDING" if recording else "READY"),
                        f"Sample: {sample_number:04d}",
                        f"Frames: {len(feature_frames)} "
                        f"(valid {valid_count}, missing {invalid_count})",
                        f"Saved samples: {saved_sample_count}",
                        status_message,
                        "SPACE start/stop | R discard | Q quit",
                    ),
                )
                cv2.imshow("SgSL Dataset Recorder", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if recording:
                        output_path, reason = save_sample(
                            intent,
                            feature_frames,
                            valid_mask,
                            sample_number,
                            data_directory,
                        )
                        recording = False
                        if output_path is None:
                            status_message = f"REJECTED: {reason}"
                            print(status_message)
                        else:
                            status_message = f"SAVED: {output_path.name}"
                            print(status_message)
                            saved_sample_count += 1
                            sample_number = next_sample_number(intent, data_directory)
                        feature_frames = []
                        valid_mask = []
                    elif (
                        session_result.state is SessionState.LOCKED
                        and active_signer is not None
                    ):
                        recording = True
                        feature_frames = []
                        valid_mask = []
                        status_message = "Recording started"
                        print(f"Recording {intent} sample {sample_number:04d}...")
                    else:
                        status_message = "WAIT: signer must be LOCKED"
                        print(status_message)
                elif key in (ord("r"), ord("R")):
                    recording = False
                    feature_frames = []
                    valid_mask = []
                    status_message = "Current recording discarded"
                    print(status_message)
                elif key in (ord("q"), ord("Q")):
                    if recording:
                        print("Unsaved recording discarded on quit.")
                    break
    finally:
        signer_session.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    arguments = parse_args()
    collect_data(arguments.intent)
