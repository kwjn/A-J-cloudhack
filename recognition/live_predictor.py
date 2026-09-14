"""Reusable per-frame live SgSL prediction pipeline."""

from types import SimpleNamespace
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


RECOGNITION_DIR = Path(__file__).resolve().parent
if str(RECOGNITION_DIR) not in sys.path:
    sys.path.insert(0, str(RECOGNITION_DIR))


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
from collect_data import MAX_CAPTURED_FRAMES
from face_identity import SessionFaceVerifier
from landmarks import extract_features
from predict import draw_status_lines, finish_capture, load_model_artifact
from signer_session import SessionState, SignerSessionController
from train import FEATURE_DIM


class LiveSignPredictor:
    """Own the stateful MediaPipe-to-classification processing pipeline."""

    def __init__(self, control_hint: str | None = None):
        self.artifact = load_model_artifact()
        print("Loaded final classes:", ", ".join(self.artifact["class_labels"]))

        if not HAND_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {HAND_MODEL_PATH}"
            )
        if not POSE_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Pose Landmarker model not found: {POSE_MODEL_PATH}"
            )

        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(HAND_MODEL_PATH)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(POSE_MODEL_PATH)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=MAX_POSES,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            hand_options
        )
        try:
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(
                pose_options
            )
        except Exception:
            self.hand_landmarker.close()
            raise

        self.signer_session = SignerSessionController(SessionFaceVerifier())
        self.tracker = self.signer_session.tracker
        self.last_timestamp_ms = -1
        self.recording = False
        self.feature_frames = []
        self.valid_mask = []
        self.last_result = None
        self.control_hint = control_hint or "SPACE start/stop | R discard | Q quit"
        self.status_message = (
            "Waiting for signer lock"
            if control_hint
            else "SPACE starts when signer is LOCKED"
        )
        self.session_result = None
        self.active_signer = None
        self.closed = False

    def _finish_recording(self):
        result, reason = finish_capture(
            self.artifact,
            self.feature_frames,
            self.valid_mask,
        )
        self.recording = False
        self.status_message = (
            f"REJECTED: {reason}"
            if reason
            else f"PREDICTION: {result['intent']}"
        )
        self.last_result = result
        if result is None:
            print(self.status_message)
        self.feature_frames = []
        self.valid_mask = []
        return result

    def start_recording(self):
        """Toggle capture exactly as the CLI SPACE control does."""
        if self.recording:
            return self._finish_recording()

        if (
            self.session_result is not None
            and self.session_result.state is SessionState.LOCKED
            and self.active_signer is not None
        ):
            self.recording = True
            self.feature_frames = []
            self.valid_mask = []
            self.status_message = "Recording prediction sequence"
            print("Recording prediction sequence...")
        else:
            self.status_message = "WAIT: signer must be LOCKED"
            print(self.status_message)
        return None

    def discard_recording(self) -> None:
        """Discard the active sequence exactly as the CLI R control does."""
        self.recording = False
        self.feature_frames = []
        self.valid_mask = []
        self.status_message = "Current sequence discarded"
        print(self.status_message)

    def process_frame(self, frame) -> tuple[np.ndarray, dict | None]:
        """Process and annotate one BGR frame, returning any new prediction."""
        if self.closed:
            raise RuntimeError("LiveSignPredictor is closed.")
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy array.")

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(
            time.monotonic_ns() // 1_000_000,
            self.last_timestamp_ms + 1,
        )
        self.last_timestamp_ms = timestamp_ms
        hand_results = self.hand_landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )
        pose_results = self.pose_landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        current_time = time.monotonic()
        observations = build_pose_observations(pose_results)
        self.session_result = self.signer_session.update(
            frame,
            observations,
            current_time,
        )
        self.active_signer = self.session_result.active_signer
        accepted_hands = associate_hands(
            hand_results,
            self.active_signer,
            self.tracker,
            current_time,
        )
        adapted_hands = prepare_results_for_features(accepted_hands)
        adapted_pose = (
            SimpleNamespace(
                pose_landmarks=SimpleNamespace(
                    landmark=self.active_signer["landmarks"]
                )
            )
            if self.active_signer is not None
            else None
        )
        frame_features = extract_features(adapted_hands, adapted_pose)
        if frame_features.shape != (self.artifact["feature_dimension"],):
            raise RuntimeError(
                f"Live feature shape {frame_features.shape} does not match "
                f"model dimension {self.artifact['feature_dimension']}."
            )

        prediction = None
        if self.recording:
            self.feature_frames.append(
                frame_features
                if self.active_signer is not None
                else np.zeros(FEATURE_DIM, dtype=np.float32)
            )
            self.valid_mask.append(self.active_signer is not None)
            if len(self.feature_frames) >= MAX_CAPTURED_FRAMES:
                prediction = self._finish_recording()

        display_signer = (
            self.active_signer or self.session_result.acquisition_candidate
        )
        if display_signer is not None:
            draw_upper_body(frame, display_signer["landmarks"])
            body_region = (
                self.tracker.body_region
                if self.tracker.locked
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

        valid_count = sum(self.valid_mask)
        prediction_text = (
            self.last_result["intent"] if self.last_result else "--"
        )
        confidence_text = (
            f"{self.last_result['confidence']:.3f}"
            if self.last_result
            else "--"
        )
        draw_status_lines(
            frame,
            (
                f"Signer state: {self.session_result.state.value}",
                "Capture: " + ("RECORDING" if self.recording else "READY"),
                f"Frames: {len(self.feature_frames)} "
                f"(valid {valid_count}, missing "
                f"{len(self.valid_mask) - valid_count})",
                f"Last prediction: {prediction_text}",
                f"Confidence: {confidence_text}",
                self.status_message,
                self.control_hint,
            ),
        )
        return frame, prediction

    def close(self) -> None:
        """Release MediaPipe and signer-session resources."""
        if self.closed:
            return
        self.closed = True
        self.signer_session.close()
        self.hand_landmarker.close()
        self.pose_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
