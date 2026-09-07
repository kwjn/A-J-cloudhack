"""Session-local anonymous face verification for signer tracking."""

from enum import Enum
from pathlib import Path

import cv2
import numpy as np


# Initial values only. Calibrate them with the target camera and environment.
FACE_DETECTION_SCORE_THRESHOLD = 0.90
FACE_MATCH_COSINE_THRESHOLD = 0.45
FACE_MISMATCH_COSINE_THRESHOLD = 0.30
FACE_REFERENCE_CONSISTENCY_THRESHOLD = 0.45
FACE_ASSOCIATION_SHOULDER_WIDTHS = 1.25

MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"
FACE_DETECTOR_MODEL_PATH = MODEL_DIRECTORY / "face_detection_yunet_2026may.onnx"
FACE_RECOGNIZER_MODEL_PATH = (
    MODEL_DIRECTORY / "face_recognition_sface_2021dec.onnx"
)


class FaceEvidence(Enum):
    """Identity evidence for one pose candidate in the current frame."""

    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    AMBIGUOUS = "AMBIGUOUS"
    UNAVAILABLE = "UNAVAILABLE"


def cosine_similarity(first, second) -> float:
    """Return cosine similarity for two normalized or raw embeddings."""
    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return -1.0
    return float(np.dot(first, second) / denominator)


def normalized(embedding):
    """Return one flat unit-length embedding, or None for invalid input."""
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    length = float(np.linalg.norm(embedding))
    if length <= 1e-12:
        return None
    return embedding / length


class SessionFaceVerifier:
    """Keep an anonymous face reference in memory for one signer session."""

    def __init__(
        self,
        detector_model_path=FACE_DETECTOR_MODEL_PATH,
        recognizer_model_path=FACE_RECOGNIZER_MODEL_PATH,
        detection_threshold=FACE_DETECTION_SCORE_THRESHOLD,
        match_threshold=FACE_MATCH_COSINE_THRESHOLD,
        mismatch_threshold=FACE_MISMATCH_COSINE_THRESHOLD,
        consistency_threshold=FACE_REFERENCE_CONSISTENCY_THRESHOLD,
    ):
        detector_model_path = Path(detector_model_path)
        recognizer_model_path = Path(recognizer_model_path)
        if not detector_model_path.is_file():
            raise FileNotFoundError(f"Face detector model not found: {detector_model_path}")
        if not recognizer_model_path.is_file():
            raise FileNotFoundError(
                f"Face recognizer model not found: {recognizer_model_path}"
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(detector_model_path),
            "",
            (320, 320),
            detection_threshold,
            0.3,
            5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(recognizer_model_path),
            "",
        )
        self.match_threshold = match_threshold
        self.mismatch_threshold = mismatch_threshold
        self.consistency_threshold = consistency_threshold
        self.reference = None
        self.acquisition_samples = []

    @property
    def reference_sample_count(self):
        return len(self.acquisition_samples)

    def reset(self):
        """Forget all temporary biometric state."""
        self.reference = None
        self.acquisition_samples = []

    def reset_acquisition(self):
        """Discard provisional samples without changing a locked reference."""
        if self.reference is None:
            self.acquisition_samples = []

    def add_acquisition_sample(self, embedding) -> bool:
        """Add an internally consistent sample for a provisional signer."""
        embedding = normalized(embedding)
        if embedding is None:
            return False
        if self.acquisition_samples:
            provisional_reference = normalized(
                np.mean(self.acquisition_samples, axis=0)
            )
            if (
                cosine_similarity(provisional_reference, embedding)
                < self.consistency_threshold
            ):
                return False
        self.acquisition_samples.append(embedding)
        return True

    def finalize_reference(self) -> bool:
        """Create the session reference from the collected memory-only samples."""
        if not self.acquisition_samples:
            return False
        self.reference = normalized(np.mean(self.acquisition_samples, axis=0))
        return self.reference is not None

    def classify(self, embedding) -> FaceEvidence:
        """Classify a visible face against the current anonymous reference."""
        if self.reference is None or embedding is None:
            return FaceEvidence.UNAVAILABLE
        similarity = cosine_similarity(self.reference, embedding)
        if similarity >= self.match_threshold:
            return FaceEvidence.MATCH
        if similarity <= self.mismatch_threshold:
            return FaceEvidence.MISMATCH
        return FaceEvidence.AMBIGUOUS

    def embeddings_for_observations(self, frame, observations):
        """Return face embeddings keyed by their associated pose index."""
        if frame is None or not observations:
            return {}

        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return {}

        possible_matches = []
        for face_index, face in enumerate(faces):
            face_center = (
                (face[0] + face[2] / 2.0) / width,
                (face[1] + face[3] / 2.0) / height,
            )
            for pose_index, observation in enumerate(observations):
                shoulder_midpoint = observation["shoulder_midpoint"]
                shoulder_width = observation["shoulder_width"]
                expected_head = (
                    shoulder_midpoint[0],
                    shoulder_midpoint[1] - shoulder_width * 0.75,
                )
                distance = np.hypot(
                    face_center[0] - expected_head[0],
                    face_center[1] - expected_head[1],
                )
                if distance <= shoulder_width * FACE_ASSOCIATION_SHOULDER_WIDTHS:
                    possible_matches.append((distance, face_index, pose_index))

        embeddings = {}
        used_faces = set()
        used_poses = set()
        for _, face_index, pose_index in sorted(possible_matches):
            if face_index in used_faces or pose_index in used_poses:
                continue
            aligned = self.recognizer.alignCrop(frame, faces[face_index])
            embedding = normalized(self.recognizer.feature(aligned))
            if embedding is not None:
                embeddings[pose_index] = embedding
                used_faces.add(face_index)
                used_poses.add(pose_index)

        return embeddings
