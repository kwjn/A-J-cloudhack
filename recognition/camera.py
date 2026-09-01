"""Display MediaPipe Tasks hand landmarks on a mirrored webcam feed."""

from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from landmarks import extract_features


MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def draw_hand(frame, hand_landmarks, handedness) -> None:
    """Draw one hand's 21 landmarks, connections and handedness label."""
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_landmarks
    ]

    for connection in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[connection.start],
            points[connection.end],
            (0, 255, 0),
            2,
        )

    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)

    label = handedness[0].category_name
    wrist_x, wrist_y = points[0]
    cv2.putText(
        frame,
        label,
        (max(0, wrist_x - 20), max(30, wrist_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def prepare_results_for_features(results):
    """Adapt Tasks results for the existing landmark feature extractor."""
    hand_landmarks = [
        SimpleNamespace(landmark=landmarks)
        for landmarks in results.hand_landmarks
    ]
    handedness = [
        SimpleNamespace(
            classification=[SimpleNamespace(label=categories[0].category_name)]
        )
        for categories in results.handedness
    ]
    return SimpleNamespace(
        multi_hand_landmarks=hand_landmarks,
        multi_handedness=handedness,
    )


def show_camera() -> None:
    """Open the default webcam and display its live video feed."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Hand Landmarker model not found: {MODEL_PATH}"
        )

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open the default webcam.")

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            last_timestamp_ms = -1

            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                # Mirror the frame so it behaves like a selfie camera.
                frame = cv2.flip(frame, 1)

                # MediaPipe Tasks expects an RGB MediaPipe Image.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                # VIDEO mode requires a timestamp that increases every frame.
                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000,
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                adapted_results = prepare_results_for_features(results)
                features = extract_features(adapted_results, None)

                for hand_landmarks, handedness in zip(
                    results.hand_landmarks,
                    results.handedness,
                ):
                    draw_hand(frame, hand_landmarks, handedness)

                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("SgSL Camera", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("p"), ord("P")):
                    print("Feature vector:", features)
                    print("Vector length:", len(features))

                if key in (ord("q"), ord("Q")):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_camera()
