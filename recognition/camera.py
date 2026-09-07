"""Display MediaPipe Tasks hand landmarks on a mirrored webcam feed."""

from pathlib import Path
from types import SimpleNamespace
import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from face_identity import SessionFaceVerifier
from landmarks import extract_features
from signer_session import SessionState, SignerSessionController


MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
POSE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker.task"
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
MAX_POSES = 4
HAND_MEMORY_SECONDS = 0.75
POSE_LANDMARK_INDICES = (11, 12, 13, 14, 15, 16)
POSE_CONNECTIONS = (
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
)
POSE_WRIST_INDICES = {"Left": 16, "Right": 15}


def point_distance(first, second) -> float:
    """Return the 2D distance between two normalized image points."""
    return math.hypot(first[0] - second[0], first[1] - second[1])


def landmark_is_visible(landmark) -> bool:
    """Return whether a pose landmark is visible and inside the frame."""
    return (
        (landmark.visibility or 0.0) >= 0.5
        and 0.0 <= landmark.x <= 1.0
        and 0.0 <= landmark.y <= 1.0
    )


def build_pose_observations(pose_results) -> list:
    """Summarize every detected pose using position and body-region data."""
    observations = []

    for pose_landmarks in pose_results.pose_landmarks:
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        if not (
            landmark_is_visible(left_shoulder)
            and landmark_is_visible(right_shoulder)
        ):
            continue

        shoulder_midpoint = (
            (left_shoulder.x + right_shoulder.x) / 2.0,
            (left_shoulder.y + right_shoulder.y) / 2.0,
        )
        shoulder_width = point_distance(
            (left_shoulder.x, left_shoulder.y),
            (right_shoulder.x, right_shoulder.y),
        )
        if shoulder_width <= 1e-6:
            continue

        visible_landmarks = [
            pose_landmarks[index]
            for index in POSE_LANDMARK_INDICES
            if landmark_is_visible(pose_landmarks[index])
        ]
        x_values = [landmark.x for landmark in visible_landmarks]
        y_values = [landmark.y for landmark in visible_landmarks]
        body_region = (
            min(x_values),
            min(y_values),
            max(x_values),
            max(y_values),
        )
        body_center = (
            (body_region[0] + body_region[2]) / 2.0,
            (body_region[1] + body_region[3]) / 2.0,
        )

        observations.append(
            {
                "landmarks": pose_landmarks,
                "shoulder_midpoint": shoulder_midpoint,
                "shoulder_width": shoulder_width,
                "body_region": body_region,
                "body_center": body_center,
            }
        )

    return observations


def point_to_segment_distance(point, start, end) -> float:
    """Return the shortest distance from a point to an arm segment."""
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    segment_length_squared = segment_x**2 + segment_y**2
    if segment_length_squared <= 1e-12:
        return point_distance(point, start)

    projection = (
        (point[0] - start[0]) * segment_x
        + (point[1] - start[1]) * segment_y
    ) / segment_length_squared
    projection = max(0.0, min(1.0, projection))
    closest_point = (
        start[0] + projection * segment_x,
        start[1] + projection * segment_y,
    )
    return point_distance(point, closest_point)


def associate_hands(results, signer, tracker, current_time) -> list:
    """Return only hands spatially consistent with the active signer."""
    if signer is None:
        return []

    pose_landmarks = signer["landmarks"]
    body_region = signer["body_region"]
    shoulder_width = signer["shoulder_width"]
    region_margin = max(0.10, shoulder_width * 0.75)
    expanded_region = (
        body_region[0] - region_margin,
        body_region[1] - region_margin,
        body_region[2] + region_margin,
        body_region[3] + region_margin,
    )
    acceptance_distance = max(0.09, shoulder_width * 0.75)
    candidates_by_side = {"Left": [], "Right": []}

    for hand_landmarks, handedness in zip(
        results.hand_landmarks,
        results.handedness,
    ):
        if len(hand_landmarks) < 21 or not handedness:
            continue

        hand_wrist = (hand_landmarks[0].x, hand_landmarks[0].y)
        inside_body_region = (
            expanded_region[0] <= hand_wrist[0] <= expanded_region[2]
            and expanded_region[1] <= hand_wrist[1] <= expanded_region[3]
        )
        media_pipe_side = handedness[0].category_name

        side_scores = []
        for side, shoulder_index, elbow_index in (
            ("Left", 12, 14),
            ("Right", 11, 13),
        ):
            shoulder = pose_landmarks[shoulder_index]
            elbow = pose_landmarks[elbow_index]
            wrist = pose_landmarks[POSE_WRIST_INDICES[side]]

            arm_distances = []
            if landmark_is_visible(shoulder) and landmark_is_visible(elbow):
                arm_distances.append(
                    point_to_segment_distance(
                        hand_wrist,
                        (shoulder.x, shoulder.y),
                        (elbow.x, elbow.y),
                    )
                )
            if landmark_is_visible(elbow) and landmark_is_visible(wrist):
                arm_distances.append(
                    point_to_segment_distance(
                        hand_wrist,
                        (elbow.x, elbow.y),
                        (wrist.x, wrist.y),
                    )
                )

            pose_wrist_distance = (
                point_distance(hand_wrist, (wrist.x, wrist.y))
                if landmark_is_visible(wrist)
                else float("inf")
            )
            previous_distance = float("inf")
            if (
                side in tracker.hand_positions
                and current_time - tracker.hand_seen_times[side]
                <= HAND_MEMORY_SECONDS
            ):
                previous_distance = point_distance(
                    hand_wrist,
                    tracker.hand_positions[side],
                )

            spatial_distance = min(
                [pose_wrist_distance, previous_distance, *arm_distances]
            )
            label_penalty = 0.04 if media_pipe_side != side else 0.0
            continuity_bonus = 0.5 * previous_distance
            score = pose_wrist_distance + label_penalty
            if math.isfinite(previous_distance):
                score = min(score, continuity_bonus + label_penalty)

            side_scores.append((score, spatial_distance, side, previous_distance))

        score, spatial_distance, side, previous_distance = min(
            side_scores,
            key=lambda item: item[0],
        )
        if spatial_distance > acceptance_distance:
            continue
        if not inside_body_region and previous_distance > acceptance_distance:
            continue

        candidates_by_side[side].append(
            {
                "score": score,
                "side": side,
                "landmarks": hand_landmarks,
                "handedness": handedness,
                "wrist": hand_wrist,
            }
        )

    accepted_hands = []
    for side in ("Left", "Right"):
        if candidates_by_side[side]:
            accepted_hands.append(
                min(candidates_by_side[side], key=lambda item: item["score"])
            )

    if tracker.locked:
        for hand in accepted_hands:
            tracker.hand_positions[hand["side"]] = hand["wrist"]
            tracker.hand_seen_times[hand["side"]] = current_time

    return accepted_hands


def draw_hand(frame, hand_landmarks, handedness, label=None) -> None:
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

    label = label or handedness[0].category_name
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


def draw_upper_body(frame, pose_landmarks, color=(0, 255, 255)) -> None:
    """Draw only visible shoulders, elbows and wrists."""
    height, width = frame.shape[:2]
    visible_points = {}

    for landmark_index in POSE_LANDMARK_INDICES:
        landmark = pose_landmarks[landmark_index]
        visibility = landmark.visibility or 0.0
        if visibility < 0.5:
            continue

        visible_points[landmark_index] = (
            int(landmark.x * width),
            int(landmark.y * height),
        )

    for start, end in POSE_CONNECTIONS:
        if start in visible_points and end in visible_points:
            cv2.line(
                frame,
                visible_points[start],
                visible_points[end],
                color,
                2,
            )

    for point in visible_points.values():
        cv2.circle(frame, point, 5, color, -1)


def draw_body_region(frame, body_region, color) -> None:
    """Draw the candidate or locked signer's approximate body region."""
    height, width = frame.shape[:2]
    top_left = (int(body_region[0] * width), int(body_region[1] * height))
    bottom_right = (int(body_region[2] * width), int(body_region[3] * height))
    cv2.rectangle(frame, top_left, bottom_right, color, 2)


def prepare_results_for_features(accepted_hands):
    """Adapt only accepted signer hands for the feature extractor."""
    hand_landmarks = [
        SimpleNamespace(landmark=hand["landmarks"])
        for hand in accepted_hands
    ]
    handedness = [
        SimpleNamespace(
            classification=[SimpleNamespace(label=hand["side"])]
        )
        for hand in accepted_hands
    ]
    return SimpleNamespace(
        multi_hand_landmarks=hand_landmarks,
        multi_handedness=handedness,
    )


def show_camera() -> None:
    """Open the default webcam and display its live video feed."""
    if not POSE_MODEL_PATH.is_file():
        print("Error: Pose Landmarker model file is missing.")
        print(f"Expected location: {POSE_MODEL_PATH}")
        return

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Hand Landmarker model not found: {MODEL_PATH}"
        )

    options = vision.HandLandmarkerOptions(
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

    try:
        with (
            vision.HandLandmarker.create_from_options(options) as landmarker,
            vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
        ):
            last_timestamp_ms = -1
            tracker = signer_session.tracker

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
                candidate = session_result.acquisition_candidate
                status_text = session_result.state.value.replace("_", " ").title()

                accepted_hands = associate_hands(
                    results,
                    active_signer,
                    tracker,
                    current_time,
                )
                adapted_results = prepare_results_for_features(accepted_hands)
                adapted_pose = (
                    SimpleNamespace(
                        pose_landmarks=SimpleNamespace(
                            landmark=active_signer["landmarks"]
                        )
                    )
                    if active_signer is not None
                    else None
                )
                features = extract_features(adapted_results, adapted_pose)

                display_signer = active_signer or candidate
                if display_signer is not None:
                    signer_color = (
                        (0, 255, 255)
                        if session_result.state is SessionState.LOCKED
                        else (0, 165, 255)
                    )
                    draw_upper_body(
                        frame,
                        display_signer["landmarks"],
                        signer_color,
                    )
                    body_region = (
                        tracker.body_region
                        if tracker.locked
                        else display_signer["body_region"]
                    )
                    draw_body_region(frame, body_region, signer_color)

                for hand in accepted_hands:
                    draw_hand(
                        frame,
                        hand["landmarks"],
                        hand["handedness"],
                        hand["side"],
                    )

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
                cv2.putText(
                    frame,
                    status_text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("SgSL Camera", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("p"), ord("P")):
                    print("Feature vector:", features)
                    print("Vector length:", len(features))

                if key in (ord("r"), ord("R")):
                    signer_session.reset()

                if key in (ord("q"), ord("Q")):
                    break
    finally:
        signer_session.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_camera()
