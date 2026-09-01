"""Convert MediaPipe hand and upper-body landmarks into model-ready features."""

import numpy as np


LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
HAND_ORDER = {"Left": 0, "Right": 1}
PALM_MCP_INDICES = (5, 9, 17)
MIN_HAND_SIZE = 1e-6

# MediaPipe Pose indices, kept in the required output order.
BODY_LANDMARK_INDICES = (11, 12, 13, 14, 15, 16)
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
POSE_VISIBILITY_THRESHOLD = 0.5
MIN_SHOULDER_WIDTH = 1e-6


def extract_hand_features(results) -> np.ndarray:
    """Return a 126-value vector ordered as left hand, then right hand."""
    features = np.zeros(
        (2, LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK),
        dtype=np.float32,
    )

    hand_landmarks_list = results.multi_hand_landmarks or []
    handedness_list = results.multi_handedness or []

    for hand_landmarks, handedness in zip(
        hand_landmarks_list,
        handedness_list,
    ):
        label = handedness.classification[0].label
        hand_index = HAND_ORDER.get(label)

        if hand_index is None:
            continue

        coordinates = np.array(
            [
                (landmark.x, landmark.y, landmark.z)
                for landmark in hand_landmarks.landmark[:LANDMARKS_PER_HAND]
            ],
            dtype=np.float32,
        )

        # Move the wrist (landmark 0) to (0, 0, 0). This removes the hand's
        # position in the camera frame while preserving its shape and pose.
        wrist = coordinates[0]
        wrist_relative = coordinates - wrist

        # Use the average 3D distance from the wrist to the index, middle and
        # little-finger MCP knuckles as the hand-size measurement. Averaging
        # these stable palm points is less sensitive to one noisy landmark.
        # Dividing by this value makes different hand sizes and camera
        # distances produce more comparable coordinates.
        palm_distances = np.linalg.norm(
            wrist_relative[list(PALM_MCP_INDICES)],
            axis=1,
        )
        hand_size = float(np.mean(palm_distances))

        # A real detected hand should have a positive size. Keep the centered
        # coordinates unchanged if the landmarks collapse to avoid dividing
        # by zero or amplifying numerical noise.
        if hand_size > MIN_HAND_SIZE:
            wrist_relative /= hand_size

        features[hand_index] = wrist_relative

    return features.flatten()


def extract_body_features(results) -> np.ndarray:
    """Return 18 normalized values for the six selected body landmarks."""
    features = np.zeros(
        (len(BODY_LANDMARK_INDICES), COORDINATES_PER_LANDMARK),
        dtype=np.float32,
    )

    pose_landmarks = getattr(results, "pose_landmarks", None)
    if pose_landmarks is None:
        return features.flatten()

    landmarks = pose_landmarks.landmark
    if len(landmarks) <= max(BODY_LANDMARK_INDICES):
        return features.flatten()

    left_shoulder = landmarks[LEFT_SHOULDER_INDEX]
    right_shoulder = landmarks[RIGHT_SHOULDER_INDEX]

    # Both shoulders are needed to define a reliable body origin and scale.
    if (
        left_shoulder.visibility < POSE_VISIBILITY_THRESHOLD
        or right_shoulder.visibility < POSE_VISIBILITY_THRESHOLD
    ):
        return features.flatten()

    left_shoulder_coordinates = np.array(
        (left_shoulder.x, left_shoulder.y, left_shoulder.z),
        dtype=np.float32,
    )
    right_shoulder_coordinates = np.array(
        (right_shoulder.x, right_shoulder.y, right_shoulder.z),
        dtype=np.float32,
    )

    # The midpoint between the shoulders becomes (0, 0, 0), removing the
    # body's position in the frame. Shoulder width is the 3D distance between
    # the shoulders; dividing by it reduces differences caused by body size
    # and distance from the camera.
    shoulder_midpoint = (
        left_shoulder_coordinates + right_shoulder_coordinates
    ) / 2.0
    shoulder_width = float(
        np.linalg.norm(left_shoulder_coordinates - right_shoulder_coordinates)
    )

    if shoulder_width <= MIN_SHOULDER_WIDTH:
        return features.flatten()

    for output_index, landmark_index in enumerate(BODY_LANDMARK_INDICES):
        landmark = landmarks[landmark_index]
        if landmark.visibility < POSE_VISIBILITY_THRESHOLD:
            continue

        coordinates = np.array(
            (landmark.x, landmark.y, landmark.z),
            dtype=np.float32,
        )
        features[output_index] = (
            coordinates - shoulder_midpoint
        ) / shoulder_width

    return features.flatten()


def extract_features(hand_results, pose_results) -> np.ndarray:
    """Return the fixed 144-value hand and upper-body feature vector."""
    return np.concatenate(
        (
            extract_hand_features(hand_results),
            extract_body_features(pose_results),
        )
    )
