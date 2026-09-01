"""MediaPipe hand and body landmark extraction utilities."""
"""Convert MediaPipe hand landmarks into model-ready features."""

import numpy as np


LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
HAND_ORDER = {"Left": 0, "Right": 1}
PALM_MCP_INDICES = (5, 9, 17)
MIN_HAND_SIZE = 1e-6


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