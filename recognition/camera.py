"""Display a mirrored live feed from the default webcam."""

import cv2
import mediapipe as mp

from landmarks import extract_features

POSE_LANDMARKS = (
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value,
)

POSE_CONNECTIONS = (
    (POSE_LANDMARKS[0], POSE_LANDMARKS[1]),
    (POSE_LANDMARKS[0], POSE_LANDMARKS[2]),
    (POSE_LANDMARKS[2], POSE_LANDMARKS[4]),
    (POSE_LANDMARKS[1], POSE_LANDMARKS[3]),
    (POSE_LANDMARKS[3], POSE_LANDMARKS[5]),
)


def draw_upper_body_pose(frame, pose_landmarks) -> None:
    """Draw only the selected shoulders, elbows and wrists."""
    height, width = frame.shape[:2]
    visible_points = {}

    for landmark_index in POSE_LANDMARKS:
        landmark = pose_landmarks.landmark[landmark_index]
        if landmark.visibility < 0.5:
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
                (0, 255, 255),
                2,
            )

    for point in visible_points.values():
        cv2.circle(frame, point, 5, (0, 255, 255), -1)


def show_camera() -> None:
    """Open the default webcam and display its live video feed."""
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open the default webcam.")

    hands = mp.solutions.hands.Hands(max_num_hands=2)
    pose = mp.solutions.pose.Pose()
    drawing = mp.solutions.drawing_utils

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            # Mirror the frame so it behaves like a selfie camera.
            frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB images, while OpenCV uses BGR.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb_frame)
            pose_results = pose.process(rgb_frame)

# This fixed-length vector will later be passed to a classifier.
            features = extract_features(hand_results, pose_results)

            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                ):
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

                    label = handedness.classification[0].label
                    wrist = hand_landmarks.landmark[0]
                    label_position = (
                        max(0, int(wrist.x * frame.shape[1]) - 20),
                        max(30, int(wrist.y * frame.shape[0]) - 10),
                    )
                    cv2.putText(
                        frame,
                        label,
                        label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if pose_results.pose_landmarks:
                draw_upper_body_pose(frame, pose_results.pose_landmarks)

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

            # waitKey returns the key pressed while the video window is active.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("p"), ord("P")):
                print("Feature vector:", features)
                print("Vector length:", len(features))

            if key in (ord("q"), ord("Q")):
                break
    finally:
        # Always release the webcam and close the window cleanly.
        hands.close()
        pose.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_camera()