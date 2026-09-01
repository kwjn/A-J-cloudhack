"""Display a mirrored live feed from the default webcam."""

import cv2
import mediapipe as mp


def show_camera() -> None:
    """Open the default webcam and display its live video feed."""
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open the default webcam.")

    hands = mp.solutions.hands.Hands(max_num_hands=2)
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
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
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
            if key in (ord("q"), ord("Q")):
                break
    finally:
        # Always release the webcam and close the window cleanly.
        hands.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_camera()
