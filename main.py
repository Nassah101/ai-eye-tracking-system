import cv2
from src.face_mesh import FaceMeshDetector
from src.eye_region import (
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    get_eye_points,
    draw_eye_contour
)


def main():
    print("Starting Phase 3 - Eye Region Extraction...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam failed to open")
        return

    detector = FaceMeshDetector()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to read frame")
            break

        # ---- Phase 2: Face Mesh ----
        results = detector.process(frame)
        landmarks = detector.get_landmarks(frame, results)

        # ---- Draw all landmarks (optional but useful) ----
        for x, y in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # ---- Phase 3: Eye Extraction ----
        if len(landmarks) > 0:
            left_eye_points = get_eye_points(landmarks, LEFT_EYE_IDX)
            right_eye_points = get_eye_points(landmarks, RIGHT_EYE_IDX)

            draw_eye_contour(frame, left_eye_points)
            draw_eye_contour(frame, right_eye_points)

        # ---- Display info ----
        cv2.putText(
            frame,
            f"Landmarks: {len(landmarks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Phase 3 - Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()