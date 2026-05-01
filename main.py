import cv2
from src.face_mesh import FaceMeshDetector


def main():
    print("Starting Phase 2 - Face Landmark Detection...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam failed to open")
        return

    detector = FaceMeshDetector()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            break

        results = detector.process(frame)
        landmarks = detector.get_landmarks(frame, results)

        print("Landmarks:", len(landmarks))

        for x, y in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.putText(
            frame,
            f"Landmarks detected: {len(landmarks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Phase 2 - Face Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()