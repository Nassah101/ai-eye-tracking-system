import cv2
from src.face_mesh import FaceMeshDetector
from src.eye_region import (
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    get_eye_points,
    draw_eye_contour,
    extract_eye_region
)
from src.pupil_tracking import find_pupil_center


def draw_pupil_on_frame(frame, pupil_center, eye_box):
    if pupil_center is None or eye_box is None:
        return

    eye_x, eye_y, _, _ = eye_box
    pupil_x, pupil_y = pupil_center

    global_x = eye_x + pupil_x
    global_y = eye_y + pupil_y

    cv2.circle(frame, (global_x, global_y), 5, (0, 0, 255), -1)


def main():
    print("Starting Phase 4 - Eye Cropping and Pupil Tracking...")

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

        if len(landmarks) > 0:
            left_eye_points = get_eye_points(landmarks, LEFT_EYE_IDX)
            right_eye_points = get_eye_points(landmarks, RIGHT_EYE_IDX)

            draw_eye_contour(frame, left_eye_points)
            draw_eye_contour(frame, right_eye_points)

            left_eye_crop, left_eye_box = extract_eye_region(frame, left_eye_points)
            right_eye_crop, right_eye_box = extract_eye_region(frame, right_eye_points)

            left_pupil, left_threshold = find_pupil_center(left_eye_crop)
            right_pupil, right_threshold = find_pupil_center(right_eye_crop)

            draw_pupil_on_frame(frame, left_pupil, left_eye_box)
            draw_pupil_on_frame(frame, right_pupil, right_eye_box)

            if left_eye_box is not None:
                x, y, w, h = left_eye_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

            if right_eye_box is not None:
                x, y, w, h = right_eye_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

            if left_eye_crop is not None and left_eye_crop.size > 0:
                cv2.imshow("Left Eye Crop", left_eye_crop)

            if right_eye_crop is not None and right_eye_crop.size > 0:
                cv2.imshow("Right Eye Crop", right_eye_crop)

            if left_threshold is not None:
                cv2.imshow("Left Eye Threshold", left_threshold)

            if right_threshold is not None:
                cv2.imshow("Right Eye Threshold", right_threshold)

        cv2.putText(
            frame,
            f"Landmarks: {len(landmarks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Phase 4 - Pupil Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()