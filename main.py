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
from src.gaze_estimation import estimate_gaze, combine_gaze
from src.attention import AttentionTracker
from src.session_tracker import SessionTracker, format_time, get_engagement_label
from src.blink_detection import BlinkDetector
from src.heatmap import GazeHeatmap


def draw_pupil_on_frame(frame, pupil_center, eye_box):
    if pupil_center is None or eye_box is None:
        return

    eye_x, eye_y, _, _ = eye_box
    pupil_x, pupil_y = pupil_center

    global_x = eye_x + pupil_x
    global_y = eye_y + pupil_y

    cv2.circle(frame, (global_x, global_y), 5, (0, 0, 255), -1)


def draw_dashboard(
    frame,
    current_gaze,
    attention_state,
    attention_score,
    session_duration,
    attentive_time,
    distraction_count,
    engagement_label,
    blink_count,
    ear_value,
    recorded_points
):
    cv2.rectangle(frame, (10, 10), (590, 340), (30, 30, 30), -1)

    cv2.putText(
        frame,
        "Online Learning Attention Monitor",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Gaze Direction: {current_gaze}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Attention State: {attention_state}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if attention_state == "Attentive" else (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Attention Score: {attention_score}%",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Session Time: {format_time(session_duration)}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Attentive Time: {format_time(attentive_time)}",
        (20, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Distraction Count: {distraction_count}",
        (20, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Blink Count: {blink_count}",
        (20, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    if ear_value is not None:
        cv2.putText(
            frame,
            f"EAR: {ear_value:.2f}",
            (300, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv2.putText(
        frame,
        f"Gaze Points Recorded: {recorded_points}",
        (20, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Student Engagement: {engagement_label}",
        (20, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


def main():
    print("Starting Phase 10 - Heatmap Visualization...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam failed to open")
        return

    detector = FaceMeshDetector()
    attention_tracker = AttentionTracker(max_history=60)
    session_tracker = SessionTracker()
    blink_detector = BlinkDetector(threshold=0.21, consecutive_frames=2)
    gaze_heatmap = GazeHeatmap()

    current_gaze = "Unknown"
    attention_state = "Unknown"
    attention_score = 0
    blink_count = 0
    ear_value = None
    last_frame_shape = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            break

        last_frame_shape = frame.shape

        results = detector.process(frame)
        landmarks = detector.get_landmarks(frame, results)

        face_detected = len(landmarks) > 0
        both_eyes_detected = False

        if face_detected:
            left_eye_points = get_eye_points(landmarks, LEFT_EYE_IDX)
            right_eye_points = get_eye_points(landmarks, RIGHT_EYE_IDX)

            draw_eye_contour(frame, left_eye_points)
            draw_eye_contour(frame, right_eye_points)

            blink_count, ear_value = blink_detector.update(
                left_eye_points,
                right_eye_points
            )

            left_eye_crop, left_eye_box = extract_eye_region(frame, left_eye_points)
            right_eye_crop, right_eye_box = extract_eye_region(frame, right_eye_points)

            left_pupil, left_threshold = find_pupil_center(left_eye_crop)
            right_pupil, right_threshold = find_pupil_center(right_eye_crop)

            draw_pupil_on_frame(frame, left_pupil, left_eye_box)
            draw_pupil_on_frame(frame, right_pupil, right_eye_box)

            both_eyes_detected = (
                left_pupil is not None
                and right_pupil is not None
            )

            if left_eye_box is not None:
                x, y, w, h = left_eye_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

            if right_eye_box is not None:
                x, y, w, h = right_eye_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

            left_gaze = "Unknown"
            right_gaze = "Unknown"

            if left_eye_crop is not None and left_eye_crop.size > 0:
                _, left_eye_width = left_eye_crop.shape[:2]
                left_gaze = estimate_gaze(left_pupil, left_eye_width)

            if right_eye_crop is not None and right_eye_crop.size > 0:
                _, right_eye_width = right_eye_crop.shape[:2]
                right_gaze = estimate_gaze(right_pupil, right_eye_width)

            current_gaze = combine_gaze(left_gaze, right_gaze)

            gaze_point = gaze_heatmap.estimate_point_from_gaze(
                current_gaze,
                frame.shape
            )

            gaze_heatmap.update_gaze_points(gaze_point)

            if gaze_point is not None:
                cv2.circle(frame, gaze_point, 8, (0, 255, 255), -1)

            # Optional debug windows
            if left_eye_crop is not None and left_eye_crop.size > 0:
                cv2.imshow("Left Eye Crop", left_eye_crop)

            if right_eye_crop is not None and right_eye_crop.size > 0:
                cv2.imshow("Right Eye Crop", right_eye_crop)

            if left_threshold is not None:
                cv2.imshow("Left Eye Threshold", left_threshold)

            if right_threshold is not None:
                cv2.imshow("Right Eye Threshold", right_threshold)

        else:
            current_gaze = "No Face Detected"
            ear_value = None

        attention_state = attention_tracker.update_attention_state(
            gaze_label=current_gaze,
            face_detected=face_detected,
            both_eyes_detected=both_eyes_detected
        )

        attention_score = attention_tracker.compute_attention_score()

        session_tracker.update(attention_state)

        session_duration = session_tracker.get_session_duration()
        attentive_time = session_tracker.get_attentive_time()
        distraction_count = session_tracker.get_distraction_count()
        engagement_label = get_engagement_label(attention_score)

        draw_dashboard(
            frame=frame,
            current_gaze=current_gaze,
            attention_state=attention_state,
            attention_score=attention_score,
            session_duration=session_duration,
            attentive_time=attentive_time,
            distraction_count=distraction_count,
            engagement_label=engagement_label,
            blink_count=blink_count,
            ear_value=ear_value,
            recorded_points=len(gaze_heatmap.points)
        )

        cv2.imshow("Phase 10 - Heatmap Visualization", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if last_frame_shape is not None:
        gaze_heatmap.generate_heatmap(last_frame_shape)


if __name__ == "__main__":
    main()