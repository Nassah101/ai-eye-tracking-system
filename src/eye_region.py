import cv2


LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def get_eye_points(landmarks, eye_indices):
    return [landmarks[i] for i in eye_indices if i < len(landmarks)]


def draw_eye_contour(frame, eye_points, color=(255, 0, 0)):
    if len(eye_points) < 2:
        return

    for i in range(len(eye_points)):
        pt1 = eye_points[i]
        pt2 = eye_points[(i + 1) % len(eye_points)]
        cv2.line(frame, pt1, pt2, color, 2)

    for point in eye_points:
        cv2.circle(frame, point, 2, (0, 255, 255), -1)