import cv2
import numpy as np


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


def extract_eye_region(frame, eye_points, padding=8):
    """
    Crops the eye region from the full frame using eye landmark points.
    Returns:
    - eye_crop
    - bounding box: x, y, w, h
    """

    if len(eye_points) == 0:
        return None, None

    points = np.array(eye_points)

    x, y, w, h = cv2.boundingRect(points)

    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = w + (padding * 2)
    h = h + (padding * 2)

    frame_h, frame_w = frame.shape[:2]

    if x + w > frame_w:
        w = frame_w - x

    if y + h > frame_h:
        h = frame_h - y

    eye_crop = frame[y:y+h, x:x+w]

    return eye_crop, (x, y, w, h)