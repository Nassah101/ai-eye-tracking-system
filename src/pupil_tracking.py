import cv2
import numpy as np


def find_pupil_center(eye_crop):
    """
    Finds the pupil center inside a cropped eye image.
    This uses a simple dark-region detection approach.
    """

    if eye_crop is None or eye_crop.size == 0:
        return None, None

    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, threshold = cv2.threshold(
        gray,
        45,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, threshold

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 5:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        return (cx, cy), threshold

    return None, threshold