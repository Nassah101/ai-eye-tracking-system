import cv2


def preprocess_eye(eye_crop, threshold_value=45):
    """
    Preprocesses the cropped eye image before pupil detection.

    Steps:
    1. Convert to grayscale
    2. Apply Gaussian blur
    3. Apply inverse threshold to highlight dark pupil regions
    """

    if eye_crop is None or eye_crop.size == 0:
        return None

    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, threshold = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV
    )

    return threshold


def find_pupil_center(eye_crop, threshold_value=45):
    """
    Finds the pupil center inside a cropped eye image.
    """

    threshold = preprocess_eye(eye_crop, threshold_value)

    if threshold is None:
        return None, None

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