def estimate_gaze(pupil_center, eye_width):
    """
    Estimates gaze direction based on pupil x-position inside the eye crop.

    Args:
        pupil_center: tuple (x, y) for pupil position inside cropped eye image
        eye_width: width of cropped eye image

    Returns:
        "Looking Left", "Looking Center", "Looking Right", or "Unknown"
    """

    if pupil_center is None or eye_width is None or eye_width == 0:
        return "Unknown"

    pupil_x, _ = pupil_center

    left_boundary = eye_width / 3
    right_boundary = (eye_width / 3) * 2

    if pupil_x < left_boundary:
        return "Looking Left"
    elif pupil_x > right_boundary:
        return "Looking Right"
    else:
        return "Looking Center"


def combine_gaze(left_gaze, right_gaze):
    """
    Combines left and right eye gaze predictions for more stable output.
    """

    if left_gaze == right_gaze:
        return left_gaze

    if left_gaze == "Unknown" and right_gaze != "Unknown":
        return right_gaze

    if right_gaze == "Unknown" and left_gaze != "Unknown":
        return left_gaze

    return "Looking Center"