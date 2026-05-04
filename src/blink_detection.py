import math


def calculate_ear(eye_points):
    """
    Calculates Eye Aspect Ratio (EAR).

    eye_points should contain 6 landmark points in this order:
    [outer_corner, upper_outer, upper_inner, inner_corner, lower_inner, lower_outer]

    EAR formula:
    EAR = (vertical_distance_1 + vertical_distance_2) / (2 * horizontal_distance)
    """

    if eye_points is None or len(eye_points) < 6:
        return None

    p1 = eye_points[0]  # outer eye corner
    p2 = eye_points[1]  # upper outer
    p3 = eye_points[2]  # upper inner
    p4 = eye_points[3]  # inner eye corner
    p5 = eye_points[4]  # lower inner
    p6 = eye_points[5]  # lower outer

    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p4)

    if horizontal == 0:
        return None

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


def detect_blink(ear, threshold=0.21):
    """
    Returns True if EAR is below threshold.
    This means the eye is likely closed.
    """

    if ear is None:
        return False

    return ear < threshold


class BlinkDetector:
    def __init__(self, threshold=0.21, consecutive_frames=2):
        """
        threshold:
            EAR value below which the eye is considered closed.

        consecutive_frames:
            Number of frames the eye must remain closed before counting a blink.
        """

        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.closed_frame_count = 0
        self.blink_count = 0
        self.eye_was_closed = False

    def update(self, left_eye_points, right_eye_points):
        """
        Updates blink count using both eyes.
        """

        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)

        if left_ear is None and right_ear is None:
            return self.blink_count, None

        if left_ear is not None and right_ear is not None:
            average_ear = (left_ear + right_ear) / 2.0
        elif left_ear is not None:
            average_ear = left_ear
        else:
            average_ear = right_ear

        eye_is_closed = detect_blink(average_ear, self.threshold)

        if eye_is_closed:
            self.closed_frame_count += 1
            self.eye_was_closed = True
        else:
            if self.eye_was_closed and self.closed_frame_count >= self.consecutive_frames:
                self.blink_count += 1

            self.closed_frame_count = 0
            self.eye_was_closed = False

        return self.blink_count, average_ear