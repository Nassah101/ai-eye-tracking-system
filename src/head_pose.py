import cv2
import numpy as np


class HeadPoseEstimator:
    def __init__(self):
        """
        3D model points of a generic human face.

        These are approximate reference points used by solvePnP.
        They are enough for a prototype head pose estimation system.
        """

        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye outer corner
            (225.0, 170.0, -135.0),      # Right eye outer corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

    def get_image_points(self, landmarks):
        """
        Extract selected 2D landmark points from MediaPipe landmarks.

        MediaPipe landmark indices:
        1   = nose tip
        152 = chin
        33  = left eye outer corner
        263 = right eye outer corner
        61  = left mouth corner
        291 = right mouth corner
        """

        if landmarks is None or len(landmarks) < 292:
            return None

        image_points = np.array([
            landmarks[1],     # Nose tip
            landmarks[152],   # Chin
            landmarks[33],    # Left eye outer corner
            landmarks[263],   # Right eye outer corner
            landmarks[61],    # Left mouth corner
            landmarks[291]    # Right mouth corner
        ], dtype=np.float64)

        return image_points

    def estimate_head_pose(self, frame, landmarks):
        """
        Estimate head pose using OpenCV solvePnP.

        Returns:
        - head_direction: text label
        - nose_line: two points for drawing nose direction line
        - pose_angles: pitch, yaw, roll
        """

        image_points = self.get_image_points(landmarks)

        if image_points is None:
            return "Head Unknown", None, None

        height, width = frame.shape[:2]

        focal_length = width
        center = (width / 2, height / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return "Head Unknown", None, None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        projection_matrix = np.hstack((rotation_matrix, translation_vector))

        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            projection_matrix
        )

        # euler_angles is a 3x1 NumPy array, so we extract actual scalar values
        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        head_direction = self.classify_head_direction(pitch, yaw)

        nose_3d_projection = np.array([
            (0.0, 0.0, 1000.0)
        ], dtype=np.float64)

        nose_2d_projection, _ = cv2.projectPoints(
            nose_3d_projection,
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs
        )

        nose_tip = tuple(image_points[0].astype(int))
        projected_nose = tuple(nose_2d_projection[0][0].astype(int))

        return head_direction, (nose_tip, projected_nose), (pitch, yaw, roll)

    def classify_head_direction(self, pitch, yaw):
        """
        Convert pitch and yaw values into readable head direction.

        yaw controls left/right movement.
        pitch controls up/down movement.
        """

        yaw_threshold = 15
        pitch_threshold = 15

        if yaw > yaw_threshold:
            return "Head Left"

        elif yaw < -yaw_threshold:
            return "Head Right"

        elif pitch > pitch_threshold:
            return "Head Down"

        elif pitch < -pitch_threshold:
            return "Head Up"

        else:
            return "Head Forward"