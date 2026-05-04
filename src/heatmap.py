import os
import cv2
import numpy as np
from datetime import datetime


class GazeHeatmap:
    def __init__(self):
        self.points = []

    def update_gaze_points(self, point):
        """
        Stores gaze point.
        point should be a tuple like (x, y).
        """

        if point is not None:
            self.points.append(point)

    def estimate_point_from_gaze(self, gaze_label, frame_shape):
        """
        Converts gaze label into an approximate point on the frame.

        This is not full calibrated screen gaze tracking.
        It is an approximate visualization of attention direction.
        """

        height, width = frame_shape[:2]

        center_y = height // 2

        if gaze_label == "Looking Left":
            return (width // 4, center_y)

        elif gaze_label == "Looking Center":
            return (width // 2, center_y)

        elif gaze_label == "Looking Right":
            return ((width * 3) // 4, center_y)

        else:
            return None

    def generate_heatmap(self, frame_shape, output_dir="outputs/heatmaps"):
        """
        Generates and saves heatmap image.
        """

        if len(self.points) == 0:
            print("No gaze points recorded. Heatmap not generated.")
            return None

        os.makedirs(output_dir, exist_ok=True)

        height, width = frame_shape[:2]

        heatmap = np.zeros((height, width), dtype=np.float32)

        for x, y in self.points:
            if 0 <= x < width and 0 <= y < height:
                heatmap[y, x] += 1

        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=35, sigmaY=35)

        heatmap = cv2.normalize(
            heatmap,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        heatmap = heatmap.astype(np.uint8)

        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"gaze_heatmap_{timestamp}.png")

        cv2.imwrite(output_path, colored_heatmap)

        print(f"Heatmap saved to: {output_path}")

        return output_path