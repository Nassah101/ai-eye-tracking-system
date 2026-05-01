import cv2
import mediapipe as mp


class FaceMeshDetector:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )

        self.detector = FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self.timestamp_ms += 33
        results = self.detector.detect_for_video(mp_image, self.timestamp_ms)

        return results

    def get_landmarks(self, frame, results):
        height, width, _ = frame.shape
        points = []

        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]

            for landmark in face_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))

        return points