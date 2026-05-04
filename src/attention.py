from collections import deque


class AttentionTracker:
    def __init__(self, max_history=60):
        """
        max_history controls how many recent frames we use
        to calculate the attention percentage.

        Example:
        60 frames is roughly about 2 seconds if webcam runs near 30 FPS.
        """
        self.history = deque(maxlen=max_history)

    def update_attention_state(self, gaze_label, face_detected, both_eyes_detected=True):
        """
        Determines if the user is attentive.

        Attentive condition:
        - face is detected
        - both eyes are detected
        - gaze is looking center
        """

        is_attentive = (
            face_detected
            and both_eyes_detected
            and gaze_label == "Looking Center"
        )

        self.history.append(is_attentive)

        if is_attentive:
            return "Attentive"
        else:
            return "Distracted"

    def compute_attention_score(self):
        """
        Computes attention percentage from recent frame history.
        """

        if len(self.history) == 0:
            return 0

        attentive_count = sum(self.history)
        total_count = len(self.history)

        score = (attentive_count / total_count) * 100

        return int(score)