import time


class SessionTracker:
    def __init__(self):
        self.start_time = time.time()
        self.attentive_seconds = 0
        self.last_update_time = time.time()
        self.distraction_count = 0
        self.previous_state = None

    def update(self, attention_state):
        """
        Updates session time, attentive time, and distraction count.
        """

        current_time = time.time()
        elapsed_since_last_update = current_time - self.last_update_time
        self.last_update_time = current_time

        if attention_state == "Attentive":
            self.attentive_seconds += elapsed_since_last_update

        if self.previous_state == "Attentive" and attention_state == "Distracted":
            self.distraction_count += 1

        self.previous_state = attention_state

    def get_session_duration(self):
        """
        Returns total session duration in seconds.
        """
        return time.time() - self.start_time

    def get_attentive_time(self):
        """
        Returns total attentive time in seconds.
        """
        return self.attentive_seconds

    def get_distraction_count(self):
        """
        Returns how many times user became distracted.
        """
        return self.distraction_count


def format_time(seconds):
    """
    Converts seconds into MM:SS format.
    """

    seconds = int(seconds)
    minutes = seconds // 60
    remaining_seconds = seconds % 60

    return f"{minutes:02d}:{remaining_seconds:02d}"


def get_engagement_label(attention_score):
    """
    Converts attention score into a business-friendly engagement label.
    """

    if attention_score >= 75:
        return "Highly Engaged"
    elif attention_score >= 50:
        return "Moderately Engaged"
    else:
        return "Low Engagement"