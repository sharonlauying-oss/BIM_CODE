# gestures/base_gesture.py

class BaseGesture:
    def __init__(self, name="Generic Gesture"):
        self.name = name
        self.stroke_count = 0  # Standardize dynamic gesture counters
        self.is_double_hand = False  # By default, all recognition is done with one hand (right hand).

    def reset_dynamic_tracking(self):
        """Resets any dynamic states when changing steps."""
        self.stroke_count = 0

    def check_gesture(self, hands_list: list, current_step: int):
        """
        Must match the signature called by vision_engine.py
        
        :param hands_list: List of filtered hand dicts
        :param current_step: Current lesson step (1 or 2)
        """
        raise NotImplementedError("Subclasses must implement the check_gesture method.")