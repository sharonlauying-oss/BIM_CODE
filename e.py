# gestures/e.py - "E" Hand Gesture Recognition
# Correct Pose: Fingers folded, thumb slightly bent and close to index, palm facing forward
import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("E")
        self.consecutive_correct = 0
        self.required_consecutive = 2

        # Gesture calibration thresholds
        self.THRESHOLDS = {
            "min_palm_height": 0.06,
            "finger_angle_max": 120,
            "thumb_close_max": 0.10,
            "thumb_angle_min": 90,
            "thumb_angle_max": 165,
            "palm_ratio_min": 0.30
        }

    def _get_distance(self, p1, p2):
        # Calculate 2D Euclidean distance between two points
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _get_world_distance(self, p1, p2):
        # Calculate 3D physical distance in meters
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    def _get_angle(self, p1, p2, p3):
        # Calculate joint angle in degrees using three landmarks
        v1x, v1y = p1.x - p2.x, p1.y - p2.y
        v2x, v2y = p3.x - p2.x, p3.y - p2.y

        dot_product = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        cos_angle = max(min(dot_product / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_angle))

    def reset_dynamic_tracking(self):
        # Reset consecutive valid frame counter
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        # Return false if no hand detected
        if not hands_list:
            self.consecutive_correct = 0
            return False if current_step == 1 else 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # Validate palm size
        palm_height = self._get_world_distance(wm[0], wm[9])
        palm_valid = palm_height >= self.THRESHOLDS["min_palm_height"]

        # Validate all four fingers are folded (tip below PIP joint)
        idx_folded = (lm[8].y > lm[6].y) and (self._get_angle(lm[5], lm[6], lm[8]) < self.THRESHOLDS["finger_angle_max"])
        mid_folded = (lm[12].y > lm[10].y) and (self._get_angle(lm[9], lm[10], lm[12]) < self.THRESHOLDS["finger_angle_max"])
        ring_folded = (lm[16].y > lm[14].y) and (self._get_angle(lm[13], lm[14], lm[16]) < self.THRESHOLDS["finger_angle_max"])
        pinky_folded = (lm[20].y > lm[18].y) and (self._get_angle(lm[17], lm[18], lm[20]) < self.THRESHOLDS["finger_angle_max"])
        fingers_valid = idx_folded and mid_folded and ring_folded and pinky_folded

        # Validate thumb: close to index and properly bent
        thumb_distance = self._get_world_distance(wm[4], wm[8])
        thumb_close = thumb_distance < self.THRESHOLDS["thumb_close_max"]
        
        thumb_angle = self._get_angle(lm[1], lm[2], lm[3])
        thumb_bent = self.THRESHOLDS["thumb_angle_min"] < thumb_angle < self.THRESHOLDS["thumb_angle_max"]
        thumb_valid = thumb_close and thumb_bent

        # Validate palm orientation (forward-facing)
        palm_width = self._get_distance(lm[5], lm[17])
        palm_h2d = self._get_distance(lm[0], lm[9])
        palm_ratio = palm_width / palm_h2d if palm_h2d > 0 else 0
        palm_forward = palm_ratio >= self.THRESHOLDS["palm_ratio_min"]

        # Final gesture validation
        gesture_valid = palm_valid and fingers_valid and thumb_valid and palm_forward

        # Update consecutive valid frames
        if gesture_valid:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0

        result = self.consecutive_correct >= self.required_consecutive
        return result if current_step == 1 else self.stroke_count