# gestures/c.py - Final Clean Version for "C" Hand Gesture
import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("C")
        self.consecutive_correct = 0
        self.required_consecutive = 2

        # Calibration thresholds tuned for standard C-shape gesture
        self.THRESHOLDS = {
            "min_palm_height": 0.08,           # Minimum valid palm size
            "finger_angle_min": 110,            # Min angle for slightly curved fingers
            "finger_angle_max": 180,            # Max angle for slightly curved fingers
            "thumb_straight_angle": 130,        # Thumb must be relatively straight
            "c_opening_min": 0.03,              # Min gap between thumb and index finger
            "c_opening_max": 0.08,              # Max gap between thumb and index finger
            "palm_ratio_max": 0.40              # Max aspect ratio for side‑facing palm
        }

    def _get_distance(self, p1, p2):
        # Calculate 2D Euclidean distance
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _get_world_distance(self, p1, p2):
        # Calculate 3D physical distance in meters
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    def _get_angle(self, p1, p2, p3):
        # Calculate joint angle in degrees
        v1x = p1.x - p2.x
        v1y = p1.y - p2.y
        v2x = p3.x - p2.x
        v2y = p3.y - p2.y

        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        cos_angle = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_angle))

    def reset_dynamic_tracking(self):
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        if not hands_list:
            self.consecutive_correct = 0
            return False if current_step == 1 else 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # Check palm size
        palm_h = self._get_world_distance(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # Check all four fingers are slightly curved
        idx_angle = self._get_angle(lm[5], lm[6], lm[8])
        mid_angle = self._get_angle(lm[9], lm[10], lm[12])
        ring_angle = self._get_angle(lm[13], lm[14], lm[16])
        pinky_angle = self._get_angle(lm[17], lm[18], lm[20])

        fingers_valid = (
            self.THRESHOLDS["finger_angle_min"] < idx_angle < self.THRESHOLDS["finger_angle_max"]
            and self.THRESHOLDS["finger_angle_min"] < mid_angle < self.THRESHOLDS["finger_angle_max"]
            and self.THRESHOLDS["finger_angle_min"] < ring_angle < self.THRESHOLDS["finger_angle_max"]
            and self.THRESHOLDS["finger_angle_min"] < pinky_angle < self.THRESHOLDS["finger_angle_max"]
        )

        # Check thumb is straight
        thumb_angle = self._get_angle(lm[2], lm[3], lm[4])
        thumb_valid = thumb_angle > self.THRESHOLDS["thumb_straight_angle"]

        # Check C‑shape gap between thumb and index finger
        c_gap = self._get_world_distance(wm[4], wm[8])
        gap_valid = self.THRESHOLDS["c_opening_min"] < c_gap < self.THRESHOLDS["c_opening_max"]

        # Check palm orientation (side facing)
        palm_w = self._get_distance(lm[5], lm[17])
        palm_h_2d = self._get_distance(lm[0], lm[9])
        palm_ratio = palm_w / palm_h_2d if palm_h_2d > 0 else 999
        palm_orient_valid = palm_ratio < self.THRESHOLDS["palm_ratio_max"]

        # Final gesture validation
        gesture_valid = (
            palm_valid
            and fingers_valid
            and thumb_valid
            and gap_valid
            and palm_orient_valid
        )

        # Update consecutive valid frames
        if gesture_valid:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0

        result = self.consecutive_correct >= self.required_consecutive
        return result if current_step == 1 else self.stroke_count