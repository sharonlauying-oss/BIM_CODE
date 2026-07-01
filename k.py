# gestures/k.py - K Gesture (Distance Adaptive Version)
# Fix: Works at any distance, all distance thresholds are relative to hand size
import math
from gestures.base_gesture import BaseGesture


class GestureChecker(BaseGesture):
    def __init__(self):
        # Initialize gesture name and anti-shake frame counter
        super().__init__("K")
        self.consecutive_correct = 0
        self.required_consecutive = 2

        # ALL DISTANCE THRESHOLDS ARE NOW RELATIVE TO HAND SIZE (0.0-1.0)
        self.T = {
            "min_palm_h": 0.06,            # Keep absolute for minimum detection range
            "palm_ratio_min": 0.25,        # Ratio is already relative
            "palm_ratio_max": 0.60,        # Ratio is already relative
            "straight_min": 135,           # Angle is always relative
            "bent_max": 100,               # Angle is always relative
            
            # NEW: Relative thresholds (distance / hand_size)
            "thumb_to_mid_ratio": 0.6,     # Thumb to middle distance / hand size
            "index_up_ratio": 0.8          # Index tip height / hand size
        }

    def _dist(self, p1, p2):
        # Calculate 2D Euclidean distance between two points
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _wdist(self, p1, p2):
        # Calculate 3D world coordinate distance between two points
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z))

    def _angle(self, p1, p2, p3):
        # Calculate joint angle at point p2 formed by p1-p2-p3
        v1x, v1y = p1.x - p2.x, p1.y - p2.y
        v2x, v2y = p3.x - p2.x, p3.y - p2.y
        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        cos_a = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_a))

    def reset_dynamic_tracking(self):
        # Reset consecutive valid frame counter
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        # Return immediately if no hand is detected
        if not hands_list:
            self.consecutive_correct = 0
            return False if current_step == 1 else 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # Calculate base hand size (wrist to middle MCP) - most stable reference
        hand_size = self._dist(lm[0], lm[9])
        # Prevent division by zero
        if hand_size < 0.01:
            hand_size = 0.1

        # 1. Validate palm size
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.T["min_palm_h"]

        # 2. Check side palm orientation: thumb on left, pinky on right
        thumb_mcp_x = lm[1].x
        pinky_mcp_x = lm[17].x
        thumb_on_left = thumb_mcp_x < pinky_mcp_x

        palm_w = self._dist(lm[5], lm[17])
        palm_h_2d = self._dist(lm[0], lm[9])
        palm_ratio = palm_w / palm_h_2d if palm_h_2d > 0 else 0
        palm_ratio_ok = self.T["palm_ratio_min"] < palm_ratio < self.T["palm_ratio_max"]
        palm_side_ok = thumb_on_left and palm_ratio_ok

        # 3. Calculate all finger joint angles
        idx_angle = self._angle(lm[5], lm[6], lm[8])
        mid_angle = self._angle(lm[9], lm[10], lm[12])
        ring_angle = self._angle(lm[13], lm[14], lm[16])
        pinky_angle = self._angle(lm[17], lm[18], lm[20])

        # 4. Validate finger states
        index_straight = idx_angle > self.T["straight_min"]
        middle_straight = mid_angle > self.T["straight_min"]
        ring_bent = ring_angle < self.T["bent_max"]
        pinky_bent = pinky_angle < self.T["bent_max"]

        # 5. Force index finger to point straight up (RELATIVE TO HAND SIZE)
        index_mcp_y = lm[5].y
        index_tip_y = lm[8].y
        index_up_dist = index_mcp_y - index_tip_y
        index_up_ratio = index_up_dist / hand_size
        index_point_up = index_up_ratio >= self.T["index_up_ratio"]

        # 6. Validate thumb position: pressed near middle finger base (RELATIVE TO HAND SIZE)
        thumb_to_mid_dist = self._dist(lm[3], lm[9])
        thumb_to_mid_ratio = thumb_to_mid_dist / hand_size
        thumb_ok = thumb_to_mid_ratio <= self.T["thumb_to_mid_ratio"]

        # Combine all K gesture recognition rules
        all_conditions = (
            palm_valid and palm_side_ok and
            index_straight and index_point_up and
            middle_straight and ring_bent and
            pinky_bent and thumb_ok
        )

        # Update consecutive valid frame counter for anti-shake
        if all_conditions:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0

        # Return final recognition result
        result = self.consecutive_correct >= self.required_consecutive
        return result if current_step == 1 else self.stroke_count