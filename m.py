# gestures/m.py - M Gesture (Final Clean Version with English Comments)
# BIM M sign: palm facing camera, thumb over pinky, index/middle/ring over thumb
# Calibrated with actual user gesture data, supports full distance adaptation

import math
from gestures.base_gesture import BaseGesture


class GestureChecker(BaseGesture):
    """
    Gesture recognizer for BIM letter 'M'.
    Adaptive to camera distance using normalized hand-size ratios.
    """
    def __init__(self):
        super().__init__("M")
        # Counter for consecutive correct frames (anti-jitter)
        self.consecutive_correct = 0
        # Require N consecutive passes to confirm gesture
        self.required_consecutive = 2

        # Tuned thresholds derived from user's real gesture samples
        self.T = {
            "min_palm_h": 0.06,                # Min palm 3D height (front-facing check)
            "palm_ratio_min": 0.35,            # Min palm aspect ratio (w/h)
            "palm_ratio_max": 0.75,            # Max palm aspect ratio (w/h)
            "straight_min": 130,                # Min angle for "straight enough" finger
            "root_proximity_max": 0.43,         # Max normalized distance for folded finger root
            "thumb_straight_min": 135,         # Min angle for straight thumb
            "thumb_to_pinky_mcp_max": 0.36,     # Max normalized distance: thumb tip -> pinky MCP
            "palm_center_tolerance": 0.45,      # Max normalized distance: fingertip -> palm center
        }

    def _dist(self, p1, p2):
        """2D Euclidean distance between two landmarks."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _wdist(self, p1, p2):
        """3D world-space Euclidean distance between two landmarks."""
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    def _angle(self, p1, p2, p3):
        """
        Calculate angle (degrees) at p2 formed by p1-p2-p3.
        Used to check if a finger is straight.
        """
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
        """Reset consecutive correct counter (called on state reset)."""
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        """
        Main gesture check pipeline for 'M'.
        Returns True if gesture is stable and valid.
        """
        # No hands detected → reset counter
        if not hands_list:
            self.consecutive_correct = 0
            return False if current_step == 1 else 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # Base hand size (wrist to middle MCP) for distance normalization
        # All distance checks are relative to this → works at any distance
        hand_size = self._dist(lm[0], lm[9])
        if hand_size < 0.01:
            hand_size = 0.1  # Avoid division by zero

        # Palm center (average of key palm landmarks)
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        palm_center = Point(
            (lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5,
            (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5
        )

        # 1. Palm orientation check
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.T["min_palm_h"]  # Palm facing forward
        thumb_on_left = lm[1].x < lm[17].x             # Thumb on left side (correct orientation)
        palm_w = self._dist(lm[5], lm[17])
        palm_h_2d = self._dist(lm[0], lm[9])
        palm_ratio = palm_w / palm_h_2d if palm_h_2d > 0 else 0
        palm_ratio_ok = self.T["palm_ratio_min"] < palm_ratio < self.T["palm_ratio_max"]
        palm_front_ok = palm_valid and thumb_on_left and palm_ratio_ok

        # 2. Pinky check (folded into palm)
        pinky_root_ratio = self._dist(lm[17], lm[18]) / hand_size
        pinky_root_close = pinky_root_ratio < self.T["root_proximity_max"]
        pinky_tip_ratio = self._dist(lm[20], palm_center) / hand_size
        pinky_tip_in_palm = pinky_tip_ratio < self.T["palm_center_tolerance"]
        pinky_ok = pinky_root_close and pinky_tip_in_palm

        # 3. Thumb check (straight, over pinky)
        thumb_angle = self._angle(lm[2], lm[3], lm[4])
        thumb_straight = thumb_angle > self.T["thumb_straight_min"]
        thumb_to_pinky_ratio = self._dist(lm[4], lm[17]) / hand_size
        thumb_near_pinky = thumb_to_pinky_ratio < self.T["thumb_to_pinky_mcp_max"]
        thumb_ok = thumb_straight and thumb_near_pinky

        # 4. Index / Middle / Ring (folded over thumb)
        def check_finger(mcp_idx, pip_idx, dip_idx, tip_idx):
            """
            Check single folded finger: root close, front straight, tip near palm center.
            """
            root_ratio = self._dist(lm[mcp_idx], lm[pip_idx]) / hand_size
            root_close = root_ratio < self.T["root_proximity_max"]
            front_angle = self._angle(lm[pip_idx], lm[dip_idx], lm[tip_idx])
            front_straight = front_angle > self.T["straight_min"]
            tip_ratio = self._dist(lm[tip_idx], palm_center) / hand_size
            tip_in_palm = tip_ratio < self.T["palm_center_tolerance"]
            return root_close and front_straight and tip_in_palm

        index_ok = check_finger(5, 6, 7, 8)
        middle_ok = check_finger(9, 10, 11, 12)
        ring_ok = check_finger(13, 14, 15, 16)
        fingers_ok = index_ok and middle_ok and ring_ok

        # Final decision
        all_ok = palm_front_ok and pinky_ok and thumb_ok and fingers_ok

        # Update stability counter
        if all_ok:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0

        # Only return True if stable for N consecutive frames
        result = self.consecutive_correct >= self.required_consecutive
        return result if current_step == 1 else self.stroke_count