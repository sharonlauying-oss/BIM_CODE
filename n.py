# gestures/n.py - N Gesture (Final Calibrated Version)
# BIM N sign: palm facing camera
# 1. Pinky + Ring finger: fully folded into palm
# 2. Thumb: pressing on top of ring finger (tip near ring MCP)
# 3. Index + Middle: folded base, straight top 2 segments, lay over thumb
import math
from gestures.base_gesture import BaseGesture


class GestureChecker(BaseGesture):
    """
    Gesture recognizer for BIM letter 'N'
    Palm facing camera
    - Ring & Pinky: fully curled into palm
    - Thumb: rests on ring finger, tip near ring MCP
    - Index + Middle: base folded, top 2 segments straight, lay over thumb
    """
    def __init__(self):
        super().__init__("N")
        self.consecutive_correct = 0
        self.required_consecutive = 2

        # Calibrated geometric thresholds
        self.T = {
            "min_palm_h": 0.06,
            "palm_ratio_min": 0.35,
            "palm_ratio_max": 0.75,
            "straight_min": 130,
            "root_proximity_max": 0.43,
            "thumb_straight_min": 135,
            "thumb_to_ring_mcp_max": 0.50,
            "palm_center_tolerance": 0.45,
            "thumb_tip_to_pinky_pip_min": 0.30,  
        }

    def _dist(self, p1, p2):
        """Calculate 2D Euclidean distance between two landmarks."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _wdist(self, p1, p2):
        """Calculate 3D metric world distance."""
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    def _angle(self, p1, p2, p3):
        """Calculate the 2D joint angle at vertex p2 in degrees."""
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
        """Reset validation history when transitioning or tracking is lost."""
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        # Return safely if no hand is detected
        if not hands_list:
            self.consecutive_correct = 0
            return False if current_step == 1 else 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # Define an adaptive hand size reference to handle distance variances
        hand_size = self._dist(lm[0], lm[9])
        if hand_size < 0.01:
            hand_size = 0.1

        # Compute the geometric center of the palm area
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        palm_center = Point(
            (lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5,
            (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5
        )

        # 1. Verify palm is facing forward and matches acceptable aspect ratios
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.T["min_palm_h"]
        thumb_on_left = lm[1].x < lm[17].x
        palm_w = self._dist(lm[5], lm[17])
        palm_h_2d = self._dist(lm[0], lm[9])
        palm_ratio = palm_w / palm_h_2d if palm_h_2d > 0 else 0
        palm_ratio_ok = self.T["palm_ratio_min"] < palm_ratio < self.T["palm_ratio_max"]
        palm_front_ok = palm_valid and thumb_on_left and palm_ratio_ok

        # 2. Check if Ring + Pinky fingers are fully folded into the palm
        def check_folded(name, mcp, pip, dip, tip):
            root_ratio = self._dist(lm[mcp], lm[pip]) / hand_size
            root_close = root_ratio < self.T["root_proximity_max"]
            tip_ratio = self._dist(lm[tip], palm_center) / hand_size
            tip_in = tip_ratio < self.T["palm_center_tolerance"]
            ok = root_close and tip_in
            return ok

        ring_ok = check_folded("Ring", 13, 14, 15, 16)
        pinky_ok = check_folded("Pinky", 17, 18, 19, 20)
        folded_ok = ring_ok and pinky_ok

        # 3. Check Thumb placement: straight, positioned over the ring finger
        thumb_angle = self._angle(lm[2], lm[3], lm[4])
        thumb_straight = thumb_angle > self.T["thumb_straight_min"]
        thumb_to_ring = self._dist(lm[4], lm[13]) / hand_size
        thumb_on_ring = thumb_to_ring < self.T["thumb_to_ring_mcp_max"]

        # Prevent false positives: Ensure the thumb tip is far enough from the pinky PIP joint
        thumb_tip_to_pinky_pip = self._dist(lm[4], lm[18]) / hand_size
        thumb_not_near_pinky = thumb_tip_to_pinky_pip > self.T["thumb_tip_to_pinky_pip_min"]

        # Combine all thumb criteria
        thumb_ok = thumb_straight and thumb_on_ring and thumb_not_near_pinky

        # 4. Check Index + Middle fingers: Base must be folded down while top segments remain straight
        def check_top_straight(name, mcp, pip, dip, tip):
            root_ratio = self._dist(lm[mcp], lm[pip]) / hand_size
            root_close = root_ratio < self.T["root_proximity_max"]
            angle = self._angle(lm[pip], lm[dip], lm[tip])
            straight = angle > self.T["straight_min"]
            ok = root_close and straight
            return ok

        index_ok = check_top_straight("Index", 5, 6, 7, 8)
        middle_ok = check_top_straight("Middle", 9, 10, 11, 12)
        top_ok = index_ok and middle_ok

        # Evaluate final gesture combination
        all_ok = palm_front_ok and folded_ok and thumb_ok and top_ok

        # Accumulate consecutive positive frames to filter out noise
        if all_ok:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0

        # Return results contextually based on the execution step
        result = self.consecutive_correct >= self.required_consecutive
        return result if current_step == 1 else self.stroke_count