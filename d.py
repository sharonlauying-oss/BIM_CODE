# gestures/d.py

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("D")
        self.consecutive_correct = 0
        self.required_consecutive = 2  # Debounce filter (requires 2 consecutive stable frames)

        # Geometric rules for gesture "D"
        self.THRESHOLDS = {
            "index_angle_min": 165,         # Index finger joint angle must be > 165° (Straight)
            "index_vertical_max": 15,       # Deviation from absolute vertical axis must be within 15°
            "index_up_diff_min": 0.2,       # Index tip must be significantly higher than its root
            "four_tips_max_distance": 0.08,  # Max 3D distance (8cm) between the clustered finger tips
            "other_fingers_bend_min": 90    # Ensures middle, ring, and pinky are bent
        }

    def _get_world_distance(self, p1, p2):
        """Calculates 3D physical distance in meters. Invariant to camera distance."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def _get_angle(self, p1, p2, p3):
        """Calculates 2D joint angle at vertex p2."""
        v1x, v1y = p1.x - p2.x, p1.y - p2.y
        v2x, v2y = p3.x - p2.x, p3.y - p2.y
        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0: 
            return 0.0
        cos = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos))

    def reset_dynamic_tracking(self):
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int) -> bool:
        if not hands_list:
            self.consecutive_correct = 0
            return False

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # FEATURE 1: Index Straightness
        index_angle = self._get_angle(lm[5], lm[6], lm[8])
        index_straight = index_angle > self.THRESHOLDS["index_angle_min"]

        # FEATURE 2: Index Verticality & Height Extension
        index_dx = lm[8].x - lm[5].x
        index_dy = lm[8].y - lm[5].y
        
        index_up_diff = lm[5].y - lm[8].y
        index_high_enough = index_up_diff > self.THRESHOLDS["index_up_diff_min"]
        
        index_vertical_angle = abs(math.degrees(math.atan2(index_dx, -index_dy)))
        index_vertical = index_vertical_angle < self.THRESHOLDS["index_vertical_max"]

        # FEATURE 3: Index Is Single Highest Point
        index_tip_y = lm[8].y
        other_tips_y = [lm[4].y, lm[12].y, lm[16].y, lm[20].y]
        index_alone_top = index_tip_y < min(other_tips_y)

        # FEATURE 4: Clustered Tips Loop (Thumb + Middle + Ring + Pinky)
        thumb_to_middle = self._get_world_distance(wm[4], wm[12])
        middle_to_ring = self._get_world_distance(wm[12], wm[16])
        ring_to_pinky = self._get_world_distance(wm[16], wm[20])
        
        four_tips_together = (
            thumb_to_middle < self.THRESHOLDS["four_tips_max_distance"] and
            middle_to_ring < self.THRESHOLDS["four_tips_max_distance"] and
            ring_to_pinky < self.THRESHOLDS["four_tips_max_distance"]
        )

        # FEATURE 5: Bending Verification for Non-Index Fingers
        middle_bend = self._get_angle(lm[9], lm[10], lm[12]) > self.THRESHOLDS["other_fingers_bend_min"]
        ring_bend = self._get_angle(lm[13], lm[14], lm[16]) > self.THRESHOLDS["other_fingers_bend_min"]
        pinky_bend = self._get_angle(lm[17], lm[18], lm[20]) > self.THRESHOLDS["other_fingers_bend_min"]
        others_bend = middle_bend and ring_bend and pinky_bend

        # Combine all rules
        gesture_matched = (
            index_straight and 
            index_high_enough and 
            index_vertical and 
            index_alone_top and 
            four_tips_together and 
            others_bend
        )

        # Debounce logic
        if gesture_matched:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0  

        return self.consecutive_correct >= self.required_consecutive