# gestures/g.py - G gesture
# Gesture: palm left, pinky side to camera; middle/ring/pinky curled; index+thumb straight left

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("G")
        self.consecutive_correct = 0
        self.required_consecutive = 2  # anti-shake

        # Tuned thresholds for your pose
        self.T = {
            'min_palm_h': 0.06,
            'bent_max_angle': 80,
            'straight_min_angle': 145,
            'thumb_index_max_dist': 0.22,
            'palm_ratio_max': 0.30  # palm facing left
        }

    # 2D Euclidean distance
    def _dist(self, p1, p2):
        return math.hypot(p1.x-p2.x, p1.y-p2.y)

    # 3D world distance
    def _wdist(self, p1, p2):
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    # Finger angle at middle joint
    def _angle(self, p1, p2, p3):
        v1x, v1y = p1.x-p2.x, p1.y-p2.y
        v2x, v2y = p3.x-p2.x, p3.y-p2.y
        dot = v1x*v2x + v1y*v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        cos_a = max(min(dot/(mag1*mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_a))

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

        # Palm size check
        palm_h = self._wdist(wm[0], wm[9])
        palm_ok = palm_h >= self.T['min_palm_h']

        # Finger joint angles
        idx_a = self._angle(lm[5], lm[6], lm[8])
        mid_a = self._angle(lm[9], lm[10], lm[12])
        ring_a = self._angle(lm[13], lm[14], lm[16])
        pinky_a = self._angle(lm[17], lm[18], lm[20])
        thumb_a = self._angle(lm[1], lm[2], lm[3])

        # Middle/ring/pinky bent (curled into palm)
        three_bent = (mid_a < self.T['bent_max_angle'] and
                       ring_a < self.T['bent_max_angle'] and
                       pinky_a < self.T['bent_max_angle'])

        # Index + thumb straight
        two_straight = (idx_a > self.T['straight_min_angle'] and
                         thumb_a > self.T['straight_min_angle'])

        # Thumb & index tip close together
        tip_dist = self._wdist(wm[4], wm[8])
        tips_ok = tip_dist < self.T['thumb_index_max_dist']

        # Palm facing left (width/height ratio)
        palm_w = self._dist(lm[5], lm[17])
        palm_h2d = self._dist(lm[0], lm[9])
        palm_ratio = palm_w / palm_h2d if palm_h2d > 0 else 0
        palm_left_ok = palm_ratio < self.T['palm_ratio_max']

        # All conditions met?
        valid = palm_ok and three_bent and two_straight and tips_ok and palm_left_ok

        if valid:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0

        result = self.consecutive_correct >= self.required_consecutive
        return result if current_step == 1 else self.stroke_count