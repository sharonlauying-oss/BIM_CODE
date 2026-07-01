# gestures/a.py

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("A")
        self.consecutive_correct = 0
        self.required_consecutive = 2  # Debounce filter
        
        # Hyper-calibrated thresholds matching your 5-sample golden dataset
        self.THRESHOLDS = {
            "min_palm_height": 0.06,      # Filters out invalid distant hands or glitches
            "finger_bend_angle": 80,       # Angle < 80° ensures fingers are clenched into a fist
            "tip_to_wrist_max": 0.23,      # Max 3D distance from finger tips to wrist (folded profile)
            "thumb_straight_angle": 145,   # Thumb angle > 145° means thumb is fully extended
            "thumb_angle_min": -10,        # Min vertical angle tolerance for upright thumb
            "thumb_angle_max": 5,          # Max vertical angle tolerance for upright thumb
            "thumb_to_index_max": 0.114,   # Strict distance ceiling keeping thumb tucked to index
            "not_thumbs_up_offset": 0.15,  # Height filter to prevent mistaking A for Thumbs-Up gesture
            "palm_ratio": 0.35             # Width/Height ratio to guarantee the palm faces the camera
        }
    
    def _get_distance(self, p1, p2):
        """Calculates 2D screen distance for aspect ratios."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def _get_world_distance(self, p1, p2):
        """Calculates 3D physical metric distance in meters."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def _get_angle(self, p1, p2, p3):
        """Calculates 2D joint angle at vertex p2."""
        v1x = p1.x - p2.x
        v1y = p1.y - p2.y
        v2x = p3.x - p2.x
        v2y = p3.y - p2.y
        
        dot = v1x * v2x + v1y * v2y
        mag1 = math.sqrt(v1x**2 + v1y**2)
        mag2 = math.sqrt(v2x**2 + v2y**2)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        return math.degrees(math.acos(cos_angle))
    
    def reset_dynamic_tracking(self):
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0
    
    def check_gesture(self, hands_list: list, current_step: int) -> bool:
        if not hands_list:
            self.consecutive_correct = 0
            return False
        
        target_hand = hands_list[0]
        lm = target_hand["landmarks"]
        wm = target_hand["world_landmarks"]
        
        # Filter out invalid size artifacts
        palm_h = self._get_world_distance(wm[0], wm[9])
        if palm_h < self.THRESHOLDS["min_palm_height"]:
            self.consecutive_correct = 0
            return False
        
        # FEATURE 1: Clenched Fist Check (Index, Middle, Ring, Pinky)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        mcps = [5, 9, 13, 17]
        
        all_fingers_tight = True
        for t, p, m in zip(tips, pips, mcps):
            cond1 = lm[t].y > lm[m].y   
            cond2 = lm[p].y < lm[m].y   
            cond3 = self._get_angle(lm[m], lm[p], lm[t]) < self.THRESHOLDS["finger_bend_angle"]
            cond4 = self._get_world_distance(wm[t], wm[0]) < self.THRESHOLDS["tip_to_wrist_max"]
            
            if not (cond1 and cond2 and cond3 and cond4):
                all_fingers_tight = False
                break
        
        # FEATURE 2: 5-Layer Strict Thumb Validation
        thumb_straight = self._get_angle(lm[2], lm[3], lm[4]) > self.THRESHOLDS["thumb_straight_angle"]
        
        thumb_dx = lm[4].x - lm[2].x
        thumb_dy = lm[4].y - lm[2].y
        thumb_angle = math.degrees(math.atan2(thumb_dx, -thumb_dy))
        thumb_up_ok = (
            thumb_angle > self.THRESHOLDS["thumb_angle_min"] and
            thumb_angle < self.THRESHOLDS["thumb_angle_max"]
        )
        
        thumb_to_index = self._get_world_distance(wm[4], wm[5])
        thumb_near = thumb_to_index < self.THRESHOLDS["thumb_to_index_max"]
        
        thumb_on_left = lm[4].x < lm[5].x
        
        not_thumbs_up = lm[4].y > (lm[5].y - self.THRESHOLDS["not_thumbs_up_offset"])
        
        thumb_ok = (
            thumb_straight and
            thumb_up_ok and
            thumb_near and
            thumb_on_left and
            not_thumbs_up
        )
        
        # FEATURE 3: Palm Facing Aspect Ratio
        palm_w = self._get_distance(lm[5], lm[17])
        palm_h_2d = self._get_distance(lm[0], lm[9])
        palm_ratio = palm_w / palm_h_2d if palm_h_2d > 0 else 0
        palm_ok = palm_ratio > self.THRESHOLDS["palm_ratio"]
        
        # Master decision logic
        current_ok = all_fingers_tight and thumb_ok and palm_ok
        
        if current_ok:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0  
        
        gesture_valid = self.consecutive_correct >= self.required_consecutive
        
        return gesture_valid if current_step == 1 else self.stroke_count