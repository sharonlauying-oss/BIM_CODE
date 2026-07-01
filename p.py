# gestures/p.py - P
# 🤟 BIM "P" Gesture Description:
# - Hand Orientation: Palm faces sideways/downward with the wrist located at the top.
# - Index & Middle Fingers: Extended straight and pointing directly downwards.
# - Thumb: Tucked horizontally across the palm; the tip (4) rests near the index 
#   middle joint (6), while the base (3) stretches away from the index root (5).
# - Ring & Pinky Fingers: Fully curled and tucked tightly into the palm.

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("P")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        self.stroke_count = 1
        
        # 🐛 Debug log flag
        self.enable_debug_log = False 

        # 🎯 Golden thresholds calibrated perfectly from your live dataset
        self.THRESHOLDS = {
            "min_palm_height": 0.04,       
            "palm_ratio_max": 0.65,        
            "finger_straight_min": 128,    
            "finger_curl_max": 118,        
            "index_mid_angle_min": 0,      
            "index_mid_angle_max": 45,     
            
            # 👍 Thumb strict core rules
            # explain: Limit to ensure thumb tip (4) stays tucked close to index middle joint (6)
            "thumb_4_to_index_6_max": 0.32, 
            # explain: Lower bound to ensure thumb base joint (3) stretches away from index base (5),
            # calibrated to 0.05 to perfectly accept natural compression values like 0.076 and 0.089.
            "thumb_3_to_index_5_min": 0.05, 
            
            "curled_tip_dist_max": 0.70    
        }

    def _dist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _wdist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)

    def _angle_3d(self, p1, p2, p3):
        v1x, v1y, v1z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        v2x, v2y, v2z = p3.x - p2.x, p3.y - p2.y, p3.z - p2.z
        dot = v1x * v2x + v1y * v2y + v1z * v2z
        mag1 = math.hypot(v1x, v1y, v1z)
        mag2 = math.hypot(v2x, v2y, v2z)
        if mag1 == 0 or mag2 == 0: return 0
        return math.degrees(math.acos(max(min(dot / (mag1 * mag2), 1.0), -1.0)))

    def _vector_angle_3d(self, p1_start, p1_end, p2_start, p2_end):
        v1x, v1y, v1z = p1_end.x - p1_start.x, p1_end.y - p1_start.y, p1_end.z - p1_start.z
        v2x, v2y, v2z = p2_end.x - p2_start.x, p2_end.y - p2_start.y, p2_end.z - p2_start.z
        dot = v1x * v2x + v1y * v2y + v1z * v2z
        mag1 = math.hypot(v1x, v1y, v1z)
        mag2 = math.hypot(v2x, v2y, v2z)
        if mag1 == 0 or mag2 == 0: return 0
        return math.degrees(math.acos(max(min(dot / (mag1 * mag2), 1.0), -1.0)))

    def reset_dynamic_tracking(self):
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        if not hands_list:
            self.consecutive_correct = 0
            return 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # explain: Normalize distances relative to the main hand size vector (Wrist to Middle Finger Base)
        hand_size = self._dist(lm[0], lm[9])
        if hand_size == 0: return 0

        # 1. Palm Orientation & Hand Tracking Status
        # explain: Verifies that a valid 3D hand is detected within reasonable depth bounds
        palm_h_3d = self._wdist(wm[0], wm[9])
        palm_valid = palm_h_3d >= self.THRESHOLDS["min_palm_height"]
        
        # explain: Validates side-facing posture profiles by capping horizontal aspect ratios
        palm_w_2d = self._dist(lm[5], lm[17])
        palm_ratio = palm_w_2d / hand_size
        palm_facing_side = palm_ratio < self.THRESHOLDS["palm_ratio_max"]
        
        # explain: Guarantees the hand skeleton points downwards (Wrist is above the knuckles)
        pointing_down = (lm[0].y < lm[9].y) or (lm[9].x < lm[0].x and lm[9].y > lm[0].y - 0.1)

        # 2. Index Finger Evaluation
        # explain: Checks if the index finger is extended straight downwards
        idx_angle = self._angle_3d(wm[5], wm[6], wm[8])
        idx_straight = idx_angle > self.THRESHOLDS["finger_straight_min"]
        idx_pointing_down = lm[8].y > lm[5].y or lm[8].x < lm[5].x

        # 3. Middle Finger Evaluation
        # explain: Checks if the middle finger is extended straight downwards
        mid_angle = self._angle_3d(wm[9], wm[10], wm[12])
        mid_straight = mid_angle > self.THRESHOLDS["finger_straight_min"]
        mid_pointing_down = lm[12].y > lm[9].y or lm[12].x < lm[9].x

        # 3.5 Inter-finger Divergence
        # explain: Monitors the structural gap/angle between the index and middle fingers
        divergence_angle = self._vector_angle_3d(wm[5], wm[8], wm[9], wm[12])
        divergence_ok = (self.THRESHOLDS["index_mid_angle_min"] 
                         <= divergence_angle 
                         <= self.THRESHOLDS["index_mid_angle_max"])

        # 4. Thumb Core Rule Verification
        # explain: Calculates normalized 2D distance for the core user rules
        dist_4_to_6 = self._dist(lm[4], lm[6]) / hand_size  
        dist_3_to_5 = self._dist(lm[3], lm[5]) / hand_size  
        
        # explain: Rule 1 - Thumb tip (4) must remain close to the index mid-joint (6)
        thumb_tip_ok = dist_4_to_6 < self.THRESHOLDS["thumb_4_to_index_6_max"]
        
        # explain: Rule 2 - Thumb joint (3) must stretch away from the index root (5) to prevent over-tucking
        thumb_base_ok = dist_3_to_5 > self.THRESHOLDS["thumb_3_to_index_5_min"]
        
        # explain: Consolidate thumb parameters into a single boolean state flag
        thumb_ok = thumb_tip_ok and thumb_base_ok

        # 5. Ring and Pinky Finger Curl Evaluation
        # explain: Determines whether the inactive fingers are securely tucked into the palm via joint angles and distances
        ring_angle = self._angle_3d(wm[13], wm[14], wm[16])
        pinky_angle = self._angle_3d(wm[17], wm[18], wm[20])
        ring_curled = ring_angle < self.THRESHOLDS["finger_curl_max"]
        pinky_curled = pinky_angle < self.THRESHOLDS["finger_curl_max"]
        
        ring_tucked_ratio = self._dist(lm[13], lm[16]) / hand_size
        pinky_tucked_ratio = self._dist(lm[17], lm[20]) / hand_size
        ring_tucked = ring_tucked_ratio < self.THRESHOLDS["curled_tip_dist_max"]
        pinky_tucked = pinky_tucked_ratio < self.THRESHOLDS["curled_tip_dist_max"]

        curled_fingers_ok = (ring_curled or ring_tucked) and (pinky_curled or pinky_tucked)

        # ⚖️ Ultimate Condition Combination
        # explain: Evaluates all architectural layout requirements in parallel
        all_ok = (palm_valid and palm_facing_side and pointing_down and 
                  idx_straight and idx_pointing_down and 
                  mid_straight and mid_pointing_down and divergence_ok and 
                  thumb_ok and curled_fingers_ok)

        # explain: Debounce filter framework to prevent frame jitter and accidental activations
        self.consecutive_correct = self.consecutive_correct + 1 if all_ok else 0
        final_ok = self.consecutive_correct >= self.required_consecutive

        if self.enable_debug_log:
            print("\n============================================================")
            print("📊 [P Gesture - Pure Metrics Version] Real-time Geometric Data Stream")
            print(f"  [Thumb Metric Validation]:")
            print(f"    - 4->6 Proximity (Tip Close): {dist_4_to_6:.3f} (Req: < {self.THRESHOLDS['thumb_4_to_index_6_max']}) -> {thumb_tip_ok}")
            print(f"    - 3->5 Separation (Base Away): {dist_3_to_5:.3f} (Req: > {self.THRESHOLDS['thumb_3_to_index_5_min']}) -> {thumb_base_ok}")
            print(f"  ⭐ [Thumb Combined Status]: {thumb_ok}")
            print(f"✅ Current Frame Match: {all_ok} | Stabilizer Counter: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")

        if current_step == 1:
            return 1 if final_ok else 0
        else:
            return self.stroke_count