# ==============================================================================
# 🤟 BIM "R" Gesture Description:
# - Hand Orientation: Right hand, palm facing the CAMERA.
# - Index & Middle Fingers: Straight and intertwined, index finger landmark 7 is close to middle finger landmark 11.
# - Ring & Pinky Fingers: Proximal phalanx folded, distal two phalanges fully straight.
# - Thumb: Thumb tip (landmark 4) presses against ring finger (landmark 15).
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("R")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        self.stroke_count = 1

        # 🐛 Enable explicit runtime console diagnostics
        self.enable_debug_log = False 

        # 🎯 Geometrical thresholds adjusted to accommodate 2D flattened perspective flattening
        self.THRESHOLDS = {
            "min_palm_height": 0.03,
            
            # Active Straight Group (Index & Middle)
            "index_straight_min": 150,     
            "middle_straight_min": 150,    
            
            # 📏 🌟 Perspective Foreshortening Ratio (Self-Adaptive Segment Length Ratios)
            # When fingers bend toward the camera, their projected length ratio drops below these values.
            "min_index_length_ratio": 0.42,   # (Dist 6 to 8) / hand_size
            "min_middle_length_ratio": 0.45,  # (Dist 10 to 12) / hand_size

            # R-Knot Lock: Proximity threshold between Index PIP (7) and Middle PIP (11)
            "knot_proximity_ratio": 0.1,  
            
            # 🌟 Folded Group (Ring & Pinky Perspective Adjustment)
            # explanation: Due to Z-depth compression, folded fingers register close to 180° in 2D. 
            # We change this to require > 110° to match the flattened profile caught in your logs.
            "folded_finger_angle_min": 110, 
            
            # 🌟 Thumb Position Lock: Relaxed from 0.150 to 0.200 to accommodate the 0.174 drift
            "thumb_to_ring_mcp_ratio": 0.20 
        }

    def _dist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _wdist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)

    def _angle(self, p1, p2, p3):
        v1x = p1.x - p2.x
        v1y = p1.y - p2.y
        v2x = p3.x - p2.x
        v2y = p3.y - p2.y
        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
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

        hand_size = self._dist(lm[0], lm[9])
        if hand_size == 0: return 0

        # 1. Base Palm Height Verification
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # 2. Index & Middle Finger Extension Profile (Angles)
        index_angle = self._angle(lm[5], lm[6], lm[8])
        middle_angle = self._angle(lm[10], lm[11], lm[12])
        
        extended_ok = (index_angle >= self.THRESHOLDS["index_straight_min"] and 
                       middle_angle >= self.THRESHOLDS["middle_straight_min"])

        # 3. 🌟 Perspective Anti-Flattening Guard (Foreshortening Length Ratio Check)
        # Calculates the projected length of the upper segments relative to palm size
        ratio_idx = self._dist(lm[6], lm[8]) / hand_size   # PIP (6) to TIP (8)
        ratio_mid = self._dist(lm[10], lm[12]) / hand_size # PIP (10) to TIP (12)

        lengths_ok = (ratio_idx >= self.THRESHOLDS["min_index_length_ratio"] and
                      ratio_mid >= self.THRESHOLDS["min_middle_length_ratio"])

        # 4. R-Knot Lock (Index 7 crossing or tightly bound to Middle 11)
        knot_dist = self._dist(lm[7], lm[11]) / hand_size
        knot_ok = knot_dist <= self.THRESHOLDS["knot_proximity_ratio"]

        # 5. Ring & Pinky Folded State Check (Adapted to flattened projection logs)
        ring_angle = self._angle(lm[14], lm[15], lm[16])
        pky_angle = self._angle(lm[18], lm[19], lm[20])
        
        # explanation: Matches the flat 161.4° ~ 180.0° perspective artifact captured in your metrics
        folded_angles_ok = (ring_angle >= self.THRESHOLDS["folded_finger_angle_min"] and
                            pky_angle >= self.THRESHOLDS["folded_finger_angle_min"])
        
        # Keep the perfectly functioning Y-axis drop check to guarantee they are down
        y_drop_ok = (lm[16].y > lm[14].y and lm[20].y > lm[18].y)

        # 6. Thumb Pressing Ring Finger Lock (Thumb Tip 4 to Ring PIP 15)
        thumb_press_dist = self._dist(lm[4], lm[15]) / hand_size
        thumb_press_ok = thumb_press_dist <= self.THRESHOLDS["thumb_to_ring_mcp_ratio"]

        # 7. Comprehensive Verification Convergence (Now requiring lengths_ok verification)
        all_ok = (palm_valid and extended_ok and lengths_ok and knot_ok and 
                  folded_angles_ok and y_drop_ok and thumb_press_ok)

        # Dynamic Debounce Logic
        self.consecutive_correct = self.consecutive_correct + 1 if all_ok else 0
        final_ok = self.consecutive_correct >= self.required_consecutive

        if self.enable_debug_log:
            print("\n============================================================")
            print(f"📊 [R Gesture Profile] Real-Time Runtime Metrics")
            print(f"  [1. Extension Angles]: Index={index_angle:.1f}° | Middle={middle_angle:.1f}° -> Status: {extended_ok}")
            print(f"  [2. 🌟 Foreshortening Ratios]: Index={ratio_idx:.3f} (Req: >{self.THRESHOLDS['min_index_length_ratio']})")
            print(f"                               Middle={ratio_mid:.3f} (Req: >{self.THRESHOLDS['min_middle_length_ratio']}) -> Status: {lengths_ok}")
            print(f"  [3. R-Knot Proximity Lock]: Normalized Dist={knot_dist:.3f} (Req: < {self.THRESHOLDS['knot_proximity_ratio']:.3f}) -> Status: {knot_ok}")
            print(f"  [4. Folded Group Angles]: Ring={ring_angle:.1f}° | Pinky={pky_angle:.1f}° (Req: > {self.THRESHOLDS['folded_finger_angle_min']}°) -> Status: {folded_angles_ok}")
            print(f"  [5. Folded Y-Axis Drop Check]: Ring_Drop={lm[16].y > lm[14].y} | Pinky_Drop={lm[20].y > lm[18].y} -> Status: {y_drop_ok}")
            print(f"  [6. Thumb Pressing Lock (4 to 15)]: Dist={thumb_press_dist:.3f} (Req: < {self.THRESHOLDS['thumb_to_ring_mcp_ratio']:.3f}) -> Status: {thumb_press_ok}")
            print(f"❌/✅ Current Frame Result: {all_ok} | Stabilizer Buffer: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")

        if current_step == 1:
            return 1 if final_ok else 0
        else:
            return self.stroke_count