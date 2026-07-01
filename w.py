# ==============================================================================
# 🤟 BIM "W" Gesture Description:
# - Hand Orientation: Right hand, palm facing the camera directly.
# - Index, Middle & Ring Fingers: Fully straight, spread apart to form W shape.
#   Validated by joint extension and perspective foreshortening length ratios.
#   Separation spaces are locked between tip pairs (8 to 12) and (12 to 16).
# - Pinky Finger: Curled and compressed down into the lower palm zone.
# - Thumb: Tip landmark 4 presses firmly down onto Pinky DIP joint landmark 19.
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("W")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        self.stroke_count = 1

        # 🐛 Enable explicit runtime console diagnostics
        self.enable_debug_log = False 

        # 🎯 Geometric thresholds optimized for perspective projection validation
        self.THRESHOLDS = {
            "min_palm_height": 0.03,
            
            # ☝️ 🖕 🖕 Joint extension boundaries
            "extended_joint_angle_min": 150.0,
            
            # 📏 🌟 Perspective Foreshortening Ratio (Anti-Flattening Guard)
            "min_index_length_ratio": 0.45,   # (Dist 6 to 8) / hand_size
            "min_middle_length_ratio": 0.45,  # (Dist 10 to 12) / hand_size
            "min_ring_length_ratio": 0.45,    # (Dist 14 to 16) / hand_size

            # 👐 W-Shape Flaring Limits (Normalized minimum distances)
            "index_to_middle_tip_min": 0.32,  # Distance between 8 and 12
            "middle_to_ring_tip_min": 0.32,   # Distance between 12 and 16

            # ✊ Pinky Fold Profile 
            "pinky_joint_angle_max": 45.0,
            
            # 👍 Cross-Palm Thumb Guard: Max 2D distance between Thumb Tip (4) and Pinky DIP (19)
            "thumb_tip_to_pinky_dip_max": 0.15
        }

    def _dist(self, p1, p2):
        # Calculate 2D Euclidean distance between two coordinate markers
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _wdist(self, p1, p2):
        # Calculate 3D World Euclidean distance using full spatial coordinates
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _angle(self, p1, p2, p3):
        # Calculate the 2D joint flexion angle using vector dot product calculation
        v1x, v1y = p1.x - p2.x, p1.y - p2.y
        v2x, v2y = p3.x - p2.x, p3.y - p2.y
        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0: return 0.0
        cos_angle = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_angle))

    def reset_dynamic_tracking(self):
        # Reset dynamic timeline memory state indicators
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        # Safety fallback: If no hand tracking framework matrix is found, drop frame count
        if not hands_list:
            self.consecutive_correct = 0
            return 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # Establish dynamic scale boundary based on palm profile (Wrist 0 to Middle MCP 9)
        hand_size = self._dist(lm[0], lm[9])
        if hand_size == 0: return 0

        # 1. Structural Baseline: Palm Verification 
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # 2. Traditional Joint Angle Verification
        idx_angle = self._angle(lm[5], lm[6], lm[8])
        mid_angle = self._angle(lm[9], lm[10], lm[12])
        rng_angle = self._angle(lm[13], lm[14], lm[16])
        
        angles_ok = (idx_angle >= self.THRESHOLDS["extended_joint_angle_min"] and 
                     mid_angle >= self.THRESHOLDS["extended_joint_angle_min"] and 
                     rng_angle >= self.THRESHOLDS["extended_joint_angle_min"])

        # 3. 🌟 Perspective Anti-Flattening Guard (Foreshortening Length Ratio Check)
        ratio_idx = self._dist(lm[6], lm[8]) / hand_size   # PIP (6) to TIP (8)
        ratio_mid = self._dist(lm[10], lm[12]) / hand_size # PIP (10) to TIP (12)
        ratio_rng = self._dist(lm[14], lm[16]) / hand_size # PIP (14) to TIP (16)

        lengths_ok = (ratio_idx >= self.THRESHOLDS["min_index_length_ratio"] and
                      ratio_mid >= self.THRESHOLDS["min_middle_length_ratio"] and
                      ratio_rng >= self.THRESHOLDS["min_ring_length_ratio"])

        # 4. Double Flaring Check (W-shape separation matching)
        dist_8_to_12 = self._dist(lm[8], lm[12]) / hand_size
        dist_12_to_16 = self._dist(lm[12], lm[16]) / hand_size
        
        w_spread_ok = (dist_8_to_12 >= self.THRESHOLDS["index_to_middle_tip_min"] and 
                       dist_12_to_16 >= self.THRESHOLDS["middle_to_ring_tip_min"])

        # 5. Fold Verification: Pinky remains compressed down against palm
        pky_angle = self._angle(lm[17], lm[18], lm[20])
        pinky_folded = pky_angle <= self.THRESHOLDS["pinky_joint_angle_max"]

        # 6. 👍 Thumb Physical Anchor Check: Corrected to track Pinky DIP joint (Landmark 19)
        thumb_to_pinky_dip = self._dist(lm[4], lm[19]) / hand_size
        thumb_anchored = thumb_to_pinky_dip <= self.THRESHOLDS["thumb_tip_to_pinky_dip_max"]

        # 7. Integrated Execution Pipeline
        all_ok = palm_valid and angles_ok and lengths_ok and w_spread_ok and pinky_folded and thumb_anchored

        # Dynamic debounce mechanism to clear noise and stabilize frame updates
        self.consecutive_correct = self.consecutive_correct + 1 if all_ok else 0
        final_ok = self.consecutive_correct >= self.required_consecutive

        if self.enable_debug_log:
            print("\n============================================================")
            print(f"📊 [W Gesture Profile] Real-Time Runtime Metrics")
            print(f"  [1. Palm Height Validity]: Depth={palm_h:.3f} -> Status: {palm_valid}")
            print(f"  [2. Three Pillars Extension]: Index={idx_angle:.1f}° | Middle={mid_angle:.1f}° | Ring={rng_angle:.1f}° -> Status: {angles_ok}")
            print(f"  [3. 🌟 Foreshortening Ratios]: Index={ratio_idx:.3f} (Req: >{self.THRESHOLDS['min_index_length_ratio']})")
            print(f"                               Middle={ratio_mid:.3f} (Req: >{self.THRESHOLDS['min_middle_length_ratio']})")
            print(f"                               Ring={ratio_rng:.3f} (Req: >{self.THRESHOLDS['min_ring_length_ratio']}) -> Status: {lengths_ok}")
            print(f"  [4. W-Shape Finger Flaring]: Split(8-12)={dist_8_to_12:.3f} | Split(12-16)={dist_12_to_16:.3f} -> Status: {w_spread_ok}")
            print(f"  [5. Pinky Fold Status]: Angle={pky_angle:.1f}° -> Status: {pinky_folded}")
            print(f"  [6. Thumb Cross Anchor (4 to 19)]: Normalized 2D Dist={thumb_to_pinky_dip:.3f} -> Status: {thumb_anchored}")
            print(f"❌/✅ Current Frame Result: {all_ok} | Stabilizer Buffer: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")

        if current_step == 1:
            return 1 if final_ok else 0
        else:
            return self.stroke_count