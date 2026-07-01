# ==============================================================================
# 🤟 BIM "U" Gesture Description:
# - Hand Orientation: Right hand, palm facing the camera directly.
# - Index & Middle Fingers: Fully straight and tightly attached to each other; restrict the distance between index landmark 8 and middle landmark 12.
# - Ring & Pinky Fingers: Base segments folded, distal two segments fully straight.
# - Thumb: Tip landmark 4 presses on ring finger landmark 15.
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("U")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        self.stroke_count = 1

        # 🐛 Enable explicit runtime console diagnostics
        self.enable_debug_log = False 

        # 🎯 Geometric thresholds calibrated from your front-facing upright telemetry logs
        self.THRESHOLDS = {
            "min_palm_height": 0.03,
            
            # ☝️ 🖕 2D Joint projection limits (unreliable during forward-tilt)
            "extended_joint_angle_min": 150.0,

            # 📏 🌟 Perspective Foreshortening Ratio (Self-Adaptive Segment Length Ratios)
            # When fingers bend toward the camera, their projected length ratio drops below these values.
            "min_index_length_ratio": 0.42,   # (Dist 6 to 8) / hand_size
            "min_middle_length_ratio": 0.45,  # (Dist 10 to 12) / hand_size
            
            # 🤝 Cohesiveness Lock: Maximum normalized distance allowed between Index Tip (8) and Middle Tip (12)
            "index_to_middle_tip_max": 0.2,

            # ✊ Folded Finger Thresholds (Ring and Pinky segments flattened against palm)
            "folded_joint_angle_max": 45.0,
            
            # 👍 Cross-Palm Thumb Guard: Max distance between Thumb Tip (4) and Ring PIP Joint (15)
            "thumb_tip_to_ring_pip_max": 0.2
        }

    def _dist(self, p1, p2):
        # Calculate 2D Euclidean distance between two coordinate markers
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _wdist(self, p1, p2):
        # Calculate 3D World Euclidean distance using full spatial coordinates
        return math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)

    def _angle(self, p1, p2, p3):
        # Calculate the 2D joint flexion angle using vector dot product calculation
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
        angles_ok = (idx_angle >= self.THRESHOLDS["extended_joint_angle_min"] and 
                     mid_angle >= self.THRESHOLDS["extended_joint_angle_min"])

        # 3. 🌟 Perspective Anti-Flattening Guard (Foreshortening Length Ratio Check)
        # Calculates the projected length of the upper segments relative to palm size
        ratio_idx = self._dist(lm[6], lm[8]) / hand_size   # PIP (6) to TIP (8)
        ratio_mid = self._dist(lm[10], lm[12]) / hand_size # PIP (10) to TIP (12)

        lengths_ok = (ratio_idx >= self.THRESHOLDS["min_index_length_ratio"] and
                      ratio_mid >= self.THRESHOLDS["min_middle_length_ratio"])

        # 4. Cohesiveness Check: Index Tip (8) and Middle Tip (12) must touch
        tip_proximity = self._dist(lm[8], lm[12]) / hand_size
        tips_together = tip_proximity <= self.THRESHOLDS["index_to_middle_tip_max"]

        # 5. Base Fold Verification: Ring and Pinky compressed against palm
        rng_angle = self._angle(lm[13], lm[14], lm[16])
        pky_angle = self._angle(lm[17], lm[18], lm[20])
        other_fingers_folded = (rng_angle <= self.THRESHOLDS["folded_joint_angle_max"] and 
                                pky_angle <= self.THRESHOLDS["folded_joint_angle_max"])

        # 6. Thumb Physical Anchor: Thumb Tip (4) pinned against Ring PIP Joint (15)
        thumb_to_ring_pip = self._dist(lm[4], lm[15]) / hand_size
        thumb_anchored = thumb_to_ring_pip <= self.THRESHOLDS["thumb_tip_to_ring_pip_max"]

        # 7. Integrated Execution Pipeline
        all_ok = palm_valid and angles_ok and lengths_ok and tips_together and other_fingers_folded and thumb_anchored

        # Dynamic debounce mechanism to clear noise and stabilize frame updates
        self.consecutive_correct = self.consecutive_correct + 1 if all_ok else 0
        final_ok = self.consecutive_correct >= self.required_consecutive

        if self.enable_debug_log:
            print("\n============================================================")
            print(f"📊 [U Gesture Profile] Real-Time Runtime Metrics")
            print(f"  [1. Palm Height Validity]: Depth={palm_h:.3f} -> Status: {palm_valid}")
            print(f"  [2. Twin Towers Extension]: Index={idx_angle:.1f}° | Middle={mid_angle:.1f}° -> Status: {angles_ok}")
            print(f"  [3. 🌟 Foreshortening Ratios]: Index={ratio_idx:.3f} (Req: >{self.THRESHOLDS['min_index_length_ratio']})")
            print(f"                               Middle={ratio_mid:.3f} (Req: >{self.THRESHOLDS['min_middle_length_ratio']}) -> Status: {lengths_ok}")
            print(f"  [4. Twin Cohesiveness (8 to 12)]: Dist={tip_proximity:.3f} (Req: < {self.THRESHOLDS['index_to_middle_tip_max']:.3f}) -> Status: {tips_together}")
            print(f"  [5. Folded Pillars Fold]: Ring={rng_angle:.1f}° | Pinky={pky_angle:.1f}° -> Status: {other_fingers_folded}")
            print(f"  [6. Thumb Cross Anchor (4 to 15)]: Dist={thumb_to_ring_pip:.3f} (Req: < {self.THRESHOLDS['thumb_tip_to_ring_pip_max']:.3f}) -> Status: {thumb_anchored}")
            print(f"❌/✅ Current Frame Result: {all_ok} | Stabilizer Buffer: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")

        if current_step == 1:
            return 1 if final_ok else 0
        else:
            return self.stroke_count