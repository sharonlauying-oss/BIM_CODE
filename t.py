# ==============================================================================
# 🤟 BIM "T" Gesture Description:
# - Hand Orientation: Pinky side facing the camera, tilted downward at 45 degrees.
# - Pinky, Ring & Middle Fingers: Fully clenched into the palm.
# - Thumb: Landmark 1 close to pinky landmark 20; Landmark 3 close to middle finger landmark 10; Landmark 4 close to index finger landmark 5.
# - Index Finger: Second segment presses against the thumb, and the finger is bent within a certain range.
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("T")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        self.stroke_count = 1

        # 🐛 Enable explicit runtime console diagnostics
        self.enable_debug_log = False 

        # 🎯 Geometric thresholds calibrated for a side-profile 45° hand tilt
        self.THRESHOLDS = {
            "min_palm_height": 0.03,
            
            # ☝️ Index Finger Flexion Range (Matches your telemetry perfectly: 66.0° to 89.8°)
            "index_flex_min": 50.0,
            "index_flex_max": 95.0,

            # ✊ Core Clenched Fist Group (Middle, Ring, Pinky)
            # Refactored: Set to 130.0° to bypass 2D projection flattening caused by the 45° camera tilt
            "curled_joint_angle_max": 130.0,
            
            # 👍 Three-Point Thumb Anchor Locks (Normalized Euclidean distance ratios)
            "thumb_tip_to_index_mcp": 0.38,     # Anchor A: Map to log telemetry (0.266 ~ 0.360)
            "thumb_joint_to_middle_mcp": 0.20,   # Anchor B: Most stable structural landmark (0.068 ~ 0.188)
            "thumb_base_to_pinky_zone": 0.52     # Anchor C: Map to log telemetry (0.374 ~ 0.506)
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

        # Establish dynamic scale boundary based on palm profile
        hand_size = self._dist(lm[0], lm[9])
        if hand_size == 0: return 0

        # 1. Structural Baseline: Palm Verification 
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # 2. Top Enclosure Shield: Index Finger Flexion Evaluation
        index_angle = self._angle(lm[5], lm[6], lm[8])
        index_ok = (self.THRESHOLDS["index_flex_min"] <= index_angle <= self.THRESHOLDS["index_flex_max"])

        # 3. Main Fist Cavity: Tight Clench Check (Middle, Ring, Pinky)
        mid_angle = self._angle(lm[10], lm[11], lm[12])
        rng_angle = self._angle(lm[14], lm[15], lm[16])
        pky_angle = self._angle(lm[18], lm[19], lm[20])

        fist_core_ok = (mid_angle < self.THRESHOLDS["curled_joint_angle_max"] and
                        rng_angle < self.THRESHOLDS["curled_joint_angle_max"] and
                        pky_angle < self.THRESHOLDS["curled_joint_angle_max"])

        # 4. 🌟 Three-Point Thumb Interlock Verification
        # Anchor A: Thumb Tip (4) sits snugly near Index MCP Joint Base (5)
        dist_4_to_5 = self._dist(lm[4], lm[5]) / hand_size
        anchor_a_ok = dist_4_to_5 <= self.THRESHOLDS["thumb_tip_to_index_mcp"]

        # Anchor B: Thumb IP Joint (3) locks down tightly over Middle MCP Joint Base (10)
        dist_3_to_10 = self._dist(lm[3], lm[10]) / hand_size
        anchor_b_ok = dist_3_to_10 <= self.THRESHOLDS["thumb_joint_to_middle_mcp"]

        # Anchor C: Thumb Base/CMC (2) drops close toward Pinky Finger Zone/Tip (20)
        dist_2_to_20 = self._dist(lm[2], lm[20]) / hand_size
        anchor_c_ok = dist_2_to_20 <= self.THRESHOLDS["thumb_base_to_pinky_zone"]

        thumb_anchors_ok = anchor_a_ok and anchor_b_ok and anchor_c_ok

        # 5. Integrated Execution Pipeline
        all_ok = palm_valid and index_ok and fist_core_ok and thumb_anchors_ok

        # Dynamic debounce mechanism to clear noise and stabilize frame updates
        self.consecutive_correct = self.consecutive_correct + 1 if all_ok else 0
        final_ok = self.consecutive_correct >= self.required_consecutive

        if self.enable_debug_log:
            print("\n============================================================")
            print(f"📊 [T Gesture Profile] Real-Time Runtime Metrics")
            print(f"  [1. Index Flexion]: Angle={index_angle:.1f}° -> Status: {index_ok}")
            print(f"  [2. Core Fist Flexion]: Mid={mid_angle:.1f}° | Rng={rng_angle:.1f}° | Pky={pky_angle:.1f}° -> Status: {fist_core_ok}")
            print(f"  [3. Thumb Anchor A (4 to 5)]: Dist={dist_4_to_5:.3f} (Req: < {self.THRESHOLDS['thumb_tip_to_index_mcp']:.3f}) -> Status: {anchor_a_ok}")
            print(f"  [4. Thumb Anchor B (3 to 10)]: Dist={dist_3_to_10:.3f} (Req: < {self.THRESHOLDS['thumb_joint_to_middle_mcp']:.3f}) -> Status: {anchor_b_ok}")
            print(f"  [5. Thumb Anchor C (2 to 20)]: Dist={dist_2_to_20:.3f} (Req: < {self.THRESHOLDS['thumb_base_to_pinky_zone']:.3f}) -> Status: {anchor_c_ok}")
            print(f"❌/✅ Current Frame Result: {all_ok} | Stabilizer Buffer: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")

        if current_step == 1:
            return 1 if final_ok else 0
        else:
            return self.stroke_count