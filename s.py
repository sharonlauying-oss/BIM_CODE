# ==============================================================================
# 🤟 BIM "S" Gesture Description:
# - Hand Orientation: Palm facing the CAMERA, similar to gesture A.
# - Four Fingers (Index, Middle, Ring, Pinky): Fully curled and clenched tightly into the palm.
# - Thumb: Thumb tip (landmark 4) presses close to ring finger landmark 14.
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("S")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        self.stroke_count = 1

        # 🐛 Enable explicit runtime console diagnostics
        self.enable_debug_log = False 

        # 🎯 Re-calibrated thresholds to account for front-facing fist depth flattening
        self.THRESHOLDS = {
            "min_palm_height": 0.03,
            
            # ✊ Perspective-Resilient Fist Group
            # explanation: Index stays clean (<15°), but Middle/Ring/Pinky projection flattens up to 180°.
            # We allow the flattened profile captured in your telemetry to bypass false rejects.
            "index_joint_angle_max": 45.0,
            
            # 👍 ASL S-Thumb Trajectory Configuration
            "thumb_straight_min": 140.0,   
            
            # 🌟 Core S-Sign Lock: Distance between Thumb Tip (4) and Ring MCP (14)
            "thumb_to_ring_mcp_ratio": 0.22 
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

        # 1. Base Palm Verification
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # 2. Four Fingers Curled/Clenched State (Fist Validation)
        idx_angle = self._angle(lm[5], lm[6], lm[8])
        mid_angle = self._angle(lm[10], lm[11], lm[12])
        rng_angle = self._angle(lm[14], lm[15], lm[16])
        pky_angle = self._angle(lm[18], lm[19], lm[20])

        # explanation: The Index finger is unobstructed and stays <45°. Middle, Ring, and Pinky 
        # angles are bypassed here because thumb compression forces them into a flat 2D projection.
        angles_ok = (idx_angle < self.THRESHOLDS["index_joint_angle_max"])

        # Y-Axis Drop Check: Verifies tips are tracking lower than root joints (Y-Axis down logic)
        # This is your primary structural anchor since it is 100% reliable in your logs.
        y_drop_ok = (lm[8].y > lm[5].y and 
                     lm[12].y > lm[10].y and 
                     lm[16].y > lm[14].y and 
                     lm[20].y > lm[18].y)

        fist_ok = angles_ok and y_drop_ok

        # 3. Thumb Extension & Orientation Evaluation
        thumb_angle = self._angle(lm[2], lm[3], lm[4])
        thumb_extended_ok = thumb_angle >= self.THRESHOLDS["thumb_straight_min"]

        # 4. S-Thumb Anchor Lock: Thumb Tip (4) tightly pressed against Ring MCP (14)
        thumb_to_ring_dist = self._dist(lm[4], lm[14]) / hand_size
        thumb_anchor_ok = thumb_to_ring_dist <= self.THRESHOLDS["thumb_to_ring_mcp_ratio"]

        # 5. Integrated Execution Decision
        all_ok = palm_valid and fist_ok and thumb_extended_ok and thumb_anchor_ok

        # Dynamic Debounce Logic
        self.consecutive_correct = self.consecutive_correct + 1 if all_ok else 0
        final_ok = self.consecutive_correct >= self.required_consecutive

        if self.enable_debug_log:
            print("\n============================================================")
            print(f"📊 [S Gesture Profile] Real-Time Runtime Metrics")
            print(f"  [1. Fist Joint Flexion]: Idx={idx_angle:.1f}° | Mid={mid_angle:.1f}° | Rng={rng_angle:.1f}° | Pky={pky_angle:.1f}° -> Status: {angles_ok}")
            print(f"  [2. Fist Y-Axis Drop Check]: {y_drop_ok}")
            print(f"  [3. Thumb Straight Angle]: {thumb_angle:.1f}° (Req: > {self.THRESHOLDS['thumb_straight_min']}°) -> Status: {thumb_extended_ok}")
            print(f"  [4. S-Thumb Anchor Lock (4 to 14)]: Normalized Dist={thumb_to_ring_dist:.3f} (Req: < {self.THRESHOLDS['thumb_to_ring_mcp_ratio']:.3f}) -> Status: {thumb_anchor_ok}")
            print(f"❌/✅ Current Frame Result: {all_ok} | Stabilizer Buffer: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")

        if current_step == 1:
            return 1 if final_ok else 0
        else:
            return self.stroke_count