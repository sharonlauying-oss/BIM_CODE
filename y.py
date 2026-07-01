# ==============================================================================
# 🤟 BIM "Y" Hand Gesture Description:
# - Hand Orientation: Right hand, palm facing the camera directly.
# - Index, Middle & Ring Fingers: Base MCP joint folded downward into palm, 
#   but the distal two segments (PIP to TIP) remain fully straight.
# - Thumb: Fully extended outward to the side. Validated strictly by joint straightness angle.
# - Pinky Finger: Fully extended outward to the side. Requires Foreshortening Guard tracking.
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("Y")
        self.consecutive_correct = 0
        self.required_consecutive = 2  # Requires 2 consecutive correct frames to pass
        self.stroke_count = 1

        # 🐛 Enable explicit runtime console diagnostics
        self.enable_debug_log = False

        # 🎯 Calibrated thresholds matching the front-facing projection metrics
        self.THRESHOLDS = {
            "min_palm_height": 0.04,
            
            # ☝️ 🖕 🖕 Middle Three Fingers: Base folded, but body segments straight
            "mid_three_body_angle_min": 145.0,    # PIP-DIP-TIP angle must be flat/straight
            "mid_three_mcp_fold_max_ratio": 0.45,  # TIP to MCP distance in 2D (drawn close to palm)
            
            # 👍 Thumb Extension (Angle Check Only)
            "thumb_angle_min": 140.0,
            
            # 🤙 📏 Pinky Foreshortening Guard (Outward Extension)
            "pinky_angle_min": 170.0,
            "min_pinky_length_ratio": 0.45,      # (Dist 18 to 20) / hand_size
            
            # 👐 Flare Guard: Ensures Thumb and Pinky are wide apart
            "thumb_to_pinky_tip_min_ratio": 1.3
        }

    def _get_distance(self, p1, p2):
        # Calculate 2D distance on the screen pixel coordinate plane
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _get_world_distance(self, p1, p2):
        # Calculate 3D physical distance in meters
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _get_angle(self, p1, p2, p3):
        # Calculate the 2D angle (in degrees) formed by three landmarks at p2
        v1x, v1y = p1.x - p2.x, p1.y - p2.y
        v2x, v2y = p3.x - p2.x, p3.y - p2.y
        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0: return 0.0
        cos_angle = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_angle))

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

        # Establish dynamic scale boundary based on 2D palm heights (Wrist 0 to Middle MCP 9)
        hand_size = self._get_distance(lm[0], lm[9])
        if hand_size == 0: return 0

        palm_h = self._get_world_distance(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # 1. ☝️ 🖕 🖕 Middle Three Fingers Tracker (Index, Middle, Ring)
        def check_folded_straight_finger(mcp_idx, pip_idx, dip_idx, tip_idx):
            body_angle = self._get_angle(lm[pip_idx], lm[dip_idx], lm[tip_idx])
            straight_ok = body_angle >= self.THRESHOLDS["mid_three_body_angle_min"]
            
            tip_to_mcp_ratio = self._get_distance(lm[tip_idx], lm[mcp_idx]) / hand_size
            fold_ok = tip_to_mcp_ratio <= self.THRESHOLDS["mid_three_mcp_fold_max_ratio"]
            
            return straight_ok and fold_ok, body_angle, tip_to_mcp_ratio

        idx_ok, idx_ang, idx_rat = check_folded_straight_finger(5, 6, 7, 8)
        mid_ok, mid_ang, mid_rat = check_folded_straight_finger(9, 10, 11, 12)
        rng_ok, rng_ang, rng_rat = check_folded_straight_finger(13, 14, 15, 16)
        mid_three_ok = idx_ok and mid_ok and rng_ok

        # 2. 👍 Thumb Extension Alignment (No Foreshortening Guard used here)
        thumb_angle = self._get_angle(lm[2], lm[3], lm[4])
        thumb_ok = thumb_angle >= self.THRESHOLDS["thumb_angle_min"]

        # 3. 🤙 Pinky Foreshortening Guard & Extension Alignment
        pinky_angle = self._get_angle(lm[17], lm[18], lm[20])
        pinky_straight_ok = pinky_angle >= self.THRESHOLDS["pinky_angle_min"]
        pinky_ratio = self._get_distance(lm[18], lm[20]) / hand_size
        pinky_guard_ok = pinky_ratio >= self.THRESHOLDS["min_pinky_length_ratio"]
        pinky_ok = pinky_straight_ok and pinky_guard_ok

        # 4. 👐 Wide Flare Detection (Ensures it's a 'Y' shape, not a loose fist)
        thumb_to_pinky_ratio = self._get_distance(lm[4], lm[20]) / hand_size
        flare_ok = thumb_to_pinky_ratio >= self.THRESHOLDS["thumb_to_pinky_tip_min_ratio"]

        # 5. Integrated Execution Pipeline Verification Convergence
        valid_frame = palm_valid and mid_three_ok and thumb_ok and pinky_ok and flare_ok

        if valid_frame:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0

        result_stable = self.consecutive_correct >= self.required_consecutive

        if self.enable_debug_log:
            print("\n" + "="*60)
            print(f"📊 [Y GESTURE TELEMETRY ENGINE] - ANALYSING CURRENT FRAME")
            print("="*60)
            
            # [1] PALM GEOMETRY
            status_symbol = "✅" if palm_valid else "❌"
            print(f"{status_symbol} [1. Palm Baseline Profile]")
            print(f"   └─ World Palm Height : {palm_h:.4f} m (Threshold: >= {self.THRESHOLDS['min_palm_height']})")
            
            # [2] MIDDLE THREE FINGERS LAYER
            status_symbol = "✅" if mid_three_ok else "❌"
            print(f"{status_symbol} [2. Middle Three Layer (Base Folded + Body Straight)]")
            print(f"   ├─ Index  : Angle = {idx_ang:>5.1f}° (Req: >{self.THRESHOLDS['mid_three_body_angle_min']}°) | TipToMcpRatio = {idx_rat:.3f} (Req: <{self.THRESHOLDS['mid_three_mcp_fold_max_ratio']})")
            print(f"   ├─ Middle : Angle = {mid_ang:>5.1f}° (Req: >{self.THRESHOLDS['mid_three_body_angle_min']}°) | TipToMcpRatio = {mid_rat:.3f} (Req: <{self.THRESHOLDS['mid_three_mcp_fold_max_ratio']})")
            print(f"   └─ Ring   : Angle = {rng_ang:>5.1f}° (Req: >{self.THRESHOLDS['mid_three_body_angle_min']}°) | TipToMcpRatio = {rng_rat:.3f} (Req: <{self.THRESHOLDS['mid_three_mcp_fold_max_ratio']})")
            
            # [3] THUMB STRAIGHTNESS CHECK
            status_symbol = "✅" if thumb_ok else "❌"
            print(f"{status_symbol} [3. Thumb Straightness Check]")
            print(f"   └─ Flexion Angle : {thumb_angle:.1f}°   (Threshold: >= {self.THRESHOLDS['thumb_angle_min']}°)")
            
            # [4] PINKY FORESHORTENING GUARD
            status_symbol = "✅" if pinky_ok else "❌"
            print(f"{status_symbol} [4. Pinky Extension & Foreshortening Guard]")
            print(f"   ├─ Flexion Angle : {pinky_angle:.1f}°   (Threshold: >= {self.THRESHOLDS['pinky_angle_min']}°)")
            print(f"   └─ Project Ratio : {pinky_ratio:.3f}    (Guard Minimum: >= {self.THRESHOLDS['min_pinky_length_ratio']})")

            # [5] WIDE FLARE SEPARATION
            status_symbol = "✅" if flare_ok else "❌"
            print(f"{status_symbol} [5. Wide Span Flare Guard (Thumb Tip 4 -> Pinky Tip 20)]")
            print(f"   └─ Tip Span Ratio: {thumb_to_pinky_ratio:.3f}    (Required Aperture: >= {self.THRESHOLDS['thumb_to_pinky_tip_min_ratio']})")
            
            print("-"*60)
            final_status = "🟢 PASSED CRITERIA" if valid_frame else "🔴 FAILED CRITERIA"
            print(f"🏁 Frame Evaluation  : {final_status}")
            print(f"🔄 Debounce Stabilizer: {self.consecutive_correct} / {self.required_consecutive} consecutive frames")
            print("="*60 + "\n")

        return result_stable if current_step == 1 else self.stroke_count