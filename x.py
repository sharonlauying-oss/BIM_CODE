# ==============================================================================
# 🤟 BIM "X" Hand Gesture Description:
# - Hand Orientation: Right hand turned sideways facing LEFT, tilted ~45° downward.
# - Index Finger: Curled forward into a distinct "hook" shape (semi-bent).
# - Middle, Ring, & Pinky Fingers: Fully tucked and tightly curled into the palm cavity.
# - Thumb: Pressed flat against the outer folded ring finger PIP joint layer (Landmark 14).
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("X")
        self.consecutive_correct = 0
        self.required_consecutive = 2  # Requires 2 consecutive correct frames to pass
        self.stroke_count = 1

        # 🐛 Enable explicit runtime console diagnostics
        self.enable_debug_log = False

        # 🎯 Calibrated strictly against your actual live telemetry data stream
        self.THRESHOLDS = {
            "min_palm_height": 0.06,
            
            # ☝️ Index Hook Bounding Box
            "index_hook_angle_min": 75.0,
            "index_hook_angle_max": 100.0,
            "index_hook_ratio_min": 0.22,
            "index_hook_ratio_max": 0.40,
            
            # ✊ Folded Middle, Ring, Pinky Limits (Calibrated to your ~115° data)
            "bent_finger_max_angle": 125.0, 
            "root_proximity_max": 0.52,      # Relaxed from 0.26 based on your 0.469 real metrics
            "tip_to_mcp_max_ratio": 0.55,    # Alternative guard: ensure tip is drawn near root
            
            # 👍 Thumb-to-Ring-Finger Anchor Locking Parameters (Landmark 4 to 14)
            "thumb_to_ring_pip_max": 0.03,   # Tightened to 0.04m since your real data is a perfect 0.025m
            
            # 🖐️ Sideways Palm Profile Aspect Ratios
            "palm_ratio_min": 0.25,
            "palm_ratio_max": 0.75
        }

    def _get_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _get_world_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _get_angle(self, p1, p2, p3):
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

        # Establish dynamic scale boundary based on 2D palm heights
        hand_size = self._get_distance(lm[0], lm[9])
        if hand_size == 0: return 0

        palm_h = self._get_world_distance(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # 1. ☝️ Index Hook Foreshortening Guard & Angle Logic
        idx_angle = self._get_angle(lm[5], lm[6], lm[8])
        idx_angle_ok = self.THRESHOLDS["index_hook_angle_min"] < idx_angle < self.THRESHOLDS["index_hook_angle_max"]
        
        ratio_idx = self._get_distance(lm[6], lm[8]) / hand_size
        idx_length_ok = self.THRESHOLDS["index_hook_ratio_min"] < ratio_idx < self.THRESHOLDS["index_hook_ratio_max"]
        index_ok = idx_angle_ok and idx_length_ok

        # 2. ✊ Folded Pile Constraints (Middle, Ring, Pinky)
        finger_diagnostics = {}

        def check_folded_finger(name, mcp_idx, pip_idx, dip_idx, tip_idx):
            root_ratio = self._get_distance(lm[mcp_idx], lm[pip_idx]) / hand_size
            root_close = root_ratio < self.THRESHOLDS["root_proximity_max"]
            
            front_angle = self._get_angle(lm[pip_idx], lm[dip_idx], lm[tip_idx])
            front_bent = front_angle < self.THRESHOLDS["bent_finger_max_angle"]
            
            # 🔥 FIX: Replace the faulty orientation check with a 2D Proximity Guard.
            # Verifies that the finger tip has drawn close to its own base MCP root in 2D space.
            tip_to_mcp_ratio = self._get_distance(lm[tip_idx], lm[mcp_idx]) / hand_size
            tip_tucked_close = tip_to_mcp_ratio < self.THRESHOLDS["tip_to_mcp_max_ratio"]
            
            passed = root_close and front_bent and tip_tucked_close
            
            finger_diagnostics[name] = {
                "root_ratio": root_ratio, "root_close": root_close,
                "front_angle": front_angle, "front_bent": front_bent,
                "tip_mcp_ratio": tip_to_mcp_ratio, "tip_tucked": tip_tucked_close, 
                "passed": passed
            }
            return passed

        mid_ok = check_folded_finger("Middle", 9, 10, 11, 12)
        ring_ok = check_folded_finger("Ring", 13, 14, 15, 16)
        pinky_ok = check_folded_finger("Pinky", 17, 18, 19, 20)
        fingers_folded_ok = mid_ok and ring_ok and pinky_ok

        # 3. 👍 Thumb Physical Anchor Constraint
        thumb_to_ring_pip_dist = self._get_world_distance(wm[4], wm[14])
        thumb_anchor_ok = thumb_to_ring_pip_dist < self.THRESHOLDS["thumb_to_ring_pip_max"]

        # 4. 🖐️ Sideways Palm Facing Logic Check
        palm_w = self._get_distance(lm[5], lm[17])
        palm_ratio = palm_w / hand_size
        palm_orientation_ok = self.THRESHOLDS["palm_ratio_min"] < palm_ratio < self.THRESHOLDS["palm_ratio_max"]

        # 5. Integrated Execution Pipeline Verification Convergence
        valid_frame = palm_valid and index_ok and fingers_folded_ok and thumb_anchor_ok and palm_orientation_ok

        if valid_frame:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0

        result_stable = self.consecutive_correct >= self.required_consecutive

        if self.enable_debug_log:
            print("\n" + "="*60)
            print(f"📊 [X GESTURE TELEMETRY ENGINE] - ANALYSING CURRENT FRAME")
            print("="*60)
            
            status_symbol = "✅" if (palm_valid and palm_orientation_ok) else "❌"
            print(f"{status_symbol} [1. Palm Profile Alignment]")
            print(f"   ├─ World Height : {palm_h:.4f} m (Threshold: >= {self.THRESHOLDS['min_palm_height']})")
            print(f"   └─ Sideways Ratio: {palm_ratio:.3f}   (Allowed Window: {self.THRESHOLDS['palm_ratio_min']} ~ {self.THRESHOLDS['palm_ratio_max']})")
            
            status_symbol = "✅" if index_ok else "❌"
            print(f"{status_symbol} [2. Index Hook Guard (Landmarks 5->6->8)]")
            print(f"   ├─ Flexion Angle: {idx_angle:.1f}°   (Allowed Window: {self.THRESHOLDS['index_hook_angle_min']}° ~ {self.THRESHOLDS['index_hook_angle_max']}°)")
            print(f"   └─ Project Ratio: {ratio_idx:.3f}   (Allowed Window: {self.THRESHOLDS['index_hook_ratio_min']} ~ {self.THRESHOLDS['index_hook_ratio_max']})")
            
            status_symbol = "✅" if fingers_folded_ok else "❌"
            print(f"{status_symbol} [3. Folded Base Stack (Middle/Ring/Pinky)]")
            for f_name, metrics in finger_diagnostics.items():
                f_status = "  ├─" if f_name != "Pinky" else "  └─"
                pass_icon = "✓" if metrics["passed"] else "𐄂"
                print(f"   {f_status} {f_name:<6} [{pass_icon}] -> Angle: {metrics['front_angle']:>5.1f}° (Req: <{self.THRESHOLDS['bent_finger_max_angle']}°) | ProxyRatio: {metrics['root_ratio']:.3f} (Req: <{self.THRESHOLDS['root_proximity_max']}) | TipToMcpRatio: {metrics['tip_mcp_ratio']:.3f} (Req: <{self.THRESHOLDS['tip_to_mcp_max_ratio']})")

            status_symbol = "✅" if thumb_anchor_ok else "❌"
            print(f"{status_symbol} [4. Thumb Physics Lock Anchor (Tip 4 -> Ring PIP 14)]")
            print(f"   └─ 3D Space Distance: {thumb_to_ring_pip_dist:.4f} m (Lock Target Threshold: < {self.THRESHOLDS['thumb_to_ring_pip_max']} m)")
            
            print("-"*60)
            final_status = "🟢 PASSED CRITERIA" if valid_frame else "🔴 FAILED CRITERIA"
            print(f"🏁 Frame Evaluation  : {final_status}")
            print(f"🔄 Debounce Stabilizer: {self.consecutive_correct} / {self.required_consecutive} consecutive frames")
            print("="*60 + "\n")

        return result_stable if current_step == 1 else self.stroke_count