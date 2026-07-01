# gestures/selamat_jalan.py
# BIM Gesture: "Selamat Jalan" (Goodbye)
# Step 1: Open palm facing camera, 5 fingers fully extended and spread apart (not glued).
# Step 2: Adaptive wave tracking. Detect 2 full left-and-right waving cycles.

import math
import time
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("Selamat_Jalan")
        
        self.stroke_count = 0
        self.required_strokes = 3  # Target sets of waves to complete the exercise
        
        # Wave tracking state machine: IDLE -> WAVE_LEFT -> WAVE_RIGHT
        self.tracking_state = "IDLE"
        self.wave_cycles_completed = 0
        self.required_wave_cycles = 2  # Must wave back and forth 2 times
        
        # Anchor points
        self.anchor_x = None
        self.last_state_change_time = 0.0
        self.max_allowable_staleness = 4.0  # Reset if stuck or waving stops for 4 seconds
        self.lost_track_frames = 0
        
        self.debug = False

    def reset_dynamic_tracking(self):
        """Resets wave parameters and cycle metrics while keeping total strokes."""
        current_strokes = self.stroke_count
        super().reset_dynamic_tracking()
        self.stroke_count = current_strokes
        self.tracking_state = "IDLE"
        self.wave_cycles_completed = 0
        self.anchor_x = None
        self.lost_track_frames = 0
        self.last_state_change_time = 0.0

    def _get_distance(self, pt1, pt2):
        return math.hypot(pt1.x - pt2.x, pt1.y - pt2.y)

    def check_static_pose(self, hand):
        """Step 1: Open hand validation with comprehensive debugging telemetry."""
        lm = hand["landmarks"]
        hand_size = self._get_distance(lm[0], lm[9])
        if hand_size < 0.01: 
            if self.debug: print("[Selamat-Jalan Step1] ❌ Error: hand_size is too small.")
            return False
        
        # 1. Calculate extension ratio for all 5 fingers
        ratio_thumb = self._get_distance(lm[4], lm[0]) / self._get_distance(lm[2], lm[0])
        ratio_index = self._get_distance(lm[8], lm[0]) / self._get_distance(lm[5], lm[0])
        ratio_middle = self._get_distance(lm[12], lm[0]) / self._get_distance(lm[9], lm[0])
        ratio_ring = self._get_distance(lm[16], lm[0]) / self._get_distance(lm[13], lm[0])
        ratio_pinky = self._get_distance(lm[20], lm[0]) / self._get_distance(lm[17], lm[0])
        
        thumb_ok = ratio_thumb > 1.1
        index_ok = ratio_index > 1.2
        middle_ok = ratio_middle > 1.2
        ring_ok = ratio_ring > 1.2
        pinky_ok = ratio_pinky > 1.2
        fingers_straight = thumb_ok and index_ok and middle_ok and ring_ok and pinky_ok

        # 2. Calculate air gap distances between adjacent finger tips (normalized by hand_size)
        gap_thumb_index = self._get_distance(lm[4], lm[8]) / hand_size
        gap_index_mid = self._get_distance(lm[8], lm[12]) / hand_size
        gap_mid_ring = self._get_distance(lm[12], lm[16]) / hand_size
        gap_ring_pinky = self._get_distance(lm[16], lm[20]) / hand_size
        
        spread_thumb_idx = gap_thumb_index > 0.45
        spread_idx_mid = gap_index_mid > 0.25
        spread_mid_ring = gap_mid_ring > 0.22
        spread_ring_pky = gap_ring_pinky > 0.25
        fingers_spread = spread_thumb_idx and spread_idx_mid and spread_mid_ring and spread_ring_pky

        # 3. Verify palm orientation (Facing camera)
        palm_ratio = self._get_distance(lm[5], lm[17]) / hand_size
        # 🌟 Optimized Threshold: Lowered from 0.60 to 0.50 to fix the edge case dropouts
        palm_visible = palm_ratio > 0.50

        # =========================================================================
        # 📊 LIVE TELEMETRY LOGS (English localized diagnostic line)
        # =========================================================================
        if self.debug:
            f_status = "OK" if fingers_straight else "BENT"
            s_status = "OK" if fingers_spread else "CLOSED/GLUED"
            p_status = "OK" if palm_visible else "SIDEWAYS"
            
            print(
                f"[S-Jalan Debug] "
                f"Fingers: {f_status} (T:{ratio_thumb:.2f}/1.1, I:{ratio_index:.2f}/1.2, M:{ratio_middle:.2f}/1.2, R:{ratio_ring:.2f}/1.2, P:{ratio_pinky:.2f}/1.2) | "
                f"Gaps: {s_status} (T-I:{gap_thumb_index:.2f}/0.45, I-M:{gap_index_mid:.2f}/0.25, M-R:{gap_mid_ring:.2f}/0.22, R-P:{gap_ring_pinky:.2f}/0.25) | "
                f"Palm: {p_status} ({palm_ratio:.2f}/0.50)      ", 
                end='\r'
            )

        return fingers_straight and fingers_spread and palm_visible

    def check_gesture(self, hands_list: list, current_step: int):
        # Handle tracking drops during active hand waving
        if not hands_list:
            if current_step == 2 and self.tracking_state != "IDLE":
                self.lost_track_frames += 1
                if self.lost_track_frames > 12:
                    if self.debug: print("\n[Selamat-Jalan] ❌ Hand tracking lost mid-wave. Resetting.")
                    self.reset_dynamic_tracking()
            return 0 if current_step == 2 else False

        hand = hands_list[0]
        lm = hand["landmarks"]

        # -----------------------------------------------------
        # STEP 1: Static Open-Palm Validation
        # -----------------------------------------------------
        if current_step == 1:
            res = self.check_static_pose(hand)
            if res and self.debug:
                print("\n[Selamat-Jalan] 🎉 STEP 1 PASSED! Hand is ready.")
            return res

        # -----------------------------------------------------
        # STEP 2: Wave Trajectory Tracking
        # -----------------------------------------------------
        elif current_step == 2:
            current_time = time.time()
            self.lost_track_frames = 0
            
            # Track the Middle MCP (Landmark 9) as the stable center mass of the palm
            palm_center = lm[9]
            hand_size = self._get_distance(lm[0], lm[9])
            if hand_size < 0.01: hand_size = 0.1

            # Timeout gate to prevent endless accumulation or half-baked waving
            if self.tracking_state != "IDLE":
                if (current_time - self.last_state_change_time) > self.max_allowable_staleness:
                    if self.debug: print("\n[Selamat-Jalan] ❌ Waving sequence timed out. Resetting counters.")
                    self.reset_dynamic_tracking()
                    return self.stroke_count

            # =================================================
            # ⚓ State 1: IDLE -> Set neutral center point
            # =================================================
            if self.tracking_state == "IDLE":
                self.anchor_x = palm_center.x
                self.tracking_state = "WAVE_LEFT"  # Start by looking for a left swing
                self.wave_cycles_completed = 0
                self.last_state_change_time = current_time
                if self.debug: 
                    print(f"\n[Selamat-Jalan] 🟢 Hand locked. Starting wave detection at center X: {self.anchor_x:.3f}")

            # Calculate relative horizontal offset from initial anchor
            dx = palm_center.x - self.anchor_x
            dx_normalized = dx / hand_size  # Scaling displacement based on size
            
            # Waving threshold: Must move hand sideways by roughly ~0.35 of hand size
            wave_threshold = 0.35

            # Telemetry display for Step 2
            if self.debug:
                print(f"[WAVING TEL] Offset: {dx_normalized:+.2f} | Cycles: {self.wave_cycles_completed}/{self.required_wave_cycles}      ", end='\r')

            # =================================================
            # 🔄 Waving State Machine (Left / Right Alternation)
            # =================================================
            if self.tracking_state == "WAVE_LEFT":
                if dx_normalized < -wave_threshold:
                    self.tracking_state = "WAVE_RIGHT"
                    self.last_state_change_time = current_time

            elif self.tracking_state == "WAVE_RIGHT":
                if dx_normalized > wave_threshold:
                    self.wave_cycles_completed += 1
                    self.last_state_change_time = current_time
                    if self.debug: 
                        print(f"\n[Selamat-Jalan] 👋 Swing cycle {self.wave_cycles_completed}/{self.required_wave_cycles} recorded!")
                    
                    # Check if requirement met
                    if self.wave_cycles_completed >= self.required_wave_cycles:
                        self.stroke_count += 1
                        if self.debug:
                            print(f"[Selamat-Jalan] ✅ Complete wave gesture successful! Global Count: {self.stroke_count}/{self.required_strokes}\n")
                        
                        self.reset_dynamic_tracking()
                    else:
                        self.tracking_state = "WAVE_LEFT"

            return self.stroke_count