# gestures/hello.py
# BIM Word "Hello"
# Step 1: Hand fully open. ALL fingers strictly straight AND fully pressed together (no spacing).
# Step 2: Adaptive Forward Push & Down-Right Translation with real-time dx, dy debugger.

import math
import time
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("Hello")
        
        # 🎯 Persistent Stroke Counter
        self.stroke_count = 0
        self.required_strokes = 3  
        
        # Trajectory State Machine: IDLE -> START_PUSH
        self.tracking_state = "IDLE"
        
        # Initial Reference Points (Snapshot taken at the start of the stroke)
        self.p_start_2d = None     # (x, y) coordinates of middle finger MCP (Joint 9)
        self.start_hand_size = 0.0 # Base hand size reference at start point
        
        # ⏳ Fluidity Timeout Control System
        self.last_state_change_time = 0.0
        self.max_allowable_staleness = 2.0  # Must complete the diagonal forward push within 2 seconds
        
        self.debug = False

    def reset_dynamic_tracking(self):
        """Resets the tracking state machine for a single stroke while maintaining the total accumulated stroke_count."""
        self.tracking_state = "IDLE"
        self.p_start_2d = None
        self.start_hand_size = 0.0
        self.last_state_change_time = 0.0

    def _get_distance(self, pt1, pt2):
        return math.hypot(pt1.x - pt2.x, pt1.y - pt2.y)

    def _calculate_angle_3pt(self, p1, p2, p3):
        """Calculates the angle formed by three points (180 degrees indicates perfectly straight)."""
        a = math.hypot(p1.x - p2.x, p1.y - p2.y)
        b = math.hypot(p3.x - p2.x, p3.y - p2.y)
        c = math.hypot(p1.x - p3.x, p1.y - p3.y)
        if a * b == 0: return 0.0
        cos_val = (a*a + b*b - c*c) / (2 * a * b)
        cos_val = max(-1.0, min(1.0, cos_val))
        return math.degrees(math.acos(cos_val))

    def _calculate_palm_facing_x(self, lm):
        """Computes the X component of the palm's 3D normal vector via bone cross product."""
        v_up = [lm[9].x - lm[0].x, lm[9].y - lm[0].y, lm[9].z - lm[0].z]
        v_right = [lm[17].x - lm[5].x, lm[17].y - lm[5].y, lm[17].z - lm[5].z]
        nx = v_up[1] * v_right[2] - v_up[2] * v_right[1]
        length = math.sqrt(nx*nx + (v_up[2]*v_right[0] - v_up[0]*v_right[2])**2 + (v_up[0]*v_right[1] - v_up[1]*v_right[0])**2)
        return (nx / length) if length != 0 else 0.0

    def check_finger_strictly_straight(self, lm, tip_idx, dip_idx, pip_idx, mcp_idx):
        """[Strict Extension Metric]: Intercepts the gesture instantly if extension ratio drops below 0.96."""
        seg1 = self._get_distance(lm[tip_idx], lm[dip_idx])
        seg2 = self._get_distance(lm[dip_idx], lm[pip_idx])
        seg3 = self._get_distance(lm[pip_idx], lm[mcp_idx])
        total_bone_length = seg1 + seg2 + seg3
        
        straight_dist = self._get_distance(lm[tip_idx], lm[mcp_idx])
        if total_bone_length == 0: return False, 0.0
        
        ratio = straight_dist / total_bone_length
        return ratio > 0.96, ratio

    def check_static_pose(self, hand):
        """Step 1: Validates static pose posture, palm orientation, AND strict finger-to-finger adjacency."""
        lm = hand["landmarks"]
        wrist = lm[0]
        
        hand_size = self._get_distance(wrist, lm[9])
        if hand_size == 0: return False
            
        # 1. Evaluate absolute extension for the 4 long fingers
        idx_ok, _ = self.check_finger_strictly_straight(lm, 8, 7, 6, 5)
        mid_ok, _ = self.check_finger_strictly_straight(lm, 12, 11, 10, 9)
        rng_ok, _ = self.check_finger_strictly_straight(lm, 16, 15, 14, 13)
        pky_ok, _ = self.check_finger_strictly_straight(lm, 20, 19, 18, 17)
        
        # 2. Evaluate absolute extension for the thumb (Angle must exceed 160 degrees)
        thumb_angle = self._calculate_angle_3pt(lm[4], lm[3], lm[2])
        thumb_ok = thumb_angle > 160.0
        
        fingers_all_straight = idx_ok and mid_ok and rng_ok and pky_ok and thumb_ok
        
        # 3. 🔥【新增核心】：相邻手指绝对并拢度检测 (食指-中指-无名指-小指相互紧贴)
        # 限制指尖间距不能超过手掌大小的 0.28 倍 (你可以根据测试严格度在 0.25-0.30 之间微调)
        proximity_threshold = hand_size * 0.28
        
        idx_mid_dist = self._get_distance(lm[8], lm[12])   # 食指-中指
        mid_rng_dist = self._get_distance(lm[12], lm[16])  # 中指-无名指
        rng_pky_dist = self._get_distance(lm[16], lm[20])  # 无名指-小指
        
        fingers_pressed_together = (idx_mid_dist < proximity_threshold and 
                                    mid_rng_dist < proximity_threshold and 
                                    rng_pky_dist < proximity_threshold)
        
        # 4. Validate palm orientation alignment vector (0.40 to 1.00)
        palm_x = self._calculate_palm_facing_x(lm)
        facing_left_ok = (0.40 <= palm_x <= 1.00)
        
        # 📊 Only stream Step 1 telemetry when IDLE to avoid excessive logging overhead
        if self.debug and self.tracking_state == "IDLE":
            finger_ok = "OK" if fingers_all_straight else "❌"
            # 增加并拢状态的实时调试打印
            press_ok = "OK" if fingers_pressed_together else f"❌(隙缝最大: {max(idx_mid_dist, mid_rng_dist, rng_pky_dist)/hand_size:.2f}x)"
            angle_status = f"Palm-X: {palm_x:.2f} -> {'PASS' if facing_left_ok else '❌'}"
            print(f"[Hello-Step1] Straight:{finger_ok} | Pressed:{press_ok} | {angle_status}", end='\r')

        return fingers_all_straight and fingers_pressed_together and facing_left_ok

    def check_gesture(self, hands_list: list, current_step: int):
        if not hands_list:
            if self.tracking_state != "IDLE":
                if self.debug: print("\n[Hello-Engine] ❌ Frame Error: Hand lost tracking. Resetting trajectory pipeline.")
                self.reset_dynamic_tracking()
            return 0 if current_step == 2 else False

        hand = hands_list[0]
        pose_ok = self.check_static_pose(hand)

        if current_step == 1:
            return pose_ok

        elif current_step == 2:
            current_time = time.time()
            track_pt = hand["landmarks"][9] # Middle finger MCP joint used for tracking translation
            hand_size = self._get_distance(hand["landmarks"][0], hand["landmarks"][9])

            if self.tracking_state != "IDLE":
                if (current_time - self.last_state_change_time) > self.max_allowable_staleness:
                    if self.debug: print("\n[Hello-Engine] ❌ Execution Error: Motion stalled during push phase. Timeout reset.")
                    self.reset_dynamic_tracking()
                    return self.stroke_count

            # =================================================
            # STATE 1: IDLE -> Lock Origin Reference
            # =================================================
            if self.tracking_state == "IDLE":
                if pose_ok:  
                    self.p_start_2d = (track_pt.x, track_pt.y)
                    self.start_hand_size = hand_size
                    self.tracking_state = "START_PUSH"
                    self.last_state_change_time = current_time
                    if self.debug: print("\n[Hello-Engine] 🟢 Anchor Locked. Smoothly push forward and down-right...")

            # =================================================
            # STATE 2: START_PUSH -> Track & Print Real-Time Data Streams
            # =================================================
            elif self.tracking_state == "START_PUSH":
                dx = track_pt.x - self.p_start_2d[0]  # Positive = Rightwards
                dy = track_pt.y - self.p_start_2d[1]  # Positive = Downwards
                
                # 📡 Adaptive Depth Verification (Z-Axis Push)
                growth_threshold = 1.02 if self.start_hand_size < 0.08 else 1.05
                current_growth = hand_size / self.start_hand_size
                is_pushed_forward = hand_size > (self.start_hand_size * growth_threshold)
                
                # 2D Spatial Translation Bounds
                minx_travel = hand_size * 0.35
                miny_travel = hand_size * 0.25
                is_moving_right_down = (dx > minx_travel) and (dy > miny_travel)

                # 📊 [High-Frequency Telemetry Tracking Stream]
                if self.debug:
                    x_status = f"X-Move(R): {dx:+.4f}/{minx_travel:.4f} [{'OK' if dx > minx_travel else '..'}]"
                    y_status = f"Y-Move(D): {dy:+.4f}/{miny_travel:.4f} [{'OK' if dy > miny_travel else '..'}]"
                    z_status = f"Z-Push(G): {current_growth:.2f}x/{growth_threshold:.2f}x [{'OK' if is_pushed_forward else '..'}]"
                    print(f"[Hello-Tracking] {x_status} | {y_status} | {z_status}", end='\r')

                # 🎉 Stroke Verification Trigger Condition
                if is_pushed_forward and is_moving_right_down:
                    self.stroke_count += 1
                    if self.debug:
                        print(f"\n[Hello-Engine] 🎉 Gesture Executed Fluently! Progress Saved: {self.stroke_count}/{self.required_strokes}")
                    self.reset_dynamic_tracking()
                    
                else:
                    # 🔥 Direction Deviation Interceptor
                    strict_misdirection = hand_size * 0.35
                    if dx < -strict_misdirection:
                        if self.debug: print(f"\n[Hello-Engine] ❌ Deviation Error: Backwards left drift detected (dx:{dx:.4f}). Resetting.")
                        self.reset_dynamic_tracking()
                    elif dy < -strict_misdirection:
                        if self.debug: print(f"\n[Hello-Engine] ❌ Deviation Error: Upward vertical lifting detected (dy:{dy:.4f}). Resetting.")
                        self.reset_dynamic_tracking()

            return self.stroke_count