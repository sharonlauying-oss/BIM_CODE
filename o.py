# 🤟 BIM "O" Gesture Description:
# - Hand Orientation: Palm facing sideways, showing a clear side-profile posture.
# - Fingers (Index to Pinky): Curved forward into an arch pointing LEFT or TOP-LEFT.
# - Thumb: Curved upward to meet the tips of the opposing fingers.
# - Gesture Shape: All five fingertips gather close together, creating a clear,
#   hollow circular "O" profile facing the side.

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("O")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        self.stroke_count = 1

        # 🐛 Debug log flag
        self.enable_debug_log = False 

        # 🎯 Dynamic thresholds covering both horizontal left and diagonal top-left postures
        self.THRESHOLDS = {
            "min_palm_height": 0.05,      # 3D absolute height (filters distant/noise hands)
            "palm_ratio_min": 0.35,       # Allow for natural side-profile widths
            "palm_ratio_max": 0.95,       # Max width/height ratio
            "thumb_bend_max": 180,        # Max thumb bending angle
            "tip_near_ratio": 0.30,       # Precise fingertip clumping proximity
            "finger_bend_max": 175,       # Prevent completely flat extended fingers
            "joint_bend_diff": -5,        # Forces true circular arch flex
            "hollow_space_min": 0.12,     # Ensures a hollow round shape exists
            
            # 🌟 Precision Direction Angles (0°=Right, 90°=Up, 180°=Left, 270°=Down)
            # explain: Configured to accept natural left to top-left pointing postures (95° to 195°)
            # while fully banning false-positive upright configurations (~90°).
            "hand_direction_angle_min": 95, 
            "hand_direction_angle_max": 195  
        }

    def _dist(self, p1, p2):
        # explain: Calculate 2D Euclidean distance between two screen-space points
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _wdist(self, p1, p2):
        # explain: Calculate 3D Euclidean distance using world landmarks for physical scaling
        return math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)

    def _angle(self, p1, p2, p3):
        # explain: Compute the inner angle at joint p2 formed by lines p1-p2 and p3-p2
        v1x = p1.x - p2.x
        v1y = p1.y - p2.y
        v2x = p3.x - p2.x
        v2y = p3.y - p2.y

        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)

        if mag1 == 0 or mag2 == 0:
            return 0

        val = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(val))

    def _check_finger_bend_shape(self, lm, mcp_idx, pip_idx, dip_idx):
        # explain: Evaluate the derivative angle curve profile (PIP angle relative to MCP angle)
        mcp_angle = self._angle(lm[mcp_idx-1], lm[mcp_idx], lm[pip_idx])
        pip_angle = self._angle(lm[mcp_idx], lm[pip_idx], lm[dip_idx])
        angle_diff = pip_angle - mcp_angle
        is_bent = angle_diff >= self.THRESHOLDS["joint_bend_diff"]
        return is_bent, (mcp_angle, pip_angle, angle_diff)

    def reset_dynamic_tracking(self):
        # explain: Clean out temporal state filters and dynamic debounce variables
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        # explain: Instantly fail and tear down stability frames if tracking is lost
        if not hands_list:
            self.consecutive_correct = 0
            return 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # explain: Calculate full dynamic tracking norm bound utilizing Wrist (0) to Middle MCP (9)
        hand_size = self._dist(lm[0], lm[9])
        if hand_size == 0:
            return 0

        # 1. Palm Tracking Verification
        # explain: Extract absolute physical size context to reject distant tracking clutter
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # explain: Check aspect ratio between structural knuckles to secure lateral perspective focus
        palm_w = self._wdist(wm[5], wm[17])
        palm_ratio = palm_w / palm_h if palm_h > 0 else 0
        face_valid = self.THRESHOLDS["palm_ratio_min"] <= palm_ratio <= self.THRESHOLDS["palm_ratio_max"]

        # 2. 🌟 Strict 2D Vector Direction Lock (Blocks Upward False Positives, Allows Left & Top-Left)
        # explain: Map spatial projection vector matching wrist root (0) up to middle knuckle (9)
        dx = lm[9].x - lm[0].x
        dy = -(lm[9].y - lm[0].y) 
        
        # explain: Compute polar coordinate angle; ensures alignment tracks in the true leftward hemisphere
        global_angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # explain: Enforce the global vector boundary constraint ruleset
        direction_angle_ok = (self.THRESHOLDS["hand_direction_angle_min"] 
                              <= global_angle 
                              <= self.THRESHOLDS["hand_direction_angle_max"])

        # 3. Thumb Analysis
        # explain: Evaluate thumb inner posture bend profile to guarantee curvature activation
        thumb_angle = self._angle(lm[2], lm[3], lm[4])
        thumb_bend = thumb_angle < self.THRESHOLDS["thumb_bend_max"]

        # 4. Fingertip Clumping Proximity
        # explain: Loop through all tips and ensure they collapse tight against the thumb landmark
        t_tip = lm[4]
        tips = [lm[8], lm[12], lm[16], lm[20]]
        tip_distances = [self._dist(t_tip, tip) for tip in tips]
        normalized_tip_dists = [d / hand_size for d in tip_distances]
        tip_valid = all(nd < self.THRESHOLDS["tip_near_ratio"] for nd in normalized_tip_dists)

        # 5. Anti-Flat Hand Hollow Verification
        # explain: Check hollow gap between thumb tip and middle knuckle to exclude flat hand mimics
        hollow_depth = self._dist(lm[9], lm[4]) / hand_size
        hollow_ok = hollow_depth > self.THRESHOLDS["hollow_space_min"]

        # 6. Segmented Finger Shape Check
        # explain: Track spatial arc progression properties across each unique finger segment chain
        idx_ok, idx_shapes = self._check_finger_bend_shape(lm, 5, 6, 7)
        mid_ok, mid_shapes = self._check_finger_bend_shape(lm, 9, 10, 11)
        ring_ok, ring_shapes = self._check_finger_bend_shape(lm, 13, 14, 15)
        pinky_ok, pky_shapes = self._check_finger_bend_shape(lm, 17, 18, 19)
        finger_shape_ok = idx_ok and mid_ok and ring_ok and pinky_ok

        # 7. Overall Finger Compression Check
        # explain: Force structural limits on full extension properties across tip terminals
        idx_angle = self._angle(lm[6], lm[7], lm[8])
        mid_angle = self._angle(lm[10], lm[11], lm[12])
        ring_angle = self._angle(lm[14], lm[15], lm[16])
        pky_angle = self._angle(lm[18], lm[19], lm[20])
        fingers_ok = all([
            idx_angle < self.THRESHOLDS["finger_bend_max"],
            mid_angle < self.THRESHOLDS["finger_bend_max"],
            ring_angle < self.THRESHOLDS["finger_bend_max"],
            pky_angle < self.THRESHOLDS["finger_bend_max"]
        ])

        # ⚖️ Ultimate Condition Combination
        # explain: Combine directional gating, shape checks, and clustering logic into a final boolean pass
        all_ok = (palm_valid and face_valid and direction_angle_ok and 
                  thumb_bend and tip_valid and hollow_ok and fingers_ok and finger_shape_ok)

        # explain: Update debounce tracking sequence for output frame stability confirmation
        self.consecutive_correct = self.consecutive_correct + 1 if all_ok else 0
        final_ok = self.consecutive_correct >= self.required_consecutive

        # 📊 Granular Real-time Metric Monitoring Log
        # explain: Print comprehensive telemetry metrics across all joints for precision tracking diagnostics
        if self.enable_debug_log:
            print("\n============================================================")
            print("📊 [O Gesture] Real-time Geometric Data Stream")
            print(f"  [🌟 Absolute Vector Direction Lock]:")
            print(f"    - Measured Hand Angle: {global_angle:.1f}°")
            print(f"    - Allowed Left Window: {self.THRESHOLDS['hand_direction_angle_min']}° to {self.THRESHOLDS['hand_direction_angle_max']}°")
            print(f"    ⭐ Direction Target Status: {direction_angle_ok}")
            print(f"  [Palm Metrics]:")
            print(f"    - 3D Height: {palm_h:.3f} (Min Req: >= {self.THRESHOLDS['min_palm_height']}) -> {palm_valid}")
            print(f"    - Aspect Ratio: {palm_ratio:.3f} (Allowed Range: {self.THRESHOLDS['palm_ratio_min']} - {self.THRESHOLDS['palm_ratio_max']}) -> {face_valid}")
            print(f"  [Anti-Flat Structural Profile]:")
            print(f"    - O-Hollow Core Depth: {hollow_depth:.3f} (Min Req: > {self.THRESHOLDS['hollow_space_min']}) -> {hollow_ok}")
            print(f"  [Normalized Tip-to-Thumb Proximity Distances]:")
            print(f"    - Index-to-Thumb: {normalized_tip_dists[0]:.3f} | Middle-to-Thumb: {normalized_tip_dists[1]:.3f}")
            print(f"    - Ring-to-Thumb:  {normalized_tip_dists[2]:.3f} | Pinky-to-Thumb:  {normalized_tip_dists[3]:.3f}")
            print(f"    ⭐ Clumping Target Status (Max Threshold < {self.THRESHOLDS['tip_near_ratio']}): {tip_valid}")
            print(f"  [Joint Bend Segment Profile (PIP - MCP Diff >= {self.THRESHOLDS['joint_bend_diff']}°)]:")
            print(f"    - Index Arc Diff: {idx_shapes[2]:.1f}° -> {idx_ok}")
            print(f"    - Middle Arc Diff: {mid_shapes[2]:.1f}° -> {mid_ok}")
            print(f"    - Ring Arc Diff: {ring_shapes[2]:.1f}° -> {ring_ok}")
            print(f"    - Pinky Arc Diff: {pky_shapes[2]:.1f}° -> {pinky_ok}")
            print(f"  [Terminal Tip Flexion Threshold Checks (Max Limit < {self.THRESHOLDS['finger_bend_max']}°)]:")
            print(f"    - Terminal Joints: Idx={idx_angle:.1f}°, Mid={mid_angle:.1f}°, Ring={ring_angle:.1f}°, Pky={pky_angle:.1f}° -> {fingers_ok}")
            print(f"❌/✅ Current Frame Validation: {all_ok} | Stabilizer Debounce: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")

        # explain: Evaluate current pipeline processing phase and dispatch results
        if current_step == 1:
            return 1 if final_ok else 0
        else:
            return self.stroke_count