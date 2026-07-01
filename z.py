# gestures/z.py
# BIM Letter "Z"
# Step 1: Index and Thumb extended forward, Middle/Ring/Pinky curled.
# Step 2: Adaptive Anchor-Based Region Segmentation (Spawn Grid from Step 1 Trigger Point)

import math
import time
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("Z")
        
        self.stroke_count = 0
        self.required_strokes = 3  
        
        # Region State Machine: IDLE -> TOP_LEFT -> WAIT_TOP_RIGHT -> WAIT_BOTTOM_LEFT -> WAIT_BOTTOM_RIGHT
        self.tracking_state = "IDLE"
        
        # ⚓ Dynamic Local Coordinate Origin (Captured at the exact frame Step 1 qualifies)
        self.anchor_x = None
        self.anchor_y = None
        
        # ⏳ Continuity Timeout & Drop-frame Tolerance System
        self.last_state_change_time = 0.0
        self.max_allowable_staleness = 3.5  # Timeout if stalled in a zone for over 3.5s
        self.lost_track_frames = 0          # Tolerance buffer for fast-swipe tracking drops
        
        self.debug = False

    def reset_dynamic_tracking(self):
        """Resets the tracking grid and state machine while preserving the stroke count."""
        current_strokes = self.stroke_count  # Persist currently accumulated strokes
        
        super().reset_dynamic_tracking()     # Call base class reset
        
        self.stroke_count = current_strokes  # Restore stroke count to prevent override
        self.tracking_state = "IDLE"
        self.anchor_x = None
        self.anchor_y = None
        self.lost_track_frames = 0
        self.last_state_change_time = 0.0

    def _get_distance(self, pt1, pt2):
        return math.hypot(pt1.x - pt2.x, pt1.y - pt2.y)

    def check_static_pose(self, hand):
        """Step 1 Static Pose Validation (Original constraints preserved)"""
        lm = hand["landmarks"]
        hand_size = self._get_distance(lm[0], lm[9])
        if hand_size == 0: return False
            
        middle_curled = self._get_distance(lm[12], lm[0]) < self._get_distance(lm[9], lm[0])
        ring_curled = self._get_distance(lm[16], lm[0]) < self._get_distance(lm[13], lm[0])
        pinky_curled = self._get_distance(lm[20], lm[0]) < self._get_distance(lm[17], lm[0])
        
        index_extended = self._get_distance(lm[8], lm[0]) > self._get_distance(lm[5], lm[0]) * 1.1
        thumb_extended = self._get_distance(lm[4], lm[0]) > self._get_distance(lm[2], lm[0])
        thumb_to_middle = self._get_distance(lm[4], lm[9]) < hand_size * 0.85
        
        return middle_curled and ring_curled and pinky_curled and index_extended and thumb_extended and thumb_to_middle

    def check_gesture(self, hands_list: list, current_step: int):
        # Handle transient frame drops during fast stroke motion in Step 2
        if not hands_list:
            if current_step == 2 and self.tracking_state != "IDLE":
                self.lost_track_frames += 1
                if self.lost_track_frames > 8:  # Timeout if hand is missing for more than 8 consecutive frames
                    if self.debug: print("[Z-Region] ❌ Hand lost tracking too long. Resetting adaptive grid.")
                    self.reset_dynamic_tracking()
            return 0 if current_step == 2 else False

        hand = hands_list[0]

        # -----------------------------------------------------
        # STEP 1: Static Pose Validation
        # -----------------------------------------------------
        if current_step == 1:
            return self.check_static_pose(hand)

        # -----------------------------------------------------
        # STEP 2: Adaptive Anchor-Based Z-Trajectory Tracking
        # -----------------------------------------------------
        elif current_step == 2:
            current_time = time.time()
            self.lost_track_frames = 0  # Frame recovered, clear frame drop counter

            # Track Index Tip (INDEX_TIP = 8) normalized coordinates
            tip = hand["landmarks"][8]
            
            # Use hand width as scale base for the adaptive grid mapping
            hand_width = self._get_distance(hand["landmarks"][5], hand["landmarks"][17])
            if hand_width == 0: hand_width = 0.1

            # ⏱️ Continuity Timeout Validation
            if self.tracking_state != "IDLE":
                if (current_time - self.last_state_change_time) > self.max_allowable_staleness:
                    if self.debug: print("[Z-Region] ❌ Trajectory stale (Timeout). Resetting adaptive grid.")
                    self.reset_dynamic_tracking()
                    return self.stroke_count

            # =================================================
            # ⚓ State 1: IDLE -> Anchor initial frame as [TOP_LEFT] origin
            # =================================================
            if self.tracking_state == "IDLE":
                self.anchor_x = tip.x
                self.anchor_y = tip.y
                self.tracking_state = "TOP_LEFT"
                self.last_state_change_time = current_time
                if self.debug: 
                    print(f"\n[Z-Region] 🟢 Step 1 Triggered! Anchor locked at [TOP_LEFT]: ({self.anchor_x:.3f}, {self.anchor_y:.3f})")
                    print("[Z-Region] -> Swipe RIGHT toward [TOP_RIGHT] zone...")

            # =================================================
            # 📐 Compute relative displacement vectors from anchor
            # =================================================
            dx = tip.x - self.anchor_x  # Positive = Rightwards
            dy = tip.y - self.anchor_y  # Positive = Downwards

            # Scale coefficients: Swipe distance scaled dynamically to hand width
            target_span_x = hand_width * 0.65
            target_span_y = hand_width * 0.55

            # =================================================
            # 🔄 Adaptive Grid State Machine Driver
            # =================================================
            
            # 2. Waiting for [TOP_RIGHT]: Displaced rightwards, with minimal vertical drop
            if self.tracking_state == "TOP_LEFT":
                if dx > target_span_x and dy < target_span_y * 0.5:
                    self.tracking_state = "WAIT_TOP_RIGHT"
                    self.last_state_change_time = current_time
                    if self.debug: print(f"[Z-Region] ➡️ Stroke 1 Clear! Registered [TOP_RIGHT] (dx: {dx:+.3f}). Slice DIAGONAL down-left to [BOTTOM_LEFT]...")

            # 3. Waiting for [BOTTOM_LEFT]: Slice back leftwards and significantly down
            elif self.tracking_state == "WAIT_TOP_RIGHT":
                if dx < target_span_x * 0.4 and dy > target_span_y:
                    self.tracking_state = "WAIT_BOTTOM_LEFT"
                    self.last_state_change_time = current_time
                    if self.debug: print(f"[Z-Region] ↙️ Stroke 2 Clear! Registered [BOTTOM_LEFT] (dy: {dy:+.3f}). Drag right for final baseline to [BOTTOM_RIGHT]...")

            # 4. Waiting for [BOTTOM_RIGHT]: Complete baseline swipe rightwards while remaining low
            elif self.tracking_state == "WAIT_BOTTOM_LEFT":
                if dx > target_span_x * 0.8 and dy > target_span_y * 0.8:
                    self.stroke_count += 1  
                    if self.debug:
                        print(f"[Z-Region] 🎉🎉 Z-Trajectory Completed Successfully! Registered global count: {self.stroke_count}/{self.required_strokes}\n")
                    
                    # Cycle state back to IDLE, ready for the next sequence mapping
                    self.reset_dynamic_tracking()

            return self.stroke_count