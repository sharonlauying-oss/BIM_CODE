# gestures/tolong.py
# BIM Gesture: "Tolong" (Anti-Occlusion Robust Version)
# Step 1: Detects one "Thumbs Up" hand and one "Flat" hand anywhere in the frame without relying on strict Left/Right labels.
# Step 2: Tracks ONLY the active Thumbs-Up hand lifting upwards, ignoring the lower hand completely to prevent MediaPipe glitching.

import math
import time
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("Tolong")
        
        # Keep this True so vision_engine passes multiple hands when available
        self.is_double_hand = True  
        
        self.consecutive_correct = 0
        self.required_consecutive = 2  # Lowered slightly to snap into action before glitching
        self.stroke_count = 0
        
        # Trajectory tracking state machine: IDLE -> LIFTING -> SUCCESS_WAIT
        self.tracking_state = "IDLE"
        self.start_y = 0.0
        
        # Optimized robust thresholds
        self.T = {
            "curl_max_angle": 75,        # Relaxed slightly for fist flexibility
            "lift_goal": 0.35            # Target upward displacement ratio (35% of hand size)
        }
        
        self.debug = True

    def _get_distance(self, pt1, pt2):
        return math.hypot(pt1.x - pt2.x, pt1.y - pt2.y)

    def _angle(self, p1, p2, p3):
        v1 = (p1.x - p2.x, p1.y - p2.y)
        v2 = (p3.x - p2.x, p3.y - p2.y)
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.hypot(*v1); mag2 = math.hypot(*v2)
        if mag1 == 0 or mag2 == 0: return 0.0
        return math.degrees(math.acos(max(min(dot / (mag1 * mag2), 1.0), -1.0)))

    def check_gesture(self, hands_list: list, current_step: int):
        # Fallback handling for tracking loss
        if not hands_list:
            self.consecutive_correct = 0
            if current_step == 2 and self.tracking_state != "IDLE":
                self.tracking_state = "IDLE"
            return False if current_step == 1 else self.stroke_count

        # =========================================================================
        # STEP 1: Anti-Glitch Label-Agnostic Static Validation
        # =========================================================================
        if current_step == 1:
            if len(hands_list) < 2:
                self.consecutive_correct = 0
                if self.debug: print("[Tolong Step1] ❌ Awaiting both hands...      ", end='\r')
                return False

            # Scan the hands and classify them purely by shape, completely ignoring MediaPipe's buggy Left/Right labels
            thumbs_up_hand = None
            flat_hand = None

            for hand in hands_list[:2]:
                lm = hand["landmarks"]
                size = self._get_distance(lm[0], lm[9])
                if size < 0.01: continue

                # Evaluate Thumbs Up Shape criteria
                idx_angle = self._angle(lm[5], lm[6], lm[8])
                mid_angle = self._angle(lm[9], lm[10], lm[12])
                ring_angle = self._angle(lm[13], lm[14], lm[16])
                four_curled = (idx_angle < self.T["curl_max_angle"] and 
                               mid_angle < self.T["curl_max_angle"] and 
                               ring_angle < self.T["curl_max_angle"])
                
                thumb_extended = self._get_distance(lm[4], lm[2]) / size > 0.45
                thumb_upward = lm[4].y < lm[2].y
                
                if four_curled and thumb_extended and thumb_upward:
                    thumbs_up_hand = hand
                    continue

                # Evaluate Flat Hand Shape criteria
                idx_straight = self._get_distance(lm[8], lm[0]) > self._get_distance(lm[5], lm[0]) * 1.1
                mid_straight = self._get_distance(lm[12], lm[0]) > self._get_distance(lm[9], lm[0]) * 1.1
                if idx_straight and mid_straight:
                    flat_hand = hand

            # Verify that both distinct roles are detected simultaneously in frame
            pose_matched = (thumbs_up_hand is not None) and (flat_hand is not None)

            if pose_matched:
                self.consecutive_correct += 1
            else:
                self.consecutive_correct = 0

            if self.debug:
                t_status = "FOUND" if thumbs_up_hand else "MISSING"
                f_status = "FOUND" if flat_hand else "MISSING"
                print(f"[Tolong Step1] Thumbs-Up: {t_status} | Flat Hand: {f_status}      ", end='\r')

            res = self.consecutive_correct >= self.required_consecutive
            if res and self.debug:
                print("\n[Tolong] 🎉 STEP 1 PASSED! Double hand structural pose established.")
            return res

        # =========================================================================
        # STEP 2: Single-Hand Target Isolation Lifting Tracking
        # =========================================================================
        elif current_step == 2:
            # Look for ANY hand currently acting as the Thumbs-Up anchor in the frame
            active_hand = None
            for hand in hands_list:
                lm = hand["landmarks"]
                idx_angle = self._angle(lm[5], lm[6], lm[8])
                # A quick loose check to find the fist/thumbs-up hand
                if idx_angle < self.T["curl_max_angle"] and lm[4].y < lm[2].y:
                    active_hand = hand
                    break
            
            # Fallback: If overlapping causes total mess, default track the first hand element in list
            if not active_hand:
                active_hand = hands_list[0]

            lm_active = active_hand["landmarks"]
            h_size = self._get_distance(lm_active[0], lm_active[9])
            if h_size < 0.01: h_size = 0.1
            
            # Track the wrist Y coordinate of this primary hand
            current_y = lm_active[0].y

            if self.tracking_state == "IDLE":
                self.tracking_state = "LIFTING"
                self.start_y = current_y
                if self.debug: 
                    print(f"\n[Tolong] 🟢 Trajectory Locked! Origin Y: {self.start_y:.3f}. Lift UPWARDS...")

            elif self.tracking_state == "LIFTING":
                # Moving upward reduces the raw screen Y coordinate value
                dy = self.start_y - current_y
                dy_normalized = dy / h_size

                if self.debug:
                    print(f"[TOLONG LIFT] Progress: {dy_normalized:.2f} / Goal: {self.T['lift_goal']}      ", end='\r')

                # Verification criteria successfully cleared
                if dy_normalized >= self.T["lift_goal"]:
                    self.stroke_count += 1
                    self.tracking_state = "SUCCESS_WAIT"
                    if self.debug: 
                        print(f"\n[Tolong] ✅ Stroke Success! Total registered repetitions: {self.stroke_count}")

            elif self.tracking_state == "SUCCESS_WAIT":
                # Wait for hands to drop down before enabling the next loop sequence
                if current_y > self.start_y - 0.05:
                    self.tracking_state = "IDLE"
                    if self.debug: print("[Tolong] System returned to neutral base. Awaiting trigger...")

            return self.stroke_count