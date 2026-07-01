import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("J")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        self.stroke_count = 0
        
        # Trajectory tracking states: 
        # IDLE (waiting for signal), TRACKING (tracking coordinates only, ignoring shape), SUCCESS_WAIT (waiting for reset)
        self.state = "IDLE"
        self.start_tip_pos = (0, 0)
        self.max_dy = 0
        self.max_dx = 0

        # Step 1 strict thresholds strictly mirroring i.py
        self.T = {
            "bent_max_angle": 85,
            "pinky_tip_up_ratio": 0.6,
            "thumb_to_mid_ratio": 0.2,
            
            # Step 2 motion trajectory thresholds (based on empirical CSV gesture ranges)
            "move_down_goal": 2.0,      # Swipe downwards past 200% of the hand size
            "move_side_goal": 0.20      # Hook sideways past 20% of the hand size
        }

    def _dist_2d(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _angle(self, p1, p2, p3):
        v1 = (p1.x - p2.x, p1.y - p2.y)
        v2 = (p3.x - p2.x, p3.y - p2.y)
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.hypot(*v1); mag2 = math.hypot(*v2)
        if mag1 == 0 or mag2 == 0: return 0.0
        return math.degrees(math.acos(max(min(dot / (mag1 * mag2), 1.0), -1.0)))

    def check_strict_i(self, lm, hand_size):
        """Strict hand shape validation matching i.py to serve as the action trigger."""
        idx_angle = self._angle(lm[5], lm[6], lm[8])
        mid_angle = self._angle(lm[9], lm[10], lm[12])
        ring_angle = self._angle(lm[13], lm[14], lm[16])
        three_bent = (idx_angle < self.T["bent_max_angle"] and
                      mid_angle < self.T["bent_max_angle"] and
                      ring_angle < self.T["bent_max_angle"])
        
        pinky_up = (lm[17].y - lm[20].y) / hand_size >= self.T["pinky_tip_up_ratio"]
        thumb_ok = (self._dist_2d(lm[4], lm[9]) / hand_size) <= self.T["thumb_to_mid_ratio"]
        return three_bent and pinky_up and thumb_ok

    def check_gesture(self, hands_list: list, current_step: int):
        if not hands_list:
            self.state = "IDLE"
            self.consecutive_correct = 0
            return False if current_step == 1 else self.stroke_count

        hand = hands_list[0]
        lm = hand["landmarks"]
        label = hand.get("label", "Right")
        hand_size = self._dist_2d(lm[0], lm[9])
        if hand_size < 0.01: hand_size = 0.1
        pinky_tip = lm[20]

        # ==========================================
        # STEP 1: Strict Hand Shape Validation (i.py specs)
        # ==========================================
        if current_step == 1:
            if self.check_strict_i(lm, hand_size):
                self.consecutive_correct += 1
            else:
                self.consecutive_correct = 0
            return self.consecutive_correct >= self.required_consecutive

        # ==========================================
        # STEP 2: Trajectory Tracking (Shape ignored mid-stroke)
        # ==========================================
        elif current_step == 2:
            if self.state == "IDLE":
                # Wait for the standard "I" pose to appear as the start signal
                if self.check_strict_i(lm, hand_size):
                    self.state = "TRACKING"
                    self.start_tip_pos = (pinky_tip.x, pinky_tip.y)
                    self.max_dy = 0
                    self.max_dx = 0
                    print(f"\n[J LOG] >>> Trigger signal detected! Anchor origin: ({pinky_tip.x:.2f}, {pinky_tip.y:.2f})")

            elif self.state == "TRACKING":
                # Core Mechanism: Absolutely no check_shape calls here.
                # Only evaluate the displacement of Landmark 20 relative to the anchor origin.
                dy = pinky_tip.y - self.start_tip_pos[1]
                dx = pinky_tip.x - self.start_tip_pos[0]
                
                if dy > self.max_dy: self.max_dy = dy
                
                # Determine the hook direction based on handedness (Right hand hooks leftwards, decreasing dx)
                if label == "Right":
                    if dx < self.max_dx: self.max_dx = dx
                else:
                    if dx > self.max_dx: self.max_dx = dx

                dy_r = self.max_dy / hand_size
                dx_r = abs(self.max_dx) / hand_size

                # Real-time tracking telemetry log
                print(f"[J TRACKING] Downward: {dy_r:.2f} | Sideways Hook: {dx_r:.2f} ", end='\r')

                # Completion Criteria Met: Targets achieved
                if dy_r >= self.T["move_down_goal"] and dx_r >= self.T["move_side_goal"]:
                    self.stroke_count += 1
                    self.state = "SUCCESS_WAIT"
                    print(f"\n[J LOG] ✅ 'J' Stroke Successful! Total count: {self.stroke_count}")

            elif self.state == "SUCCESS_WAIT":
                # Wait for the hand to return to the upper position and re-establish the "I" pose.
                # This guarantees that consecutive strokes are distinct and isolated movements.
                is_reset = pinky_tip.y < self.start_tip_pos[1] + 0.05
                if is_reset and self.check_strict_i(lm, hand_size):
                    self.state = "IDLE"
                    print("[J LOG] Reset complete. Awaiting next tracking trigger signal...")

            return self.stroke_count