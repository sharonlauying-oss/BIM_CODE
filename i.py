# ==============================================================================
# 🤟 BIM Letter "I" Hand Gesture Specification
# 1. Palm Orientation: Palm faces the camera directly. For right hand: lm[1].x < lm[17].x. Palm aspect ratio is used to limit tilt.
# 2. Distance Adaptation
#    - Define hand_size as 2D distance from wrist(lm[0]) to middle finger MCP(lm[9]).
#    - All distances are converted to relative ratios to avoid misjudgment at different distances.
# 3. Finger Status
#    - Index, middle & ring finger: Fully bent, MCP-PIP-TIP angle < 85°.
#    - Pinky: Fully stretched upward. Verified by Y-axis offset ratio and length ratio to reduce perspective error.
#    - Thumb: Bent inward against palm. Normalized distance between thumb tip(lm[4]) and middle MCP(lm[9]) < 0.2.
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture


class GestureChecker(BaseGesture):
    def __init__(self):
        # Initialize gesture name and anti-shake frame counter
        super().__init__("I")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        
        # DEBUG SWITCH: Set to True to print detailed tuning logs in console
        self.debug = False

        # ALL THRESHOLDS ARE NOW RELATIVE TO HAND SIZE (0.0-1.0)
        # No more absolute values, works at any distance
        self.T = {
            "min_palm_h": 0.06,            # Minimum real-world palm height (in meters) to start detection
            "palm_ratio_min": 0.35,        # Minimum allowed ratio of palm width to palm height
            "palm_ratio_max": 0.75,        # Maximum allowed ratio of palm width to palm height
            "bent_max_angle": 85,          # Maximum joint angle allowed for a finger to be considered "curled"
            "thumb_bent_max_angle": 160,   # Maximum joint angle allowed for the thumb to be considered "bent"
            "thumb_z_offset": -0.05,       # Z-depth offset to ensure thumb is physically in front of the palm
            
            # NEW: Relative thresholds (pixel distance / hand_size)
            "pinky_tip_up_ratio": 0.6,     # Required vertical extension of pinky relative to hand size
            "pinky_length_ratio": 0.8,     # Required straight-line pinky length relative to hand size
            "thumb_to_mid_ratio": 0.2      # Maximum allowed distance from thumb tip to middle MCP joint
        }

    def _dist(self, p1, p2):
        # Calculate 2D Euclidean distance between two points in screen space (0.0 to 1.0)
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _wdist(self, p1, p2):
        # Calculate 3D world coordinate distance (in metric space, e.g., meters)
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    def _angle(self, p1, p2, p3):
        # Calculate joint angle in degrees at vertex p2 formed by lines p1-p2 and p3-p2
        v1x, v1y = p1.x - p2.x, p1.y - p2.y
        v2x, v2y = p3.x - p2.x, p3.y - p2.y
        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        cos_a = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_a))

    def reset_dynamic_tracking(self):
        # Reset consecutive valid frame counter
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        # Return immediately if no hand is detected
        if not hands_list:
            self.consecutive_correct = 0
            return False if current_step == 1 else 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # Calculate base hand size (Wrist to Middle finger MCP) - acts as our scale-invariant baseline
        hand_size = self._dist(lm[0], lm[9])
        # Prevent division by zero errors for hands far away or poorly detected
        if hand_size < 0.01:
            hand_size = 0.1

        # 1. Validate palm size
        # Uses 3D world landmarks to check if the physical hand size is within a realistic camera range
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.T["min_palm_h"]

        # 2. Check palm facing camera: thumb on left, pinky on right (assuming right hand)
        thumb_mcp_x = lm[1].x
        pinky_mcp_x = lm[17].x
        thumb_on_left = thumb_mcp_x < pinky_mcp_x

        # Validate aspect ratio of the palm to ensure the hand isn't turned sideways
        palm_w = self._dist(lm[5], lm[17])
        palm_h_2d = self._dist(lm[0], lm[9])
        palm_ratio = palm_w / palm_h_2d if palm_h_2d > 0 else 0
        palm_ratio_ok = self.T["palm_ratio_min"] < palm_ratio < self.T["palm_ratio_max"]
        palm_front_ok = thumb_on_left and palm_ratio_ok

        # 3. Calculate all finger joint angles
        idx_angle = self._angle(lm[5], lm[6], lm[8])
        mid_angle = self._angle(lm[9], lm[10], lm[12])
        ring_angle = self._angle(lm[13], lm[14], lm[16])
        thumb_angle = self._angle(lm[1], lm[2], lm[4])

        # 4. Validate index/middle/ring fingers are curled (bent sharply into the palm)
        three_bent = (idx_angle < self.T["bent_max_angle"] and
                      mid_angle < self.T["bent_max_angle"] and
                      ring_angle < self.T["bent_max_angle"])

        # 5. Validate little finger: fully stretched and pointing up (RELATIVE TO HAND SIZE)
        pinky_mcp_y = lm[17].y
        pinky_tip_y = lm[20].y
        pinky_up_dist = pinky_mcp_y - pinky_tip_y  # Positive value means tip is above MCP
        pinky_length = self._dist(lm[17], lm[20])
        
        # Convert to relative ratios so the verification works at any camera distance
        pinky_up_ratio = pinky_up_dist / hand_size
        pinky_length_ratio = pinky_length / hand_size
        
        pinky_point_up = (pinky_up_ratio >= self.T["pinky_tip_up_ratio"] and
                          pinky_length_ratio >= self.T["pinky_length_ratio"])

        # 6. Validate thumb: bent + in front + close to middle finger (RELATIVE TO HAND SIZE)
        thumb_bent = thumb_angle < self.T["thumb_bent_max_angle"]
        # Use 3D Z-depth to confirm thumb is resting on top/in-front of the closed fingers
        thumb_front = wm[4].z >= wm[0].z + self.T["thumb_z_offset"]
        
        # Convert distance to a relative ratio to measure thumb proximity to the fist
        thumb_to_mid_dist = self._dist(lm[4], lm[9])
        thumb_to_mid_ratio = thumb_to_mid_dist / hand_size
        thumb_close = thumb_to_mid_ratio <= self.T["thumb_to_mid_ratio"]
        
        thumb_ok = thumb_bent and thumb_front and thumb_close

        # Combine all I gesture recognition rules
        all_conditions = (
            palm_valid and palm_front_ok and
            three_bent and pinky_point_up and thumb_ok
        )

        # DETAILED TUNING LOGS
        if self.debug:
            print("\n--- [I GESTURE DIAGNOSTIC LOG] ---")
            print(f"Hand Size Baseline (2D): {hand_size:.3f}")
            print(f"[1. Palm Valid]: {palm_valid} | World Palm H: {palm_h:.3f} (Min: {self.T['min_palm_h']})")
            print(f"[2. Palm Front]: {palm_front_ok} | Thumb Left: {thumb_on_left}, Ratio: {palm_ratio:.2f} (Target: {self.T['palm_ratio_min']}~{self.T['palm_ratio_max']})")
            print(f"[3-4. Curled Fingers]: {three_bent} | Angles -> Index: {idx_angle:.1f}°, Mid: {mid_angle:.1f}°, Ring: {ring_angle:.1f}° (Max Allowed: {self.T['bent_max_angle']}°)")
            print(f"[5. Pinky Extended]: {pinky_point_up} | Up Ratio: {pinky_up_ratio:.2f} (Min: {self.T['pinky_tip_up_ratio']}), Length Ratio: {pinky_length_ratio:.2f} (Min: {self.T['pinky_length_ratio']})")
            print(f"[6. Thumb Position]: {thumb_ok} | Bent: {thumb_bent} ({thumb_angle:.1f}°), Front: {thumb_front} (Z-diff: {(wm[4].z - wm[0].z):.3f}), Close: {thumb_close} (Ratio: {thumb_to_mid_ratio:.2f} / Max: {self.T['thumb_to_mid_ratio']})")
            print(f"[RESULT] -> Match: {all_conditions} | Consecutive Frames: {self.consecutive_correct + 1 if all_conditions else 0}/{self.required_consecutive}")

        # Update consecutive valid frame counter for anti-shake/smoothing filter
        if all_conditions:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0

        # Return final recognition result if frames hold stable
        result = self.consecutive_correct >= self.required_consecutive
        return result if current_step == 1 else self.stroke_count