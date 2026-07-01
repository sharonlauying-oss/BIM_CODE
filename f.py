# ==============================================================================
# 🤟 BIM "F" Hand Gesture Description:
# - Hand Orientation: Right hand, palm facing the CAMERA directly.
# - Index Finger: Curved forward to form a circle/pinch with the thumb.
# - Middle, Ring, & Pinky Fingers: Fully extended straight up and closely closed together.
# - Thumb: Oriented upward, with its tip overlapping or pressed close to the index tip.
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("F")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        self.stroke_count = 1

        # 🐛 Enable explicit runtime console diagnostics
        self.enable_debug_log = False

        # Calibration thresholds for stable recognition
        self.THRESHOLDS = {
            "min_palm_height": 0.06,
            "index_forward_threshold": 0.003,
            
            # ☝️ 🖕 2D Joint projection limits for extended straight fingers
            "extended_finger_min": 130.0,
            "extended_finger_max": 180.0,
            
            # 📏 🌟 Perspective Foreshortening Ratios (Self-Adaptive Segment Length Ratios)
            # When fingers bend toward the camera, their projected length ratio drops below these values.
            "min_middle_length_ratio": 0.45,  # (Dist 10 to 12) / hand_size
            "min_ring_length_ratio": 0.42,    # (Dist 14 to 16) / hand_size
            "min_pinky_length_ratio": 0.38,   # (Dist 18 to 20) / hand_size
            
            # 🤝 Cohesiveness Lock parameters
            "adjacent_finger_max_dist": 0.06,
            "palm_ratio_min": 0.30,
            
            # 👍 Thumb Alignment parameters
            "thumb_index_tip_max": 0.03,
            "thumb_up_min": 120.0,
            "thumb_index_close_max": 0.07
        }

    def _get_distance(self, p1, p2):
        # Calculate 2D Euclidean distance between two coordinate markers
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _get_world_distance(self, p1, p2):
        # Calculate 3D physical distance in meters using full spatial coordinates
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    def _get_angle(self, p1, p2, p3):
        # Calculate the 2D joint flexion angle using vector dot product calculation
        v1x, v1y = p1.x - p2.x, p1.y - p2.y
        v2x, v2y = p3.x - p2.x, p3.y - p2.y
        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0: return 0.0
        cos_angle = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_angle))

    def reset_dynamic_tracking(self):
        # Reset consecutive valid frame counter memory state
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        # Safety fallback: If no hand tracking framework matrix is found, drop frame count
        if not hands_list:
            self.consecutive_correct = 0
            return False if current_step == 1 else 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # Establish dynamic scale boundary based on palm profile (Wrist 0 to Middle MCP 9)
        hand_size = self._get_distance(lm[0], lm[9])
        if hand_size == 0: return 0

        # 1. Structural Baseline: Palm Verification
        palm_h = self._get_world_distance(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # Get finger tip landmarks
        idx_tip = lm[8]
        mid_tip = lm[12]
        ring_tip = lm[16]
        pinky_tip = lm[20]

        # 2. Index Finger Check: Must be curved forward relative to middle finger depth plane
        index_valid = idx_tip.z < mid_tip.z - self.THRESHOLDS["index_forward_threshold"]

        # 3. Traditional Joint Angle Verification (Middle, Ring, Pinky extended)
        mid_angle = self._get_angle(lm[9], lm[10], mid_tip)
        ring_angle = self._get_angle(lm[13], lm[14], ring_tip)
        pinky_angle = self._get_angle(lm[17], lm[18], pinky_tip)

        mid_extended = self.THRESHOLDS["extended_finger_min"] < mid_angle < self.THRESHOLDS["extended_finger_max"]
        ring_extended = self.THRESHOLDS["extended_finger_min"] < ring_angle < self.THRESHOLDS["extended_finger_max"]
        pinky_extended = self.THRESHOLDS["extended_finger_min"] < pinky_angle < self.THRESHOLDS["extended_finger_max"]
        angles_ok = mid_extended and ring_extended and pinky_extended

        # 4. 🌟 Perspective Anti-Flattening Guard (Foreshortening Length Ratio Check)
        # Calculates the projected length of upper segments to catch forward-bending cheating vectors
        ratio_mid = self._get_distance(lm[10], lm[12]) / hand_size # PIP (10) to TIP (12)
        ratio_rng = self._get_distance(lm[14], lm[16]) / hand_size # PIP (14) to TIP (16)
        ratio_pky = self._get_distance(lm[18], lm[20]) / hand_size # PIP (18) to TIP (20)

        lengths_ok = (ratio_mid >= self.THRESHOLDS["min_middle_length_ratio"] and
                      ratio_rng >= self.THRESHOLDS["min_ring_length_ratio"] and
                      ratio_pky >= self.THRESHOLDS["min_pinky_length_ratio"])

        # 5. Cohesiveness Check: Middle, Ring, and Pinky must remain tightly closed together
        mid_ring_dist = self._get_world_distance(wm[12], wm[16])
        ring_pinky_dist = self._get_world_distance(wm[16], wm[20])
        fingers_closed = (mid_ring_dist < self.THRESHOLDS["adjacent_finger_max_dist"]) and (ring_pinky_dist < self.THRESHOLDS["adjacent_finger_max_dist"])
        
        fingers_valid = index_valid and angles_ok and lengths_ok and fingers_closed

        # 6. Thumb Physical Anchor: Upward position, overlapping or close to the bent index finger tip
        thumb_tip_dist = self._get_world_distance(wm[4], wm[8])
        thumb_tip_close = thumb_tip_dist < self.THRESHOLDS["thumb_index_tip_max"]
        thumb_angle = self._get_angle(lm[1], lm[2], lm[3])
        thumb_up = thumb_angle > self.THRESHOLDS["thumb_up_min"]
        thumb_index_dist = self._get_world_distance(wm[4], wm[5])
        thumb_close = thumb_index_dist < self.THRESHOLDS["thumb_index_close_max"]
        thumb_valid = thumb_tip_close and thumb_up and thumb_close

        # 7. Palm Orientation Check: Front-facing flat layer evaluation
        palm_w = self._get_distance(lm[5], lm[17])
        palm_h2d = self._get_distance(lm[0], lm[9])
        palm_ratio = palm_w / palm_h2d if palm_h2d > 0 else 0
        palm_forward = palm_ratio > self.THRESHOLDS["palm_ratio_min"]

        # 8. Integrated Execution Pipeline Verification Convergence
        valid = palm_valid and fingers_valid and thumb_valid and palm_forward

        # Dynamic debounce mechanism to filter structural flutter and noise patterns
        if valid:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0

        result = self.consecutive_correct >= self.required_consecutive
        
        if self.enable_debug_log:
            print("\n============================================================")
            print(f"📊 [F Gesture Profile] Real-Time Runtime Metrics")
            print(f"  [1. Base Palm Validity]: Depth={palm_h:.3f} | FrontFacingRatio={palm_ratio:.2f} -> Status: {palm_valid and palm_forward}")
            print(f"  [2. Curved Index Tracking]: Index Z-Depth Delta={idx_tip.z - mid_tip.z:.4f} -> Status: {index_valid}")
            print(f"  [3. Extension Angles]: Middle={mid_angle:.1f}° | Ring={ring_angle:.1f}° | Pinky={pinky_angle:.1f}° -> Status: {angles_ok}")
            print(f"  [4. 🌟 Foreshortening Ratios]: Middle={ratio_mid:.3f} (Req: >{self.THRESHOLDS['min_middle_length_ratio']})")
            print(f"                               Ring={ratio_rng:.3f} (Req: >{self.THRESHOLDS['min_ring_length_ratio']})")
            print(f"                               Pinky={ratio_pky:.3f} (Req: >{self.THRESHOLDS['min_pinky_length_ratio']}) -> Status: {lengths_ok}")
            print(f"  [5. Column Cohesiveness]: Mid-Ring={mid_ring_dist:.3f} | Ring-Pinky={ring_pinky_dist:.3f} -> Status: {fingers_closed}")
            print(f"  [6. Pinch Thumb Anchor]: TipProximity={thumb_tip_dist:.3f} | JointAngle={thumb_angle:.1f}° -> Status: {thumb_valid}")
            print(f"❌/✅ Current Frame Result: {valid} | Stabilizer Buffer: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")

        return result if current_step == 1 else self.stroke_count