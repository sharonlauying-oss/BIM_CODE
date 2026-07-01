# ==============================================================================
# 🤟 BIM "H" Gesture Description:
# - Hand Orientation: Right hand, back of the palm facing the CAMERA directly.
# - Finger Direction: Hand is rotated sideways so extended fingers point LEFT.
# - Index & Middle Fingers: Fully straight and extended tightly parallel to each other.
# - Ring & Pinky Fingers: Tucked/bent firmly down into the inner palm base cavity.
# - Thumb: Tucked flat along the index finger/palm layer, not sticking out.
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        # Initialize gesture name and dynamic anti-shake stabilization parameters
        super().__init__("H")
        self.consecutive_correct = 0
        self.required_consecutive = 2  # Confirm after 2 consecutive valid frames
        self.stroke_count = 1

        # 🐛 Enable explicit runtime console diagnostics
        self.enable_debug_log = False 

        # 🎯 Geometric thresholds calibrated for the horizontal "H" profile matrix
        self.THRESHOLDS = {
            "min_palm_height": 0.06,          # Min structural palm height (world scale metrics)
            "straight_min_angle": 145.0,      # Min 2D angle required for an extended straight finger
            "bent_max_angle": 80.0,           # Max 2D angle required for ring/pinky folding verification
            "palm_ratio_min": 0.35,           # Min aspect ratio for verifying flat horizontal hand layer
            
            # 📏 🌟 Perspective Foreshortening Ratio (Self-Adaptive Segment Length Ratios)
            # Since H points laterally left, forward-tilting fingers flatten and shrink horizontally.
            "min_index_length_ratio": 0.42,   # (2D Dist 6 to 8) / palm_h_2d
            "min_middle_length_ratio": 0.45,  # (2D Dist 10 to 12) / palm_h_2d
            
            # 👍 Thumb Position and Plane Alignment Guards
            "thumb_max_forward_z": 0.05,      # Limit to prevent thumb from drifting toward lens space
            "thumb_max_y_offset": 0.1         # Vertical bounding limit for thumb placement
        }

    def _dist(self, p1, p2):
        # Calculate 2D Euclidean distance between two coordinate markers
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _wdist(self, p1, p2):
        # Calculate 3D World Euclidean distance using full spatial coordinates
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _angle(self, p1, p2, p3):
        # Calculate the 2D joint flexion angle using vector dot product calculation
        v1x, v1y = p1.x - p2.x, p1.y - p2.y
        v2x, v2y = p3.x - p2.x, p3.y - p2.y
        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0: return 0.0
        cos_a = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_a))

    def reset_dynamic_tracking(self):
        # Reset dynamic timeline memory state indicators
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        # Safety fallback: If no hand tracking framework matrix is found, drop frame count
        if not hands_list:
            self.consecutive_correct = 0
            return 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        # Establish dynamic baseline scaling dimensions using 2D palm structure height
        palm_h_2d = self._dist(lm[0], lm[9])
        if palm_h_2d <= 0: return 0

        # 1. Structural Baseline: Palm Height Verification
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # 2. Traditional Joint Angle Verification (Index & Middle straight; Ring & Pinky bent)
        idx_a = self._angle(lm[5], lm[6], lm[8])
        mid_a = self._angle(lm[9], lm[10], lm[12])
        ring_a = self._angle(lm[13], lm[14], lm[16])
        pinky_a = self._angle(lm[17], lm[18], lm[20])

        two_straight = (idx_a > self.THRESHOLDS["straight_min_angle"] and
                        mid_a > self.THRESHOLDS["straight_min_angle"])
        two_bent = (ring_a < self.THRESHOLDS["bent_max_angle"] and
                    pinky_a < self.THRESHOLDS["bent_max_angle"])

        # 3. Direction Lock: Extended tracking tips must project to the left of their roots
        idx_left = lm[8].x < lm[5].x
        mid_left = lm[12].x < lm[9].x
        direction_ok = idx_left and mid_left

        # 4. Aspect Ratio Alignment: Structural check for side-turned palm surface area
        palm_w = self._dist(lm[5], lm[17])
        palm_ratio = palm_w / palm_h_2d
        palm_back_ok = palm_ratio > self.THRESHOLDS["palm_ratio_min"]

        # 5. 🌟 Perspective Anti-Flattening Guard (Foreshortening Length Ratio Check)
        # Binds the 2D projected segment lengths to verify they point straight left instead of curling forward
        ratio_idx = self._dist(lm[6], lm[8]) / palm_h_2d    # PIP (6) to TIP (8)
        ratio_mid = self._dist(lm[10], lm[12]) / palm_h_2d  # PIP (10) to TIP (12)

        lengths_ok = (ratio_idx >= self.THRESHOLDS["min_index_length_ratio"] and
                      ratio_mid >= self.THRESHOLDS["min_middle_length_ratio"])

        # 6. Thumb Safety Bounding Constraints
        thumb_tip = lm[4]
        pinky_mcp = lm[17]
        middle_mcp = lm[9]
        thumb_tip_w = wm[4]
        palm_center_w = wm[0]

        # Thumb must not flare outwards to the right past the pinky base
        thumb_not_out_right = thumb_tip.x <= pinky_mcp.x + 0.05
        # Thumb must not cross or rise structurally above the middle finger track line
        thumb_below_middle = thumb_tip.y >= middle_mcp.y - self.THRESHOLDS["thumb_max_y_offset"]
        # Spatial Depth Guard: Prevents thumb from pushing forward out of the palm plane array
        thumb_behind_palm = thumb_tip_w.z <= palm_center_w.z + self.THRESHOLDS["thumb_max_forward_z"]

        thumb_ok = thumb_not_out_right and thumb_below_middle and thumb_behind_palm

        # 7. Integrated Execution Pipeline (Now requiring lengths_ok verification)
        all_conditions = (
            palm_valid and two_straight and two_bent and lengths_ok and
            direction_ok and palm_back_ok and thumb_ok
        )

        # Dynamic debounce mechanism to filter structural flutter and noise patterns
        self.consecutive_correct = self.consecutive_correct + 1 if all_conditions else 0
        final_ok = self.consecutive_correct >= self.required_consecutive

        if self.enable_debug_log:
            print("\n============================================================")
            print(f"📊 [H Gesture Profile] Real-Time Runtime Metrics")
            print(f"  [1. Palm Structural Baseline]: Depth={palm_h:.3f} | AspectRatio={palm_ratio:.2f} -> Status: {palm_valid and palm_back_ok}")
            print(f"  [2. Twin Towers Extension]: Index={idx_a:.1f}° | Middle={mid_a:.1f}° -> Status: {two_straight}")
            print(f"  [3. 🌟 Foreshortening Ratios]: Index={ratio_idx:.3f} (Req: >{self.THRESHOLDS['min_index_length_ratio']})")
            print(f"                               Middle={ratio_mid:.3f} (Req: >{self.THRESHOLDS['min_middle_length_ratio']}) -> Status: {lengths_ok}")
            print(f"  [4. Lateral Vector Heading]: Index_Points_Left={idx_left} | Middle_Points_Left={mid_left} -> Status: {direction_ok}")
            print(f"  [5. Closed Clusters State]: Ring={ring_a:.1f}° | Pinky={pinky_a:.1f}° -> Status: {two_bent}")
            print(f"  [6. Thumb Bounding Alignment]: UnderMiddle={thumb_below_middle} | BehindPalmPlane={thumb_behind_palm} -> Status: {thumb_ok}")
            print(f"❌/✅ Current Frame Result: {all_conditions} | Stabilizer Buffer: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")

        if current_step == 1:
            return 1 if final_ok else 0
        else:
            return self.stroke_count