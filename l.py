# ==============================================================================
# 🤟 BIM "L" Gesture Description:
# - Hand Orientation: Right hand, palm facing the CAMERA directly.
# - Index Finger: Fully straight, pointing vertically upward.
# - Thumb: Fully extended horizontally out to the side to construct a clear "L" right angle.
# - Middle, Ring, & Pinky Fingers: Folded flush down against the inner palm base cavity.
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("L")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        self.stroke_count = 1

        # 🐛 Enable explicit runtime console diagnostics
        self.enable_debug_log = False 

        # 🎯 Geometrical thresholds calibrated for the orthogonal "L" projection matrix
        self.THRESHOLDS = {
            "min_palm_height": 0.06,
            "palm_ratio_min": 0.35,
            "palm_ratio_max": 0.75,
            
            # ☝️ Extended Index Finger Limits
            "straight_min": 130.0,
            "index_up_ratio": 0.6,
            
            # 📏 🌟 Perspective Foreshortening Ratio (Self-Adaptive Segment Length Ratios)
            # When the index finger bends toward the camera, its projected length ratio drops below this value.
            "min_index_length_ratio": 0.42,   # (Dist 6 to 8) / hand_size

            # 👍 Thumb Orientation & Extension Guards
            "thumb_joint_min": 145.0,
            "thumb_middle_angle_min": 115.0,
            "thumb_middle_angle_max": 135.0,
            
            # ✊ Folded Group constraints (Middle, Ring, Pinky)
            "root_proximity_max": 0.25
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

    def _vector_angle(self, v1, v2):
        # Calculate angular deviation spread between two distinct directional vectors
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])
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

        # Establish dynamic scale boundary based on palm profile (Wrist 0 to Middle MCP 9)
        hand_size = self._dist(lm[0], lm[9])
        if hand_size < 0.01: hand_size = 0.1

        # 1. Structural Baseline: Palm Verification 
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # Structural Orientation Lock: Check if hand belongs to a right hand facing forward
        thumb_on_left = lm[1].x < lm[17].x

        palm_w = self._dist(lm[5], lm[17])
        palm_ratio = palm_w / hand_size if hand_size > 0 else 0
        palm_ratio_ok = self.THRESHOLDS["palm_ratio_min"] < palm_ratio < self.THRESHOLDS["palm_ratio_max"]
        palm_front_ok = palm_valid and thumb_on_left and palm_ratio_ok

        # 2. Thumb Extension & Angular L-Shape Separation Check
        thumb_ip_angle = self._angle(lm[2], lm[3], lm[4])
        thumb_straight = thumb_ip_angle > self.THRESHOLDS["thumb_joint_min"]

        middle_vec = (lm[12].x - lm[9].x, lm[12].y - lm[9].y)
        thumb_vec = (lm[4].x - lm[2].x, lm[4].y - lm[2].y)
        thumb_middle_angle = self._vector_angle(middle_vec, thumb_vec)
        angle_in_range = self.THRESHOLDS["thumb_middle_angle_min"] < thumb_middle_angle < self.THRESHOLDS["thumb_middle_angle_max"]
        thumb_ok = thumb_straight and angle_in_range

        # 3. Index Finger Extension & Directional Profile
        idx_angle = self._angle(lm[5], lm[6], lm[8])
        idx_straight = idx_angle > self.THRESHOLDS["straight_min"]
        idx_up_dist = lm[5].y - lm[8].y
        idx_up_ratio = idx_up_dist / hand_size
        idx_point_up = idx_up_ratio >= self.THRESHOLDS["index_up_ratio"]

        # 4. 🌟 Perspective Anti-Flattening Guard (Foreshortening Length Ratio Check)
        # Calculates the projected length of the upper index segment relative to palm size
        ratio_idx = self._dist(lm[6], lm[8]) / hand_size   # PIP (6) to TIP (8)
        index_length_ok = ratio_idx >= self.THRESHOLDS["min_index_length_ratio"]

        index_ok = idx_straight and idx_point_up and index_length_ok

        # 5. Folded Pillars Cluster Verification (Middle, Ring, and Pinky down)
        def check_finger(mcp_idx, pip_idx, dip_idx, tip_idx):
            root_ratio = self._dist(lm[mcp_idx], lm[pip_idx]) / hand_size
            root_close = root_ratio < self.THRESHOLDS["root_proximity_max"]
            
            front_angle = self._angle(lm[pip_idx], lm[dip_idx], lm[tip_idx])
            front_straight = front_angle > self.THRESHOLDS["straight_min"]
            
            tip_below = lm[tip_idx].y > lm[mcp_idx].y
            return root_close and front_straight and tip_below

        mid_ok = check_finger(9, 10, 11, 12)
        ring_ok = check_finger(13, 14, 15, 16)
        pinky_ok = check_finger(17, 18, 19, 20)
        fingers_ok = mid_ok and ring_ok and pinky_ok

        # 6. Integrated Execution Pipeline
        all_ok = palm_front_ok and thumb_ok and index_ok and fingers_ok

        # Dynamic debounce mechanism to clear noise and stabilize frame updates
        self.consecutive_correct = self.consecutive_correct + 1 if all_ok else 0
        final_ok = self.consecutive_correct >= self.required_consecutive

        if self.enable_debug_log:
            print("\n============================================================")
            print(f"📊 [L Gesture Profile] Real-Time Runtime Metrics")
            print(f"  [1. Face Palm Validity]: Depth={palm_h:.3f} | RightHandOrientation={thumb_on_left} | AspectRatio={palm_ratio:.2f} -> Status: {palm_front_ok}")
            print(f"  [2. L-Spur Thumb Lock]: OuterAngle={thumb_middle_angle:.1f}° | IPJoint={thumb_ip_angle:.1f}° -> Status: {thumb_ok}")
            print(f"  [3. Twin Pillar Extension]: 2DAngle={idx_angle:.1f}° | LiftRatio={idx_up_ratio:.2f} -> Status: {idx_straight and idx_point_up}")
            print(f"  [4. 🌟 Foreshortening Ratio]: Index={ratio_idx:.3f} (Req: >{self.THRESHOLDS['min_index_length_ratio']}) -> Status: {index_length_ok}")
            print(f"  [5. Folded Clusters State]: Middle={mid_ok} | Ring={ring_ok} | Pinky={pinky_ok} -> Status: {fingers_ok}")
            print(f"❌/✅ Current Frame Result: {all_ok} | Stabilizer Buffer: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")

        if current_step == 1:
            return 1 if final_ok else 0
        else:
            return self.stroke_count