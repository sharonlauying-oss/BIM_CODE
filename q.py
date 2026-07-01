# ==============================================================================
# 🤟 BIM "Q" Gesture Description:
# - Hand Orientation: Palm facing DOWNWARD, showing a side profile to the camera.
# - Hand Direction: The main hand extension axis points straight to the RIGHT.
# - Active Fingers (Thumb & Index): Pointing straight DOWNWARD into the screen.
# - Closed Fingers (Middle, Ring, Pinky): Fully curled and tucked tightly into the palm.
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("Q")
        self.consecutive_correct = 0
        self.required_consecutive = 2
        self.stroke_count = 1

        # 🐛 Enable runtime debug logging
        self.enable_debug_log = False 

        # 🎯 Dynamic threshold ranges precisely tailored from your latest 9 log streams
        self.THRESHOLDS = {
            "min_palm_height": 0.03,       
            
            # 🌟 Global Hand Orientation
            # explanation: Covers standard downward postures (10° to 160°) AND lateral profiles (310° to 340°)
            "hand_direction_angle_min": 10, 
            "hand_direction_angle_max": 160,  
            "hand_direction_lateral_min": 310, # Catches the exact 316.8° ~ 326.9° drift from your logs
            "hand_direction_lateral_max": 340,
            
            # Active Beak Group (Index & Thumb)
            "index_bend_min": 170,         
            "thumb_straight_min": 170,     
            
            # 🌟 Folded Defense Group (Middle, Ring, Pinky)
            # explanation: Your middle finger hit 126.3°. Loosened to 135 to ensure absolute passage.
            "folded_finger_angle_max": 135, 
            # explanation: Your middle distance spiked to 0.622 due to perspective flat-outs. Expanded to 0.65.
            "folded_tip_to_mcp_ratio": 0.65 
        }

    def _dist(self, p1, p2):
        # explanation: Computes the 2D Euclidean distance on the camera viewport
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _wdist(self, p1, p2):
        # explanation: Computes the 3D absolute physical distance in world coordinates
        return math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)

    def _angle(self, p1, p2, p3):
        # explanation: Computes the 2D bending angle with p2 as the vertex joint
        v1x = p1.x - p2.x
        v1y = p1.y - p2.y
        v2x = p3.x - p2.x
        v2y = p3.y - p2.y
        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0: return 0
        return math.degrees(math.acos(max(min(dot / (mag1 * mag2), 1.0), -1.0)))

    def reset_dynamic_tracking(self):
        # explanation: Flushes the debouncing buffer if tracking is dropped
        super().reset_dynamic_tracking()
        self.consecutive_correct = 0

    def check_gesture(self, hands_list: list, current_step: int):
        if not hands_list:
            self.consecutive_correct = 0
            return 0

        hand = hands_list[0]
        lm = hand["landmarks"]
        wm = hand["world_landmarks"]

        hand_size = self._dist(lm[0], lm[9])
        if hand_size == 0: return 0

        # 1. Base Palm Size Validation
        palm_h = self._wdist(wm[0], wm[9])
        palm_valid = palm_h >= self.THRESHOLDS["min_palm_height"]

        # 2. Global Hand Rotation Angle Tracking
        dx = lm[9].x - lm[0].x
        dy = -(lm[9].y - lm[0].y) 
        global_angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # explanation: Permits normal downward sector OR the lateral flat-out 310°~340° sector
        direction_ok = ((self.THRESHOLDS["hand_direction_angle_min"] <= global_angle <= self.THRESHOLDS["hand_direction_angle_max"]) or
                        (self.THRESHOLDS["hand_direction_lateral_min"] <= global_angle <= self.THRESHOLDS["hand_direction_lateral_max"]))

        # 3. Active Beak Group Inspection (Index & Thumb State)
        index_angle = self._angle(lm[5], lm[6], lm[8])
        index_ok = index_angle >= self.THRESHOLDS["index_bend_min"]

        thumb_angle = self._angle(lm[2], lm[3], lm[4])
        thumb_ok = thumb_angle >= self.THRESHOLDS["thumb_straight_min"]

        # 4. Folded Defense Group Inspection (Middle, Ring, Pinky State)
        mid_angle = self._angle(lm[10], lm[11], lm[12])
        ring_angle = self._angle(lm[14], lm[15], lm[16])
        pky_angle = self._angle(lm[18], lm[19], lm[20])
        
        # Joint flexion check
        angles_ok = (mid_angle < self.THRESHOLDS["folded_finger_angle_max"] and
                     ring_angle < self.THRESHOLDS["folded_finger_angle_max"] and
                     pky_angle < self.THRESHOLDS["folded_finger_angle_max"])

        # Normalized palm retraction check
        mid_dist = self._dist(lm[12], lm[9]) / hand_size
        ring_dist = self._dist(lm[16], lm[13]) / hand_size
        pky_dist = self._dist(lm[20], lm[17]) / hand_size
        
        dists_ok = (mid_dist < self.THRESHOLDS["folded_tip_to_mcp_ratio"] and
                    ring_dist < self.THRESHOLDS["folded_tip_to_mcp_ratio"] and
                    pky_dist < self.THRESHOLDS["folded_tip_to_mcp_ratio"])

        # explanation: Topological lock ensures the index tip (8) extends further along X than the middle tip (12)
        beak_extended_ok = lm[8].x > lm[12].x

        # 5. Integrated Execution Decision
        all_ok = (palm_valid and direction_ok and index_ok and thumb_ok and 
                  angles_ok and dists_ok and beak_extended_ok)

        # Time-series debouncing logic
        self.consecutive_correct = self.consecutive_correct + 1 if all_ok else 0
        final_ok = self.consecutive_correct >= self.required_consecutive

        if self.enable_debug_log:
            print("\n============================================================")
            print(f"📊 [Q Real Data Aligned Profile] Real-Time Runtime Metrics")
            print(f"  [1. Global Angle]: {global_angle:.1f}° -> Status: {direction_ok}")
            print(f"  [2. Index Angle]: {index_angle:.1f}° (Req: > {self.THRESHOLDS['index_bend_min']}°) -> {index_ok}")
            print(f"  [3. Thumb Angle]: {thumb_angle:.1f}° (Req: > {self.THRESHOLDS['thumb_straight_min']}°) -> {thumb_ok}")
            print(f"  [4. Folded Joint Flexion]:")
            print(f"      - Middle: {mid_angle:.1f}° | Pinky: {pky_angle:.1f}° (Req: < {self.THRESHOLDS['folded_finger_angle_max']}°) -> Status: {angles_ok}")
            print(f"  [5. Folded Palm Distance]: Middle={mid_dist:.3f}, Pinky={pky_dist:.3f} (Req: < {self.THRESHOLDS['folded_tip_to_mcp_ratio']:.3f}) -> Status: {dists_ok}")
            print(f"  [6. Index Protrusion Lock]: {beak_extended_ok}")
            print(f"❌/✅ Current Frame Result: {all_ok} | Stabilizer Buffer: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")

        if current_step == 1:
            return 1 if final_ok else 0
        else:
            return self.stroke_count