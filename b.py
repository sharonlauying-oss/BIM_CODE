# ==============================================================================
# 🤟 BIM "B" Hand Gesture Description:
# - Hand Orientation: Right hand, palm facing the CAMERA directly.
# - Four Fingers (Index, Middle, Ring, Pinky): Fully straight, pointing vertically upward,
#   and tightly closed together side-by-side to form a flat plane.
# - Thumb: Crossed over and pressed flat against the inner palm layer area.
# ==============================================================================

import math
from gestures.base_gesture import BaseGesture

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("B")
        self.consecutive_correct = 0
        self.required_consecutive = 2  # Requires 2 consecutive correct frames to pass
        self.stroke_count = 1
        
        # 🐛 Enable explicit runtime console diagnostics
        self.enable_debug_log = False 
        
        # 🎯 Thresholds calibrated to your gesture telemetry data stream
        self.THRESHOLDS = {
            "min_palm_height": 0.06,        # Minimum palm height to filter out small/distant hands
            "finger_straight_angle": 140.0, # Minimum 2D angle for a finger to be considered straight
            
            # 📏 🌟 Perspective Foreshortening Ratios (Self-Adaptive Segment Length Ratios)
            # Prevents forward bending by confirming that the 2D projected finger lengths remain full.
            "min_index_length_ratio": 0.42,   # (Dist 6 to 8) / hand_size
            "min_middle_length_ratio": 0.45,  # (Dist 10 to 12) / hand_size
            "min_ring_length_ratio": 0.42,    # (Dist 14 to 16) / hand_size
            "min_pinky_length_ratio": 0.38,   # (Dist 18 to 20) / hand_size
            
            # 🤝 Cohesiveness Lock parameters
            "tip_spacing_max": 0.04,        # Maximum 3D distance between adjacent finger tips
            "pinky_spacing_max": 0.045,     # Maximum 3D distance allowed for the pinky gap
            "palm_ratio": 0.35,             # Minimum palm aspect ratio for front-facing check
            
            # 👍 Thumb Position and Alignment Locks
            "thumb_straight_angle": 140.0,  # Minimum angle for the thumb joints to be straight
            "thumb_angle_min": 35.0,        # Minimum thumb inclination angle
            "thumb_angle_max": 75.0,        # Maximum thumb inclination angle
            "thumb_to_palm_max": 0.09       # Maximum distance from thumb tip to palm center
        }
    
    def _get_distance(self, p1, p2):
        # Calculate 2D distance on the screen pixel/image coordinate plane
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
    
    def _get_world_distance(self, p1, p2):
        # Calculate 3D physical distance in meters (independent of camera distance)
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def _get_angle(self, p1, p2, p3):
        # Calculate the 2D angle (in degrees) formed by three landmarks at p2
        v1x, v1y = p1.x - p2.x, p1.y - p2.y
        v2x, v2y = p3.x - p2.x, p3.y - p2.y
        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0: return 0.0
        cos = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos))
    
    def reset_dynamic_tracking(self):
        # Reset tracking state and consecutive frame counter memory states
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
        
        # Base validation: Filter out hands that are too small or far away
        palm_h = self._get_world_distance(wm[0], wm[9])
        if palm_h < self.THRESHOLDS["min_palm_height"]:
            self.consecutive_correct = 0
            return False if current_step == 1 else 0
        
        # 1. Four fingers check: Verify Index, Middle, Ring, and Pinky joint flexion angles
        idx_angle = self._get_angle(lm[5], lm[6], lm[8])
        mid_angle = self._get_angle(lm[9], lm[10], lm[12])
        rng_angle = self._get_angle(lm[13], lm[14], lm[16])
        pky_angle = self._get_angle(lm[17], lm[18], lm[20])
        
        all_fingers_straight = (idx_angle > self.THRESHOLDS["finger_straight_angle"] and
                                mid_angle > self.THRESHOLDS["finger_straight_angle"] and
                                rng_angle > self.THRESHOLDS["finger_straight_angle"] and
                                pky_angle > self.THRESHOLDS["finger_straight_angle"])
        
        # 2. Orientation check: Ensure all 4 finger tips point upwards relative to knuckles
        all_fingers_up = (lm[8].y < lm[5].y and 
                          lm[12].y < lm[9].y and 
                          lm[16].y < lm[13].y and 
                          lm[20].y < lm[17].y)
        
        # 3. 🌟 Perspective Anti-Flattening Guard (Foreshortening Length Ratio Check)
        # Binds 2D projected segment lengths to catch forward-bending cheating vectors across all 4 tracks
        ratio_idx = self._get_distance(lm[6], lm[8]) / hand_size
        ratio_mid = self._get_distance(lm[10], lm[12]) / hand_size
        ratio_rng = self._get_distance(lm[14], lm[16]) / hand_size
        ratio_pky = self._get_distance(lm[18], lm[20]) / hand_size

        lengths_ok = (ratio_idx >= self.THRESHOLDS["min_index_length_ratio"] and
                      ratio_mid >= self.THRESHOLDS["min_middle_length_ratio"] and
                      ratio_rng >= self.THRESHOLDS["min_ring_length_ratio"] and
                      ratio_pky >= self.THRESHOLDS["min_pinky_length_ratio"])
        
        # 4. Column Cohesiveness Check: Ensure the 4 extended fingers are tightly closed together
        index_middle_dist = self._get_world_distance(wm[8], wm[12])
        middle_ring_dist = self._get_world_distance(wm[12], wm[16])
        ring_pinky_dist = self._get_world_distance(wm[16], wm[20])
        
        fingers_closed = (index_middle_dist < self.THRESHOLDS["tip_spacing_max"] and
                          middle_ring_dist < self.THRESHOLDS["tip_spacing_max"] and
                          ring_pinky_dist < self.THRESHOLDS["pinky_spacing_max"])
        
        # 5. Thumb Position Check: Must be straight, correctly oriented, and tucked across the palm plane
        thumb_straight = self._get_angle(lm[2], lm[3], lm[4]) > self.THRESHOLDS["thumb_straight_angle"]
        
        thumb_dx = lm[4].x - lm[2].x
        thumb_dy = lm[4].y - lm[2].y
        thumb_angle = math.degrees(math.atan2(thumb_dx, -thumb_dy))
        is_thumb_correct_angle = self.THRESHOLDS["thumb_angle_min"] < thumb_angle < self.THRESHOLDS["thumb_angle_max"]
        
        is_thumb_near_palm = self._get_world_distance(wm[4], wm[9]) < self.THRESHOLDS["thumb_to_palm_max"]
        thumb_ok = thumb_straight and is_thumb_correct_angle and is_thumb_near_palm
        
        # 6. Palm Orientation check: Ensure the palm faces the camera directly
        palm_w = self._get_distance(lm[5], lm[17])
        palm_h_2d = self._get_distance(lm[0], lm[9])
        palm_aspect_ratio = palm_w / palm_h_2d if palm_h_2d > 0 else 0
        is_palm_facing = palm_aspect_ratio > self.THRESHOLDS["palm_ratio"]
        
        # 7. Integrated Execution Pipeline Convergence
        current_frame_correct = (all_fingers_straight and all_fingers_up and lengths_ok and 
                                 fingers_closed and thumb_ok and is_palm_facing)
        
        # Zero-latency stabilization buffer calculation
        if current_frame_correct:
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0
        
        gesture_valid = self.consecutive_correct >= self.required_consecutive
        
        if self.enable_debug_log:
            print("\n============================================================")
            print(f"📊 [B Gesture Profile] Real-Time Runtime Metrics")
            print(f"  [1. Base Palm Validity]: Depth={palm_h:.3f} | FrontFacingRatio={palm_aspect_ratio:.2f} -> Status: {is_palm_facing}")
            print(f"  [2. Extension Angles]: Index={idx_angle:.1f}° | Middle={mid_angle:.1f}° | Ring={rng_angle:.1f}° | Pinky={pky_angle:.1f}° -> Status: {all_fingers_straight and all_fingers_up}")
            print(f"  [3. 🌟 Foreshortening Ratios]: Index={ratio_idx:.3f} (Req: >{self.THRESHOLDS['min_index_length_ratio']})")
            print(f"                               Middle={ratio_mid:.3f} (Req: >{self.THRESHOLDS['min_middle_length_ratio']})")
            print(f"                               Ring={ratio_rng:.3f} (Req: >{self.THRESHOLDS['min_ring_length_ratio']})")
            print(f"                               Pinky={ratio_pky:.3f} (Req: >{self.THRESHOLDS['min_pinky_length_ratio']}) -> Status: {lengths_ok}")
            print(f"  [4. Column Cohesiveness]: Idx-Mid={index_middle_dist:.3f} | Mid-Rng={middle_ring_dist:.3f} | Rng-Pky={ring_pinky_dist:.3f} -> Status: {fingers_closed}")
            print(f"  [5. Flat Thumb Lock]: JointAngle={self._get_angle(lm[2], lm[3], lm[4]):.1f}° | Inclination={thumb_angle:.1f}° | PalmProximity={self._get_world_distance(wm[4], wm[9]):.3f} -> Status: {thumb_ok}")
            print(f"❌/✅ Current Frame Result: {current_frame_correct} | Stabilizer Buffer: {self.consecutive_correct}/{self.required_consecutive}")
            print("============================================================\n")
        
        return gesture_valid if current_step == 1 else self.stroke_count