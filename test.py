# gestures/test.py
import math
import csv
import time
import os
from datetime import datetime
from gestures.base_gesture import BaseGesture
from core.vision_engine import SmoothedPoint

class GestureChecker(BaseGesture):
    def __init__(self):
        super().__init__("UNIVERSAL_DATA_COLLECTOR")
        
        # ====================== 核心配置 ======================
        self.MAX_HISTORY_FRAMES = 1000  # 最大缓存帧数（约30秒）
        self.DATA_SAVE_DIR = "collected_data"  # 数据保存目录
        
        # ====================== 采集状态控制 ======================
        self.is_collecting = False  # 连续采集开关
        self.single_capture = False # 单次捕获开关
        self.frame_count = 0        # 当前采集帧数
        
        # ====================== 数据缓存 ======================
        self.full_data_log = []     # 全量数据缓存
        self.landmark_history = []  # 关键点历史（用于计算动态特征）
        
        # ====================== 预定义常量 ======================
        # 所有21个关键点的名称（对应MediaPipe索引）
        self.LANDMARK_NAMES = [
            "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
            "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
            "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]
        
        # 需要计算角度的关节（起点-顶点-终点）
        self.ANGLE_JOINTS = [
            ("THUMB_MCP", "THUMB_CMC", "WRIST"),
            ("THUMB_IP", "THUMB_MCP", "THUMB_CMC"),
            ("THUMB_TIP", "THUMB_IP", "THUMB_MCP"),
            ("INDEX_PIP", "INDEX_MCP", "WRIST"),
            ("INDEX_DIP", "INDEX_PIP", "INDEX_MCP"),
            ("INDEX_TIP", "INDEX_DIP", "INDEX_PIP"),
            ("MIDDLE_PIP", "MIDDLE_MCP", "WRIST"),
            ("MIDDLE_DIP", "MIDDLE_PIP", "MIDDLE_MCP"),
            ("MIDDLE_TIP", "MIDDLE_DIP", "MIDDLE_PIP"),
            ("RING_PIP", "RING_MCP", "WRIST"),
            ("RING_DIP", "RING_PIP", "RING_MCP"),
            ("RING_TIP", "RING_DIP", "RING_PIP"),
            ("PINKY_PIP", "PINKY_MCP", "WRIST"),
            ("PINKY_DIP", "PINKY_PIP", "PINKY_MCP"),
            ("PINKY_TIP", "PINKY_DIP", "PINKY_PIP")
        ]
        
        # 需要计算的关键距离
        self.KEY_DISTANCES = [
            ("THUMB_TIP", "INDEX_TIP"), ("THUMB_TIP", "INDEX_MCP"),
            ("THUMB_TIP", "WRIST"), ("INDEX_TIP", "WRIST"),
            ("MIDDLE_TIP", "WRIST"), ("RING_TIP", "WRIST"),
            ("PINKY_TIP", "WRIST"), ("INDEX_MCP", "PINKY_MCP")
        ]
        
        # 自动创建数据保存目录
        os.makedirs(self.DATA_SAVE_DIR, exist_ok=True)
        
        print("\n" + "="*70)
        print("✅ 通用全量数据采集器已加载")
        print("📌 快捷键说明：")
        print("   [S] 开始/停止 连续采集（录制动态动作）")
        print("   [空格] 单次捕获（保存当前静态姿势）")
        print("   [R] 重置所有采集数据")
        print("="*70 + "\n")

    # ====================== UI 控制接口 ======================
    def toggle_collection(self):
        """切换连续采集状态（S键触发）"""
        self.is_collecting = not self.is_collecting
        if self.is_collecting:
            self.frame_count = 0
            self.full_data_log = []
            self.landmark_history = []
            print(f"\n🚀 开始连续采集... 按 [S] 停止")
        else:
            print(f"\n🛑 停止采集，共捕获 {self.frame_count} 帧数据")
            self._save_full_data_to_csv()

    def trigger_single_capture(self):
        """单次捕获当前帧（空格键触发）"""
        self.single_capture = True
        print("\n📸 单次捕获触发")

    def reset_all(self):
        """重置所有数据（R键触发）"""
        self.is_collecting = False
        self.single_capture = False
        self.frame_count = 0
        self.full_data_log = []
        self.landmark_history = []
        print("\n🔄 所有采集数据已重置")

    # ====================== 核心计算方法 ======================
    def _get_world_distance(self, p1, p2):
        """计算3D世界坐标距离（单位：米）"""
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    def _get_2d_distance(self, p1, p2):
        """计算2D像素坐标距离"""
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

    def _get_angle(self, p1, p2, p3):
        """计算三点形成的角度（单位：度）"""
        v1x, v1y = p1.x-p2.x, p1.y-p2.y
        v2x, v2y = p3.x-p2.x, p3.y-p2.y
        dot = v1x*v2x + v1y*v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        cos_angle = max(min(dot/(mag1*mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_angle))

    def _calculate_dynamic_features(self, lm_3d, frame_interval=0.033):
        """计算动态特征：速度、加速度"""
        features = {"speed_mps": 0.0, "acceleration_mps2": 0.0}
        
        if len(self.landmark_history) < 2:
            return features
            
        # 计算拇指指尖的速度（最能代表手部动作）
        prev_thumb = self.landmark_history[-1]["THUMB_TIP"]
        curr_thumb = lm_3d[self.LANDMARK_NAMES.index("THUMB_TIP")]
        distance = self._get_world_distance(prev_thumb, curr_thumb)
        features["speed_mps"] = distance / frame_interval
        
        if len(self.landmark_history) >= 3:
            prev_prev_thumb = self.landmark_history[-2]["THUMB_TIP"]
            prev_distance = self._get_world_distance(prev_prev_thumb, prev_thumb)
            prev_speed = prev_distance / frame_interval
            features["acceleration_mps2"] = (features["speed_mps"] - prev_speed) / frame_interval
            
        return features

    # ====================== 数据采集与保存 ======================
    def _collect_single_frame_data(self, hands_list):
        """采集单帧完整数据"""
        if not hands_list:
            return None
            
        hand = hands_list[0]
        lm_2d = hand["landmarks"]
        lm_3d = hand["world_landmarks"]
        
        # 基础元数据
        frame_data = {
            "timestamp": time.time(),
            "frame_number": self.frame_count,
            "hand_type": hand["real_hand"],
            "palm_height_m": self._get_world_distance(lm_3d[0], lm_3d[9]),
            "palm_width_m": self._get_world_distance(lm_3d[5], lm_3d[17])
        }
        frame_data["palm_aspect_ratio"] = frame_data["palm_width_m"] / frame_data["palm_height_m"] if frame_data["palm_height_m"] > 0 else 0.0
        
        # 1. 采集所有21个关键点的2D坐标
        for i, name in enumerate(self.LANDMARK_NAMES):
            frame_data[f"{name}_2d_x"] = lm_2d[i].x
            frame_data[f"{name}_2d_y"] = lm_2d[i].y
            frame_data[f"{name}_2d_z"] = lm_2d[i].z
            
        # 2. 采集所有21个关键点的3D世界坐标（核心！不受远近影响）
        for i, name in enumerate(self.LANDMARK_NAMES):
            frame_data[f"{name}_3d_x"] = lm_3d[i].x
            frame_data[f"{name}_3d_y"] = lm_3d[i].y
            frame_data[f"{name}_3d_z"] = lm_3d[i].z
            
        # 3. 计算所有关节角度
        for joint_name, a, b, c in [(f"{j1}_{j2}_{j3}_angle", j1, j2, j3) for j1,j2,j3 in self.ANGLE_JOINTS]:
            idx_a = self.LANDMARK_NAMES.index(a)
            idx_b = self.LANDMARK_NAMES.index(b)
            idx_c = self.LANDMARK_NAMES.index(c)
            frame_data[joint_name] = self._get_angle(lm_2d[idx_a], lm_2d[idx_b], lm_2d[idx_c])
            
        # 4. 计算所有关键距离（3D世界坐标）
        for dist_name, a, b in [(f"{a}_to_{b}_m", a, b) for a,b in self.KEY_DISTANCES]:
            idx_a = self.LANDMARK_NAMES.index(a)
            idx_b = self.LANDMARK_NAMES.index(b)
            frame_data[dist_name] = self._get_world_distance(lm_3d[idx_a], lm_3d[idx_b])
            
        # 5. 计算动态特征
        dynamic_features = self._calculate_dynamic_features(lm_3d)
        frame_data.update(dynamic_features)
        
        # 保存到历史缓存
        self.landmark_history.append({name: lm_3d[i] for i, name in enumerate(self.LANDMARK_NAMES)})
        if len(self.landmark_history) > self.MAX_HISTORY_FRAMES:
            self.landmark_history.pop(0)
            
        return frame_data

    def _save_full_data_to_csv(self):
        """将所有采集的数据保存为CSV文件"""
        if not self.full_data_log:
            print("⚠️ 没有可保存的数据！")
            return
            
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hand_data_{timestamp}.csv"
        filepath = os.path.join(self.DATA_SAVE_DIR, filename)
        
        # 获取所有字段名（按逻辑排序）
        fieldnames = list(self.full_data_log[0].keys())
        
        # 写入CSV
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.full_data_log)
            
        print(f"✅ 数据已保存至：{filepath}")
        print(f"📊 共保存 {len(self.full_data_log)} 帧，{len(fieldnames)} 个特征")

    # ====================== 主检测入口（系统自动调用） ======================
    def check_gesture(self, hands_list: list, current_step: int):
        # 1. 单次捕获处理
        if self.single_capture:
            frame_data = self._collect_single_frame_data(hands_list)
            if frame_data:
                self.full_data_log.append(frame_data)
                print(f"📸 第 {len(self.full_data_log)} 帧静态数据已捕获")
                self._save_full_data_to_csv()  # 单次捕获立即保存
            self.single_capture = False
            
        # 2. 连续采集处理
        if self.is_collecting:
            frame_data = self._collect_single_frame_data(hands_list)
            if frame_data:
                self.full_data_log.append(frame_data)
                self.frame_count += 1
                # 每10帧打印一次状态
                if self.frame_count % 10 == 0:
                    print(f"⏳ 正在采集... 已捕获 {self.frame_count} 帧")
                    
        # 3. 永远返回False/0，让系统停留在当前页面（纯采集模式）
        return False if current_step == 1 else 0

    # ====================== 状态重置 ======================
    def reset_dynamic_tracking(self):
        super().reset_dynamic_tracking()
        self.reset_all()