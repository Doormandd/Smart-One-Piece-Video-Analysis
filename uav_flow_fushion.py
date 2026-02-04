import cv2
import numpy as np
import json
from ultralytics import YOLO
from collections import defaultdict
import random

class UAVFlowGeo:
    def __init__(self, video_path: str, ref_path: str, homography_path: str, model_path: str):
        # 初始化视频捕获
        self.cap = cv2.VideoCapture(video_path)
        self.ref_img = cv2.imread(ref_path)
        
        # 加载单应性矩阵
        with open(homography_path, 'r') as f:
            self.H_init = np.array(json.load(f))
        
        # 初始化跟踪参数
        self.prev_frame = None
        self.prev_gray = None
        self.current_H = self.H_init.copy()
        
        # 创建显示窗口
        cv2.namedWindow('Original Frame')
        cv2.namedWindow('Warped Result')

        # 添加YOLO模型和轨迹相关属性
        self.model = YOLO(model_path)
        self.model.tracker = "bytetrack.yaml"  # 启用跟踪器
        self.tracks = defaultdict(list)  # 存储轨迹
        self.track_colors = {}  # 存储轨迹颜色

        # 添加VisDrone数据集的类别映射
        self.class_names = {
            0: 'pedestrian',
            1: 'people',
            2: 'bicycle',
            3: 'car',
            4: 'van',
            5: 'truck',
            6: 'tricycle',
            7: 'awning-tricycle',
            8: 'bus',
            9: 'motor'
        }

    def get_track_color(self, track_id):
        """为每个跟踪ID生成唯一的颜色"""
        if track_id not in self.track_colors:
            self.track_colors[track_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        return self.track_colors[track_id]

    def draw_tracks(self, image):
        """绘制所有目标轨迹"""
        for track_id, points in self.tracks.items():
            if len(points) > 1:
                color = self.get_track_color(track_id)
                for i in range(1, len(points)):
                    cv2.line(image,
                            (int(points[i-1][0]), int(points[i-1][1])),
                            (int(points[i][0]), int(points[i][1])),
                            color, 2)

    def compute_flow_transform(self, prev_gray, curr_gray):
        """使用光流法计算帧间变换"""
        # 检测特征点
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=3000,
                                         qualityLevel=0.01, minDistance=7)
        
        if prev_pts is None:
            return np.eye(3)
        
        # 计算光流
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # 筛选好的匹配点
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]
        
        if len(good_prev) < 10:
            return np.eye(3)
        
        # 计算变换矩阵
        H, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
        return H if H is not None else np.eye(3)

    def run(self):
        ret, first_frame = self.cap.read()
        if not ret:
            return
        
        # 处理第一帧
        first_warped = cv2.warpPerspective(
            first_frame, self.H_init,
            (self.ref_img.shape[1], self.ref_img.shape[0])
        )
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 目标检测和跟踪
            results = self.model.track(frame, persist=True, verbose=False)[0]
            
            # 光流计算和变换
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H_flow = self.compute_flow_transform(self.prev_gray, curr_gray)
            self.current_H = self.current_H @ np.linalg.inv(H_flow)
            
            # 应用变换
            warped = cv2.warpPerspective(
                frame, self.current_H,
                (self.ref_img.shape[1], self.ref_img.shape[0])
            )
            
            # 创建结果图像
            result = self.ref_img.copy()
            
            # 处理检测结果
            if hasattr(results, 'boxes') and results.boxes.id is not None:
                boxes_data = results.boxes.data.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy()
                
                for box, track_id in zip(boxes_data, track_ids):
                    x1, y1, x2, y2, conf, cls_id = box[:6]
                    if conf < 0.3:  # 只保留置信度判断
                        continue
                    
                    # 修正：确保以浮点数形式处理坐标
                    box_pts = np.array([
                        [float(x1), float(y1)],
                        [float(x2), float(y1)],
                        [float(x2), float(y2)],
                        [float(x1), float(y2)]
                    ], dtype=np.float32).reshape(-1, 1, 2)
                    
                    # 变换检测框
                    transformed_box = cv2.perspectiveTransform(box_pts, self.current_H)
                    transformed_box = transformed_box.reshape(-1, 2)
                    
                    # 绘制检测框，使用更粗的线条
                    color = self.get_track_color(track_id)
                    box_points = np.int32(transformed_box)
                    cv2.polylines(result, [box_points], True, color, 3)
                    
                    # 计算变换后的中心点（使用变换后的边界框计算）
                    center_x = np.mean(transformed_box[:, 0])
                    center_y = transformed_box[2, 1]  # 使用底边中点
                    
                    # 存储轨迹点
                    track_id = int(track_id)
                    self.tracks[track_id].append([center_x, center_y])
                    
                    # 修改标签显示逻辑
                    class_id = int(cls_id)
                    class_name = self.class_names.get(class_id, f'class_{class_id}')
                    label = f"car #{track_id}"
                    cv2.putText(result, label,
                              (int(center_x), int(center_y)-20),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            # 绘制轨迹
            self.draw_tracks(result)
            
            # 叠加原始帧
            result = cv2.addWeighted(result, 0.7, warped, 0.3, 0)
            
            # 显示结果
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Warped Result', result)
            
            # 更新前一帧
            self.prev_gray = curr_gray.copy()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    uav_geo = UAVFlowGeo(
        video_path='caotangvis.mov',
        ref_path='uav_road_ref.png',
        homography_path='h.json',
        model_path='best.pt'  # 添加YOLO模型路径
    )
    uav_geo.run()
