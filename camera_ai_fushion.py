import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import random
import json

class VPSSystem:
    def __init__(self, video_path: str, points_path: str, ref_image_path: str, model_path: str):
        # 初始化视频捕获
        self.cap = cv2.VideoCapture(video_path)
        self.ref_img = cv2.imread(ref_image_path)
        
        # 加载YOLO模型
        self.model = YOLO(model_path)
        self.H = self.compute_homography('h_camera.json')
        
        # 创建显示窗口
        cv2.namedWindow('Source Frame')
        cv2.namedWindow('VPS Result')
        
        # 轨迹跟踪相关
        self.tracks = defaultdict(list)
        self.track_colors = {}
        self.alpha = 0.7

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('dt.mov', fourcc, 30.0, 
                                 (self.ref_img.shape[1], self.ref_img.shape[0]))

    def compute_homography(self, json_path: str) -> np.ndarray:
        """从h.json读取单应性矩阵"""
        try:
            with open(json_path, 'r') as f:
                h_list = json.load(f)
                # 将列表转换回numpy数组
                H = np.array(h_list)
                print("成功加载变换矩阵")
                return H
        except Exception as e:
            print(f"读取变换矩阵失败: {e}")
            # 返回单位矩阵作为默认值
            return np.eye(3)

    def transform_bbox(self, bbox, H):
        """转换边界框坐标"""
        # 提取边界框的四个角点
        x1, y1, x2, y2 = bbox
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ])
        
        # 应用变换
        transformed = (H @ corners.T).T
        transformed = transformed / transformed[:, 2:3]
        
        # 返回变换后的边界框坐标
        min_x = np.min(transformed[:, 0])
        min_y = np.min(transformed[:, 1])
        max_x = np.max(transformed[:, 0])
        max_y = np.max(transformed[:, 1])
        
        return [int(min_x), int(min_y), int(max_x), int(max_y)]

    def warp_frame(self, frame):
        """将视频帧变换到参考图像坐标系"""
        return cv2.warpPerspective(
            frame,
            self.H,
            (self.ref_img.shape[1], self.ref_img.shape[0])
        )

    def rotate_frame(self, frame):
        """向右旋转90度"""
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    def get_track_color(self, track_id):
        """为每个跟踪ID生成唯一的颜色"""
        if track_id not in self.track_colors:
            self.track_colors[track_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        return self.track_colors[track_id]

    def draw_tracks(self, result_img):
        """绘制所有对象的轨迹"""
        for track_id, points in self.tracks.items():
            if len(points) > 1:
                color = self.get_track_color(track_id)
                # 绘制轨迹线
                for i in range(1, len(points)):
                    cv2.line(result_img,
                            (int(points[i-1][0]), int(points[i-1][1])),
                            (int(points[i][0]), int(points[i][1])),
                            color, 2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 只进行旋转，不再进行稳定
            frame = self.rotate_frame(frame)
            
            # 直接使用旋转后的帧进行处理
            results = self.model.track(frame, persist=True, verbose=False)[0]
            
            # 创建结果图像副本并添加半透明视频帧
            result_img = self.ref_img.copy()
            warped_frame = self.warp_frame(frame)
            result_img = cv2.addWeighted(result_img, 1-self.alpha, warped_frame, self.alpha, 0)
            
            if hasattr(results, 'boxes') and results.boxes.id is not None:
                boxes_data = results.boxes.data.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy()
                
                for box, track_id in zip(boxes_data, track_ids):
                    try:
                        x1, y1, x2, y2, conf, cls_id = box[:6]
                        if conf < 0.3:
                            continue
                        
                        track_id = int(track_id)
                        transformed_bbox = self.transform_bbox(
                            [float(x1), float(y1), float(x2), float(y2)],
                            self.H
                        )
                        
                        # 更新轨迹
                        center = ((transformed_bbox[0] + transformed_bbox[2])//2,
                                transformed_bbox[3])  # 使用边界框底部中心点
                        self.tracks[track_id].append(center)
                        
                        # 限制轨迹长度
                        max_trail_points = 50
                        if len(self.tracks[track_id]) > max_trail_points:
                            self.tracks[track_id] = self.tracks[track_id][-max_trail_points:]
                        
                        # 绘制检测框和标签
                        color = self.get_track_color(track_id)
                        cv2.rectangle(result_img, 
                                    (transformed_bbox[0], transformed_bbox[1]),
                                    (transformed_bbox[2], transformed_bbox[3]),
                                    color, 2)
                        
                        label = f"car #{track_id}"
                        
                        cv2.putText(result_img, label,
                                  (transformed_bbox[0], transformed_bbox[1]-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                    except Exception as e:
                        print(f"处理检测框时出错: {e}")
                        continue
            
            # 绘制轨迹
            self.draw_tracks(result_img)
            
            # 写入视频帧
            self.out.write(result_img)
            
            # 显示结果
            cv2.imshow('Source Frame', frame)
            cv2.imshow('VPS Result', result_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.alpha = min(1.0, self.alpha + 0.1)
            elif key == ord('s'):
                self.alpha = max(0.0, self.alpha - 0.1)
        
        # 释放资源
        self.out.release()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    vps = VPSSystem(
        video_path="cross.MOV",  # 使用预处理后的稳定视频
        points_path="points.txt",
        ref_image_path="ref.png",
        model_path="best.pt"
    )
    vps.run()
