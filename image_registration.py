import cv2
import numpy as np
import json

class ImageRegistration:
    def __init__(self, source_path: str, reference_path: str):
        # 读取图像
        self.source_img = cv2.imread(source_path)
        self.reference_img = cv2.imread(reference_path)
        
        # 存储点击的特征点
        self.source_points = []
        self.reference_points = []
        
        # 当前选择模式（source或reference）
        self.current_mode = 'source'
        
        # 创建窗口
        cv2.namedWindow('Source Image')
        cv2.namedWindow('Reference Image')
        cv2.namedWindow('Registration Result')
        
        cv2.setMouseCallback('Source Image', self.mouse_callback_source)
        cv2.setMouseCallback('Reference Image', self.mouse_callback_reference)
        
        # 存储变换后的图像
        self.warped_img = None
        
    def mouse_callback_source(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_mode == 'source':
            self.source_points.append([x, y])
            # 绘制点
            cv2.circle(self.source_img, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(self.source_img, str(len(self.source_points)), (x+5, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.current_mode = 'reference'
            print(f"选择源图像点 {len(self.source_points)}: ({x}, {y})")
    
    def mouse_callback_reference(self, event, x, y, flags, param):
        """更新参考图像的点击处理"""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_mode == 'reference':
            self.reference_points.append([x, y])
            # 在原始位置绘制点
            cv2.circle(self.reference_img, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(self.reference_img, str(len(self.reference_points)), 
                       (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            self.current_mode = 'source'
            print(f"选择参考图像点 {len(self.reference_points)}: ({x}, {y})")
            
            # 如果有足够的点对，更新变换
            if len(self.source_points) >= 4 and len(self.source_points) == len(self.reference_points):
                self.update_transform()
    
    def save_homography(self, H: np.ndarray):
        """保存单应性矩阵到json文件"""
        # 将numpy数组转换为列表
        h_list = H.tolist()
        
        # 保存到json文件
        with open('h.json', 'w') as f:
            json.dump(h_list, f, indent=2)
        print("\n变换矩阵已保存到h.json")

    def update_transform(self):
        if len(self.source_points) >= 4:
            # 计算单应性矩阵
            H, _ = cv2.findHomography(
                np.float32(self.source_points),
                np.float32(self.reference_points),
                cv2.RANSAC
            )
            
            # 保存变换矩阵
            self.save_homography(H)
            
            # 应用变换
            self.warped_img = cv2.warpPerspective(
                self.source_img,
                H,
                (self.reference_img.shape[1], self.reference_img.shape[0])
            )
            
            # 创建半透明叠加效果
            alpha = 0.5
            self.blended = cv2.addWeighted(
                self.reference_img,
                1-alpha,
                self.warped_img,
                alpha,
                0
            )
    
    def run(self):
        while True:
            # 显示原始图像
            cv2.imshow('Source Image', self.source_img)
            cv2.imshow('Reference Image', self.reference_img)
            
            # 显示配准结果
            if self.warped_img is not None:
                cv2.imshow('Registration Result', self.blended)
            
            # 显示当前模式
            mode_text = "当前选择: " + ("源图像" if self.current_mode == 'source' else "参考图像")
            print(f"\r{mode_text}", end="")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # 重置
                self.source_points = []
                self.reference_points = []
                self.source_img = cv2.imread('cross/frame_0000.jpg')
                self.reference_img = cv2.imread('ref.png')
                self.warped_img = None
                self.current_mode = 'source'
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    source_path = "caotangvis/frame_0000.jpg"
    reference_path = "uav_road_ref.png"
    registration = ImageRegistration(source_path, reference_path)
    registration.run()
