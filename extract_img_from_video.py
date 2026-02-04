import cv2
import os

def extract_frames(video_path: str, output_dir: str, interval: int = 30):
    """
    从视频中每隔指定帧数提取一帧图像
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        interval: 抽帧间隔
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 每隔interval帧保存一次
        if frame_count % interval == 0:
            # 生成输出文件名
            output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            # 保存图片
            cv2.imwrite(output_path, frame)
            saved_count += 1
            print(f"已保存第{frame_count}帧到: {output_path}")
        
        frame_count += 1
    
    cap.release()
    print(f"\n完成! 共处理{frame_count}帧，保存{saved_count}张图片。")

if __name__ == "__main__":
    video_path = "caotangvis3.mov"
    output_dir = "caotangvis3"
    extract_frames(video_path, output_dir)
