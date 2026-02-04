# Smart One Piece Video Analysis / 智能视频分析系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

---

## 项目简介 / Project Introduction

This project implements an **intelligent video analysis system** that combines multiple computer vision techniques for comprehensive video understanding and processing.

本项目实现了一个**智能视频分析系统**，结合多种计算机视觉技术，用于全面的视频理解和处理。

---

## 项目特点 / Features

- ✅ **Camera AI Fusion**: Camera AI integration system / **相机AI融合**：相机AI集成系统
- ✅ **UAV Optical Flow**: UAV video flow estimation / **UAV光流估计**：无人机视频流估计
- ✅ **Image Registration**: Manual image alignment tool / **图像配准**：手动图像配准工具
- ✅ **Video Frame Extraction**: Extract frames from video at intervals / **视频帧提取**：按间隔从视频提取帧
- ✅ **VPS Object Detection**: Video processing system / **VPS目标检测**：视频处理系统
- ✅ **YOLO Integration**: Object detection with YOLOv8 / **YOLO集成**：使用YOLOv8进行目标检测
- ✅ **Visualization**: Track drawing and trajectory display / **可视化**：轨迹绘制和轨迹显示

---

## 技术栈 / Technology Stack

### Core Technologies / 核心技术

| Technology / 技术 | Version / 版本 | Purpose / 用途 |
|----------------|---------------|-------------|
| **OpenCV** | 4.x | Computer vision library / 计算机视觉库 |
| **NumPy** | Latest | Numerical computing / 数值计算 |
| **PyTorch** | Latest | Deep learning framework / 深度学习框架 |
| **Ultralytics YOLOv8** | Latest | Object detection / 目标检测 |

### Algorithms / 算法

- **Optical Flow (LK)**: Sparse optical flow estimation / **光流法（LK）**：稀疏光流估计
- **Homography**: Image alignment and transformation / **单应性矩阵**：图像对齐和变换
- **Object Tracking**: ByteTrack algorithm / **目标跟踪**：ByteTrack算法
- **Camera AI**: Camera pose estimation / **相机AI**：相机姿态估计
- **Deep Learning**: YOLO object detection / **深度学习**：YOLO目标检测

---

## 项目结构 / Project Structure

```
smart_one_picutre/
├── camera_ai_fushion.py      # Camera AI fusion system
├── uav_flow_fushion.py        # UAV optical flow estimation
├── image_registration.py       # Image registration tool
├── extract_img_from_video.py  # Video frame extraction
├── caotangvis.mp4              # Test video (105MB)
├── cross.MOV                    # Cross-shot video (27MB)
├── uav_road_ref.png           # Reference image (1940x1536)
├── uav_flow_fushion.py         # UAV flow fusion (4.9MB)
└── .gitignore                   # Git ignore configuration
```

---

## 核心模块详解 / Core Modules

### 1. Camera AI Fusion (camera_ai_fushion.py)

**功能描述 / Features**:
- Video capture and processing / 视频捕获和处理
- YOLO model loading and inference / YOLO模型加载和推理
- Homography transformation from JSON / 从JSON加载单应性矩阵
- Bounding box coordinate transformation / 边界框坐标变换
- Video writing and visualization / 视频写入和可视化

**核心类 / Key Classes**:
- `VPSSystem`: Main processing class / 主要处理类
- Methods: `transform_bbox()`, `warp_frame()`, `rotate_frame()` / 方法

### 2. UAV Optical Flow (uav_flow_fushion.py)

**功能描述 / Features**:
- Video frame capture / 视频帧捕获
- VPS dataset class mapping / VPS数据集类别映射
- YOLO-based object tracking / 基于YOLO的目标跟踪
- Optical flow computation / 光流计算
- Track visualization with unique colors / 带唯一颜色的轨迹可视化

**核心类 / Key Classes**:
- `UAVFlowGeo`: Flow estimation class / 流估计类
- Methods: `compute_flow_transform()`, `get_track_color()`, `draw_tracks()` / 方法

### 3. Image Registration (image_registration.py)

**功能描述 / Features**:
- Manual point selection for image alignment / 手动点选择用于图像对齐
- Homography matrix computation / 单应性矩阵计算
- Interactive mouse callback / 交互式鼠标回调
- Save transformation to JSON / 保存变换到JSON

**核心类 / Key Classes**:
- `ImageRegistration`: Registration tool class / 配准工具类
- Methods: `mouse_callback_source()`, `mouse_callback_reference()`, `save_homography()` / 方法

### 4. Video Frame Extraction (extract_img_from_video.py)

**功能描述 / Features**:
- Extract frames from video at specified intervals / 按指定间隔从视频提取帧
- Automatic output directory creation / 自动创建输出目录
- Progress logging / 进度日志

**使用方法 / Usage**:
```python
python extract_img_from_video.py
```

---

## 数据集信息 / Dataset Information

### Test Videos / 测试视频

| File / 文件 | Size / 大小 | Description / 描述 |
|-------------|-----------|-------------|
| caotangvis.mp4 | 105MB | Test video for analysis / 测试视频 |
| cross.MOV | 27MB | Cross-shot scene video / 交叉场景视频 |

### Reference Image / 参考图像

- **File**: uav_road_ref.png
- **Size**: 1940 x 1536 pixels
- **Format**: PNG, RGBA
- **Purpose**: Reference image for UAV flow estimation / UAV流估计的参考图像

---

## 使用方法 / Usage

### 环境要求 / Requirements

```bash
# Python environment / Python环境
python >= 3.8

# Install dependencies / 安装依赖
pip install opencv-python
pip install numpy
pip install torch
pip install ultralytics
```

### 视频帧提取 / Video Frame Extraction

```python
# Extract frames every 30 frames
python extract_img_from_video.py
```

### 相机AI融合 / Camera AI Fusion

```python
from camera_ai_fushion import VPSSystem

# Initialize with video, reference image, homography, and model paths
vps = VPSSystem(
    video_path="caotangvis.mp4",
    points_path="h.json",
    ref_image_path="uav_road_ref.png",
    model_path="best.pt"
)

# Run processing
vps.run()
```

### UAV流估计 / UAV Flow Estimation

```python
from uav_flow_fushion import UAVFlowGeo

# Initialize with video and reference image
uav_flow = UAVFlowGeo(
    video_path="cross.MOV",
    ref_path="uav_road_ref.png",
    homography_path="h.json",
    model_path="best.pt"
)

# Run flow estimation
uav_flow.run()
```

### 图像配准 / Image Registration

```python
from image_registration import ImageRegistration

# Initialize with source and reference images
reg = ImageRegistration(
    source_path="source.jpg",
    reference_path="reference.jpg"
)

# Manual point selection (interactive)
# Click 4 corresponding points on source and reference images
# Homography matrix will be computed and saved to h.json
```

---

## 算法详解 / Algorithm Details

### 光流估计 / Optical Flow Estimation

**技术 / Technology**:
- **Algorithm**: Lucas-Kanade Optical Flow / **算法**：Lucas-Kanade光流
- **Features**: GoodFeaturesToTrack / **特征**：GoodFeaturesToTrack
- **Win Size**: 21x21 pixels / **窗口大小**：21x21像素
- **Max Level**: 3 / **最大金字塔层数**：3
- **Method**: RANSAC / **方法**：RANSAC for robust estimation / 鲁棒估计

**应用 / Applications**:
- Frame-to-frame motion estimation / 帧间运动估计
- UAV trajectory analysis / 无人机轨迹分析
- Video stabilization / 视频稳定

### 单应性变换 / Homography Transformation

**技术 / Technology**:
- **Algorithm**: RANSAC (RANdom SAmple Consensus) / **算法**：RANSAC（随机抽样一致）
- **Method**: FindHomography / **方法**：FindHomography
- **Purpose**: Image alignment and geometric transformation / **目的**：图像对齐和几何变换
- **Storage**: JSON format for easy loading / **存储**：JSON格式便于加载

**应用 / Applications**:
- Image stitching / 图像拼接
- Perspective correction / 透视校正
- Multi-view analysis / 多视角分析

---

## 可视化 / Visualization

### 轨迹绘制 / Track Drawing

- Unique colors for each track / 每个轨迹的唯一颜色
- Trajectory lines connecting track points / 连接轨迹点的轨迹线
- Bounding box visualization / 边界框可视化
- FourCC video output for compatibility / FourCC视频输出以保持兼容性

### 视频输出 / Video Output

- **Input Videos**:
  - caotangvis.mp4 (H.264 codec, 30fps)
  - cross.MOV (cross.MOV, 30fps)

- **Output Videos**:
  - dt.mov (fourcc: mp4v, 30fps, 30.0fps)
  - Resolution: Reference image size (1940x1536)

---

## 项目应用场景 / Project Applications

1. **无人机监控 / UAV Surveillance**:
   - 目标跟踪和轨迹分析
   - 流估计和运动检测
   - 相机AI融合

2. **智能交通 / Intelligent Transportation**:
   - 车辆检测和跟踪
   - 交通流分析
   - 场景理解和监控

3. **视频分析 / Video Analytics**:
   - 多视角视频融合
   - 运动模式识别
   - 异常检测

4. **图像配准 / Image Registration**:
   - 图像拼接和镶嵌
   - 多视角场景重建
   - 透视校正和变换

---

## 性能指标 / Performance Metrics

### 处理速度 / Processing Speed

- **视频帧提取**: ~30 frames/second / **Video Frame Extraction**: ~30帧/秒
- **光流计算**: Real-time capable / **Optical Flow**: 实时能力
- **YOLO推理**: Fast inference / **YOLO Inference**: 快速推理

### 视频质量 / Video Quality

- **测试视频**: 105MB, H.264, 30fps
- **交叉视频**: 27MB, cross.MOV, 30fps
- **输出视频**: mp4v codec, 30fps

---

## 未来改进方向 / Future Improvements

- [ ] **深度学习光流** / **Deep Learning Optical Flow**:
  - 使用深度学习模型替代传统光流
  - 提高复杂场景下的精度

- [ ] **实时处理优化** / **Real-time Optimization**:
  - GPU加速所有模块
  - 多线程并行处理

- [ ] **自动特征点选择** / **Automatic Feature Point Selection**:
  - 使用SIFT、SURF等特征检测
  - 自动匹配对应点

- [ ] **端到端训练** / **End-to-End Training**:
  - 训练端到端的视频理解模型
  - 融合所有子模块

- [ ] **移动端部署** / **Mobile Deployment**:
  - 使用TensorFlow Lite或PyTorch Mobile
  - 实时视频分析在移动设备

---

## 贡献指南 / Contributing

欢迎贡献！如果你希望改进这个项目 / Contributions are welcome! If you want to improve this project:

1. Fork this repository / Fork this repository
2. Create a feature branch / 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. Commit your changes / 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch / 推送到分支 (`git push origin feature/AmazingFeature`)
5. Open a Pull Request / 开启Pull Request

---

## 联系方式 / Contact

- **作者 / Author**: DR
- **邮箱 / Email**: dr1012324010@qq.com
- **GitHub**: https://github.com/Doormandd

---

## 致谢 / Acknowledgments

- **OpenCV**: Computer vision library / 计算机视觉库 - https://opencv.org/
- **NumPy**: Numerical computing / 数值计算 - https://numpy.org/
- **PyTorch**: Deep learning framework / 深度学习框架 - https://pytorch.org/
- **Ultralytics**: YOLOv8 implementation / YOLOv8实现 - https://github.com/ultralytics/ultralytics
- **VisDrone Dataset**: VPS benchmark dataset / VPS基准数据集

---

## 参考链接 / References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Lucas-Kanade Optical Flow](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_optical_flow_algorithm)
- [Image Registration Techniques](https://en.wikipedia.org/wiki/Image_registration)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [VisDrone Dataset](http://www.visdrone-dataset.com/)

---

**注意 / Note**: This project is for educational and research purposes. Performance may vary depending on video quality and complexity.

**注意 / Note**: 本项目用于教育和研究目的。性能可能因视频质量和复杂度而异。

---

## 许可证 / License

本项目采用 **MIT License** 开源 / This project is open source under the **MIT License**.

**MIT License / MIT 许可证**:
- ✅ 允许商业使用 / Commercial use allowed
- ✅ 允许修改和分发 / Modification and distribution allowed
- ✅ 需要包含许可证和版权声明 / Must include license and copyright notice
- ✅ 适用于开源和闭源项目 / Suitable for open and closed source projects