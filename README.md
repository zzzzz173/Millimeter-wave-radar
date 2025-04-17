# 毫米波雷达手势识别系统

本项目基于毫米波雷达信号处理与深度学习，集成了经典信号处理算法（FFT、CFAR、MUSIC）与多种神经网络模型（含CBAM注意力机制、NonLocal模块、轻量化网络等），实现了多目标检测、角度估计与手势识别，并提供丰富的可视化分析。

## 目录结构
- `final version.py`：主流程代码，包含信号处理、特征提取、神经网络模型（含CBAM）等。
- `final_version_non_local.py`：集成NonLocal模块的改进版神经网络。
- `lightweighting.py`：轻量化网络实现与相关信号处理。
- `image/`：可视化结果图片（3D点云、热力图、MUSIC谱等）。
- `README.md`：项目说明文档。
- 其他：`adc_data.bin`（原始数据）、`bin_to_npy_converter.m`（数据转换脚本）、`non_local模块`、`可视化模块pro`（模块文件夹）。

## 安装依赖
建议使用Python 3.8+，主要依赖如下：
```bash
pip install numpy torch torchvision matplotlib scikit-learn tqdm seaborn scipy
```

## 快速开始
1. 数据准备：
   - 可使用`adc_data.bin`或自定义生成数据。
   - 支持直接调用`generate_moving_target_data`函数生成模拟目标。
2. 运行主流程：
   ```bash
   python final\ version.py
   ```
3. 查看可视化结果：
   - 结果图片保存在`image/`目录。
   - 支持3D点云、距离-多普勒热力图、MUSIC谱、极坐标等多种可视化。

## 主要功能模块
### 1. 信号处理
#### 1.1 距离-多普勒处理（FFT）
- 基于快速傅里叶变换（FFT）算法，提取目标的距离和速度信息，实现高效的距离-速度二维谱分析。
- 支持多目标分离，提升运动目标检测能力。
#### 1.2 CFAR自适应检测
- 采用恒虚警率（CFAR）算法，动态调整检测阈值，显著提升在复杂背景下的目标检测鲁棒性。
- 支持多种CFAR变体（如CA-CFAR、OS-CFAR等），适应不同场景需求。
#### 1.3 MUSIC高分辨率角度估计
- 利用MUSIC（Multiple Signal Classification）算法，实现对目标方位角和俯仰角的高精度估计。
- 具备多目标分辨能力，适用于密集目标环境。

### 2. 特征提取与建模
#### 2.1 RDAI多维特征融合
- 融合距离、速度、角度和信号强度等多维信息，生成丰富的热力图特征。
- 支持自定义特征组合，提升模型泛化能力。
#### 2.2 深度神经网络模型
- 集成多种神经网络结构，包括ResNet、CBAM注意力机制、NonLocal模块等。
- CBAM模块增强网络对关键区域的关注，NonLocal模块提升全局特征建模能力。
- 提供轻量化网络实现，适用于资源受限设备。
#### 2.3 手势识别与多目标分类
- 支持多类别手势识别，兼容多目标同时检测与分类。
- 可扩展至更多手势类别和复杂动作。

### 3. 可视化分析
#### 3.1 距离-多普勒热力图
- 直观展示目标在距离和速度维度的分布，便于运动特征分析。
#### 3.2 MUSIC谱三维曲面
- 展示角度估计结果，验证MUSIC算法性能。
#### 3.3 3D点云可视化
- 空间分布与速度信息一体化展示，支持多目标动态跟踪。
#### 3.4 极坐标视图
- 以极坐标方式展示目标方位与距离关系，辅助空间感知。

#### 示例图片
![距离-多普勒视图](image/距离-多普勒视图.png)
![MUSIC谱分析](image/music谱分析.png)
![3D点云可视化](image/3D点云可视化.png)
![极坐标视图](image/极坐标视图.png)

## 代码示例
```python
from final_version_non_local import GestureNet
import torch
model = GestureNet(num_classes=6)
input_tensor = torch.randn(1, 4, 64, 64)  # 假设输入为4通道64x64
output = model(input_tensor)
```

## 技术亮点与创新
- 多模态数据整合，提升检测与识别精度。
- 动态交互式可视化，支持多场景分析。
- 军事与民用场景兼容，适用性广泛。

## 致谢
部分参数与数据参考自[DI-HGR/cross_domain_gesture_dataset](https://github.com/DI-HGR/cross_domain_gesture_dataset)。

## License
本项目仅供学术研究与交流，禁止商业用途。
