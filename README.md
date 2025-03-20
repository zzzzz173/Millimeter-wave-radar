# Millimeter-wave-radar
毫米波雷达

## 数据采集
参数来源：<https://github.com/DI-HGR/cross_domain_gesture_dataset>
我们自定义生成了4种速度的物体的数据，两种为较慢速度，两种为中等速度
## 信号处理

## 特征提取
构建RDAI热力图，并从中提取静态和动态特征,以下是解决思路
### 数据生成与预处理：
使用`generate_moving_target_data`函数生成模拟数据，然后进行距离 - 多普勒处理和 CFAR 目标检测。
### 角度估计：
使用`music_algorithm`函数对每个检测到的目标进行方位角和仰角估计。
### 构建 RDAI 热力图：
将距离、方位角、仰角和强度信息整合到一个 3D 散点图中，其中颜色表示强度。
我可以帮你开发更多类型的可视化来丰富这个毫米波雷达系统的结果展示。以下是我可以添加的几种可视化方式：

## 可视化模块增强

我们添加了七种新的可视化方法，大大增强了对毫米波雷达数据的分析能力。下面解释每种新增的可视化的作用：

### 1. 距离-多普勒图 (Range-Doppler Map)

这个可视化以热力图的形式展示了距离-多普勒域中的信号强度分布，并用红色×标记标出经过CFAR算法检测的目标。这是雷达信号处理中最基础也是最重要的可视化之一，可以直观地看出目标在距离和速度维度上的分布情况。

### 2. MUSIC角度谱分析

这个可视化包含两部分：
- 2D热力图展示了方位角-仰角域中的MUSIC谱能量分布
- 3D表面图提供了更直观的角度谱能量峰值展示

这个可视化能帮助我们理解MUSIC算法的角度估计性能，观察角度域的分辨率以及可能存在的虚假目标。

### 3. 距离剖面图 (Range Profile)

展示了沿距离维度的能量分布，并用红色虚线标记出真实目标的距离。这个剖面图可以帮助我们了解雷达在距离维度上的分辨能力和检测效果。

### 4. 多普勒剖面图 (Doppler Profile)

在目标所在的距离bin上，展示了沿多普勒维度的能量分布，并标记出真实目标的速度。这个剖面图可以帮助我们了解雷达在速度维度上的分辨能力。

### 5. 极坐标检测图 (Polar Detection View)

以极坐标的形式（类似雷达扫描显示）展示目标的距离和方位角，提供了一个俯视图视角。这对于理解目标的空间分布非常直观，类似于传统雷达显示屏的效果。

### 6. 检测置信度分布图 (Detection Confidence)

分析每个检测点与真实目标的距离，从而得出检测置信度，并以散点图的形式展示。这个可视化有助于评估检测结果的质量和可靠性。

### 7. 一键生成所有可视化的集成函数

提供了`plot_all_visualizations()`函数，可以一次性生成所有的可视化图形，并将图像保存为PNG文件，便于后续分析和报告制作。

### 对现有可视化的增强

对原有的两个可视化函数也进行了增强：
- 添加了图像保存功能
- 改进了图像布局和标注
- 统一了颜色方案和点大小


## 模型测试
