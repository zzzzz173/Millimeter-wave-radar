# Millimeter-wave-radar
毫米波雷达

## 数据采集
参数来源：<https://github.com/DI-HGR/cross_domain_gesture_dataset>
## 信号处理

## 特征提取
构建RDAI热力图，并从中提取静态和动态特征,以下是解决思路
### 数据生成与预处理：
使用**generate_moving_target_data**函数生成模拟数据，然后进行距离 - 多普勒处理和 CFAR 目标检测。
### 角度估计：
使用**music_algorithm**函数对每个检测到的目标进行方位角和仰角估计。
### 构建 RDAI 热力图：
将距离、方位角、仰角和强度信息整合到一个 3D 散点图中，其中颜色表示强度。

## 模型测试
