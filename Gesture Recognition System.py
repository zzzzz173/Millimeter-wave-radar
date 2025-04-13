import numpy as np
import torch
from fmcw import GestureNet
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def load_data_file(file_path):
    """加载数据文件"""
    try:
        data = np.load(file_path)
        # 转换为float64以提高数值精度
        data = data.astype(np.float64)
        # 对原始数据进行归一化，使用更稳定的方法
        data_mean = np.mean(data, dtype=np.float64)
        data_std = np.std(data, dtype=np.float64)
        if data_std == 0:
            data_std = 1e-10
        data = (data - data_mean) / data_std
        return data.astype(np.float32)  # 转回float32用于模型输入
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        return None


def preprocess_radar_data(radar_data):
    """预处理雷达数据"""
    try:
        print(f"原始数据形状: {radar_data.shape}")

        # 转换为float64进行计算
        radar_data = radar_data.astype(np.float64)

        # 处理2维数据
        if len(radar_data.shape) == 2:
            print("检测到2维数据，尝试转换为4维...")

            time_window = 128
            num_channels = radar_data.shape[1]
            num_windows = min(10, radar_data.shape[0] // time_window)

            if num_windows == 0:
                raise ValueError("数据太小，无法处理")

            print(f"将处理 {num_windows} 个时间窗口，每个窗口大小为 {time_window}")
            window_data = radar_data[:time_window, :]
            window_data = window_data.T.reshape(1, num_channels, time_window, 1)
            radar_data = window_data

        # 确保数据维度正确
        if len(radar_data.shape) != 4:
            raise ValueError(f"数据维度不正确，期望4维，实际为{len(radar_data.shape)}维")

        # 提取特征
        batch_size, num_channels, time_steps, _ = radar_data.shape

        # 计算range profile (在时间维度上进行FFT)
        range_profile = np.abs(np.fft.fft(radar_data, axis=2))
        range_profile = np.clip(range_profile, 0, None)

        # 计算doppler profile (在通道维度上进行FFT)
        doppler_profile = np.abs(np.fft.fft(radar_data, axis=1))
        doppler_profile = np.clip(doppler_profile, 0, None)

        # 计算方位角和仰角
        complex_sum = np.sum(radar_data, axis=2, keepdims=True)
        azimuth = np.angle(complex_sum + 1e-10)

        complex_sum_elev = np.sum(radar_data, axis=1, keepdims=True)
        elevation = np.angle(complex_sum_elev + 1e-10)

        # 广播方位角和仰角到相同的形状
        azimuth = np.broadcast_to(azimuth, (batch_size, num_channels, time_steps, 1))
        elevation = np.broadcast_to(elevation, (batch_size, num_channels, time_steps, 1))

        # 组合特征
        features = np.concatenate([
            range_profile,
            doppler_profile,
            azimuth,
            elevation
        ], axis=-1)

        # 对每个特征进行归一化，使用更稳定的方法
        for i in range(features.shape[-1]):
            feature = features[..., i]
            feature_mean = np.mean(feature, dtype=np.float64)
            feature_std = np.std(feature, dtype=np.float64)
            if feature_std == 0:
                feature_std = 1e-10
            features[..., i] = (feature - feature_mean) / feature_std
            features[..., i] = np.clip(features[..., i], -10, 10)

        print(f"最终特征形状: {features.shape}")
        return features.astype(np.float32)  # 转回float32用于模型输入

    except Exception as e:
        print(f"特征提取时出错: {str(e)}")
        raise


def predict_gesture(model, data, device):
    """预测手势并返回所有类别的概率"""
    model.eval()
    with torch.no_grad():
        try:
            # 确保数据形状正确
            if len(data.shape) == 4:
                # 如果数据是 [batch, channels, time, features]，转换为模型期望的格式
                data = data.transpose(0, 3, 1, 2)  # 变为 [batch, features, channels, time]

            # 检查数据是否包含NaN或inf
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print("警告：数据中包含NaN或inf值，将被替换为0")
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            data = torch.FloatTensor(data).to(device)
            outputs = model(data)

            # 使用更稳定的softmax实现
            outputs = outputs - outputs.max(dim=1, keepdim=True)[0]  # 数值稳定性
            exp_outputs = torch.exp(outputs)
            probabilities = exp_outputs / exp_outputs.sum(dim=1, keepdim=True)

            predicted = torch.argmax(probabilities, dim=1)

            # 确保概率和为1
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

            return predicted.item(), probabilities[0].cpu().numpy()
        except Exception as e:
            print(f"预测时出错: {str(e)}")
            raise


def visualize_predictions(probabilities, gesture_classes):
    """可视化预测结果"""
    plt.figure(figsize=(12, 6))

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建条形图
    bars = plt.bar(range(len(probabilities)), probabilities)

    # 设置颜色渐变
    colors = sns.color_palette("husl", len(probabilities))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 添加数值标签
    for i, v in enumerate(probabilities):
        plt.text(i, v, f'{v:.2%}', ha='center', va='bottom')

    # 设置图表属性
    plt.title('手势识别预测概率', fontsize=14, pad=20)
    plt.xlabel('手势类别', fontsize=12)
    plt.ylabel('预测概率', fontsize=12)

    # 设置x轴标签
    plt.xticks(range(len(gesture_classes)), gesture_classes, rotation=45, ha='right')

    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()


def get_gesture_classes(data_path):
    """从数据文件名中提取所有手势类别"""
    gesture_types = set()
    try:
        for file in os.listdir(data_path):
            if file.endswith('.npy'):
                parts = file.split('_')
                if len(parts) >= 2:
                    gesture_type = parts[1]
                    gesture_types.add(gesture_type)
        return sorted(list(gesture_types))
    except Exception as e:
        print(f"提取手势类别时出错: {str(e)}")
        return None


def main():
    # 指定数据文件夹路径
    data_dir = r"C:\Users\刘涛\Desktop\dataset"
    model_file = "best_model.pth"  # 请确保模型文件在正确的位置

    # 检查文件夹是否存在
    if not os.path.exists(data_dir):
        print(f"错误：找不到文件夹 {data_dir}")
        return

    if not os.path.exists(model_file):
        print(f"错误：找不到模型文件 {model_file}")
        return

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 定义与训练时相同的手势类别（按照混淆矩阵的顺序）
    gesture_classes = [
        "Clockwise", "Counterclockwise", "Pull", "Push",
        "SlideLeft", "SlideRight", "liftleft", "liftright",
        "sit", "stand", "turn", "walking", "waving"
    ]
    num_classes = len(gesture_classes)

    # 加载模型
    model = GestureNet(num_classes=num_classes).to(device)
    try:
        model.load_state_dict(torch.load(model_file, weights_only=True))
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return

    # 获取文件夹中的所有.npy文件
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    if not npy_files:
        print(f"错误：在 {data_dir} 中未找到任何.npy文件")
        return

    print(f"\n找到 {len(npy_files)} 个.npy文件")

    # 处理每个.npy文件
    for file_name in npy_files:
        file_path = os.path.join(data_dir, file_name)
        print(f"\n正在处理文件: {file_name}")

        # 加载数据
        data = load_data_file(file_path)
        if data is None:
            continue

        # 预处理数据
        try:
            processed_data = preprocess_radar_data(data)
            print("数据预处理成功")
        except Exception as e:
            print(f"预处理数据时出错: {str(e)}")
            continue

        # 预测手势
        try:
            gesture_idx, probabilities = predict_gesture(model, processed_data, device)

            # 打印预测结果
            print("\n预测结果:")
            print(f"文件名: {file_name}")
            print(f"识别到的手势: {gesture_classes[gesture_idx]}")
            print("\n各类别概率:")
            for i, (gesture, prob) in enumerate(zip(gesture_classes, probabilities)):
                print(f"{gesture}: {prob:.2%}")

            # 可视化预测结果
            visualize_predictions(probabilities, gesture_classes)

        except Exception as e:
            print(f"预测手势时出错: {str(e)}")
            continue

        print("-" * 50)  # 分隔线


if __name__ == "__main__":
    main()