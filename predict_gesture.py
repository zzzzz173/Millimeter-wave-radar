import torch
import numpy as np
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# 定义神经网络模型
class GestureNet(nn.Module):
    def __init__(self, num_classes):
        super(GestureNet, self).__init__()

        # 输入通道数为4（range, velocity, azimuth, elevation）
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.cbam1 = CBAM(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.cbam4 = CBAM(256)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = F.max_pool2d(x, 2)

        # 全局平均池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


def preprocess_data(radar_data):
    """数据预处理"""
    try:
        # 确保数据是float32类型并创建副本
        radar_data = radar_data.astype(np.float32).copy()

        # 处理(N, 32, 32)格式的数据
        if len(radar_data.shape) == 3 and radar_data.shape[1:] == (32, 32):
            # 1. 计算时间维度的统计特征
            mean_time = np.mean(radar_data, axis=0)  # 时间平均
            std_time = np.std(radar_data, axis=0)  # 时间标准差
            max_time = np.max(radar_data, axis=0)  # 时间最大值
            min_time = np.min(radar_data, axis=0)  # 时间最小值

            # 2. 组合四个通道
            radar_data = np.stack([
                mean_time,  # 时间平均作为第一个通道
                std_time,  # 时间标准差作为第二个通道
                max_time,  # 时间最大值作为第三个通道
                min_time  # 时间最小值作为第四个通道
            ], axis=-1)
        else:
            raise ValueError(f"输入数据形状不正确: {radar_data.shape}, 期望形状: (N, 32, 32)")

        # 确保数据形状正确
        if radar_data.shape != (32, 32, 4):
            raise ValueError(f"预处理后数据形状错误: {radar_data.shape}, 期望形状: (32, 32, 4)")

        # 数据归一化（对每个通道分别进行）
        for i in range(4):
            channel = radar_data[..., i]
            channel_mean = np.mean(channel)
            channel_std = np.std(channel)
            if channel_std == 0:  # 处理常量通道
                channel_std = 1
            radar_data[..., i] = (channel - channel_mean) / (channel_std + 1e-8)

        return radar_data

    except Exception as e:
        print(f"数据预处理错误: {str(e)}")
        return np.zeros((32, 32, 4), dtype=np.float32)


def predict_gesture(data_path, model_path="best_model.pth", class_mapping_path="class_mapping.json"):
    """预测手势类型"""
    try:
        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if not os.path.exists(class_mapping_path):
            raise FileNotFoundError(f"类别映射文件不存在: {class_mapping_path}")

        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        # 加载类别映射
        with open(class_mapping_path, 'r') as f:
            class_to_idx = json.load(f)

        # 反转映射，获取索引到类别的映射
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = list(class_to_idx.keys())
        num_classes = len(class_names)

        print(f"支持的手势类别: {class_names}")

        # 创建模型
        model = GestureNet(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        # 加载数据
        radar_data = np.load(data_path)

        # 数据预处理
        processed_data = preprocess_data(radar_data)

        # 转换为张量并调整维度顺序
        input_tensor = torch.FloatTensor(processed_data)
        input_tensor = input_tensor.permute(2, 0, 1)  # 从(32, 32, 4)变为(4, 32, 32)
        input_tensor = input_tensor.unsqueeze(0)  # 添加batch维度
        input_tensor = input_tensor.to(device)

        # 预测
        print("执行预测...")
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)

            # 获取预测类别和概率
            prob_values, predicted_class = torch.max(probabilities, 1)
            predicted_class = predicted_class.item()
            confidence = prob_values.item() * 100

            # 获取所有类别的概率
            all_probs = probabilities.squeeze().cpu().numpy() * 100

        # 打印结果
        print("\n预测结果:")
        print(f"预测手势: {idx_to_class[predicted_class]}")
        print(f"置信度: {confidence:.2f}%")

        print("\n所有类别的概率:")
        for i, prob in enumerate(all_probs):
            print(f"{idx_to_class[i]}: {prob:.2f}%")

        # 可视化预测结果
        plt.figure(figsize=(12, 6))

        # 绘制预处理后的数据
        plt.subplot(1, 2, 1)
        plt.imshow(processed_data[:, :, 0], cmap='jet')
        plt.title("处理后的雷达数据 (通道1)")
        plt.colorbar()

        # 绘制预测结果
        plt.subplot(1, 2, 2)
        plt.bar(range(len(class_names)), all_probs)
        plt.title('预测概率')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right', fontsize=9)
        plt.ylabel('概率 (%)')

        # 在柱状图上添加概率值
        for i, prob in enumerate(all_probs):
            if prob > 1.0:  # 只为较大的概率值添加标签
                plt.text(i, prob + 2, f"{prob:.1f}%", ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
        plt.close()

        return idx_to_class[predicted_class], all_probs

    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # 设置文件路径
    data_path = r"D:\桌面\大创111\MCD-Gesture-DRAI\y_Pull_e4_u1_p4_s3.npy"
    model_path = "best_model.pth"
    class_mapping_path = "class_mapping.json"

    # 执行预测
    predicted_class, probabilities = predict_gesture(data_path, model_path, class_mapping_path)

    if predicted_class is not None:
        print("\n预测完成！结果已保存到 prediction_result.png")