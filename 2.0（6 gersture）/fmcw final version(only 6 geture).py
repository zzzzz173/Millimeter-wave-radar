import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import json
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, maximum_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ========================== 雷达参数设置 ==========================
fc = 77e9  # 雷达载频 77 GHz
c = 3e8  # 光速
lambda_ = c / fc  # 波长
d = lambda_ / 2  # 阵元间距 (均匀线阵)
num_tx = 2  # 发射天线数
num_rx = 4  # 接收天线数
num_chirps = 128  # 每帧啁啾数
num_samples = 128  # 每啁啾采样点数
S = 99.987e12  # 频率斜率
theta_scan = np.linspace(-90, 90, 181)  # 方位角搜索范围
phi_scan = np.linspace(-90, 90, 181)  # 仰角搜索范围
sample_rate = 4e6  # 采样率
T_chirp = 40e-6
T_idle = 380e-6
chirp_time = T_chirp + T_idle  # 啁啾周期
frame_time = chirp_time * num_chirps  # 帧周期


# ========================== 雷达信号处理函数 ==========================
def generate_moving_target_data(num_tx, num_rx, num_chirps, num_samples, fc, c, targets):
    """生成移动目标的雷达数据"""
    lambda_ = c / fc
    t = np.arange(num_samples) / sample_rate  # 时间轴
    sample_data = np.zeros((num_tx, num_rx, num_chirps, num_samples), dtype=complex)

    for velocity, distance, azimuth, elevation in targets:
        f_doppler = 2 * velocity / lambda_
        f_range = 2 * distance * S / c

        for tx in range(num_tx):
            for rx in range(num_rx):
                for chirp in range(num_chirps):
                    phase_shift = 2 * np.pi * f_doppler * chirp * chirp_time
                    azimuth_shift = 2 * np.pi * d * np.sin(np.deg2rad(azimuth)) * tx / lambda_
                    elevation_shift = 2 * np.pi * d * np.sin(np.deg2rad(elevation)) * rx / lambda_
                    sample_data[tx, rx, chirp, :] += np.exp(
                        1j * (2 * np.pi * f_range * t + phase_shift + azimuth_shift + elevation_shift))

    return sample_data


def cfar_2d(matrix, guard_win=1, train_win=2, false_alarm=1e-6):
    """2D CFAR检测"""
    num_range, num_doppler = matrix.shape
    mask = np.zeros((num_range, num_doppler), dtype=bool)

    for r in range(num_range):
        for d in range(num_doppler):
            # 定义训练单元和保护单元
            r_start = max(0, r - train_win - guard_win)
            r_end = min(num_range, r + train_win + guard_win + 1)
            d_start = max(0, d - train_win - guard_win)
            d_end = min(num_doppler, d + train_win + guard_win + 1)

            # 定义保护区域
            guard_r_start = max(0, r - guard_win)
            guard_r_end = min(num_range, r + guard_win + 1)
            guard_d_start = max(0, d - guard_win)
            guard_d_end = min(num_doppler, d + guard_win + 1)

            # 提取训练单元
            training_cells = matrix[r_start:r_end, d_start:d_end].copy()
            guard_cells = matrix[guard_r_start:guard_r_end, guard_d_start:guard_d_end]
            training_cells[r - r_start:r - r_start + guard_cells.shape[0],
            d - d_start:d - d_start + guard_cells.shape[1]] = 0

            # 计算训练单元均值和标准差
            train_mean = np.mean(training_cells[training_cells != 0])
            train_std = np.std(training_cells[training_cells != 0])

            # 使用更宽松的阈值
            alpha = 1.0  # 降低阈值系数
            threshold = train_mean + alpha * train_std
            if matrix[r, d] > threshold:
                mask[r, d] = True

    return mask


def music_algorithm(snapshots, theta_scan, phi_scan, d, lambda_, num_tx, num_rx, K=1):
    """MUSIC角度估计"""
    M, N = snapshots.shape
    R = (snapshots @ snapshots.conj().T) / N
    eigen_values, eigen_vectors = np.linalg.eigh(R)
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:, sorted_indices]
    Un = eigen_vectors[:, K:]

    music_spectrum = np.zeros((len(theta_scan), len(phi_scan)))
    for i, theta in enumerate(theta_scan):
        for j, phi in enumerate(phi_scan):
            a_tx = np.exp(-1j * 2 * np.pi * d * np.sin(np.deg2rad(theta)) * np.arange(num_tx) / lambda_)
            a_rx = np.exp(-1j * 2 * np.pi * d * np.sin(np.deg2rad(phi)) * np.arange(num_rx) / lambda_)
            a = np.kron(a_tx, a_rx).reshape(-1, 1)
            music_spectrum[i, j] = 1 / np.abs(a.conj().T @ Un @ Un.conj().T @ a).squeeze()

    return music_spectrum / np.max(music_spectrum)  # 归一化


# ========================== 神经网络模型 ==========================
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
        # 输入已经是 [batch_size, 4, 32, 32]，不需要调整维度顺序
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


# 定义数据集类
class GestureDataset(Dataset):
    def __init__(self, data_path, transform=None, is_training=True, val_split=0.2):
        self.data_path = data_path
        self.transform = transform
        self.is_training = is_training
        self.val_split = val_split
        self.samples = []
        self.labels = []
        self.class_to_idx = {}

        # 检查数据路径是否存在
        if not os.path.exists(data_path):
            raise ValueError(f"数据路径不存在: {data_path}")

        print(f"正在加载数据集，路径: {data_path}")

        # 加载数据
        self._load_data()

        if len(self.samples) == 0:
            raise ValueError(f"未找到任何数据样本，请检查路径: {data_path}")

        print(f"成功加载 {len(self.samples)} 个样本")

    def _extract_gesture_type(self, filename):
        """从文件名中提取手势类型和采样类型"""
        parts = filename.split('_')
        if len(parts) >= 2:
            # 检查是否为负采样
            if parts[0] == 'n':
                return parts[1], 'negative'
            # 检查是否为正采样
            elif parts[0] == 'y':
                return parts[1], 'positive'
        return None, None

    def _load_data(self):
        """加载数据集"""
        try:
            all_files = [f for f in os.listdir(self.data_path) if f.endswith('.npy')]

            if not all_files:
                raise ValueError(f"在 {self.data_path} 中未找到任何.npy文件")

            # 只收集正采样的手势类型
            gesture_types = set()
            negative_samples = []

            # 首先遍历所有文件，收集手势类型和负采样
            for file in all_files:
                gesture_type, sample_type = self._extract_gesture_type(file)
                if gesture_type:
                    if sample_type == 'positive':
                        gesture_types.add(gesture_type)
                    elif sample_type == 'negative':
                        negative_samples.append(file)

            # 排序手势类型
            gesture_types = sorted(list(gesture_types))
            print(f"找到 {len(gesture_types)} 个手势类别：{gesture_types}")
            print(f"找到 {len(negative_samples)} 个负采样样本")

            # 创建类别到索引的映射
            self.class_to_idx = {gesture: idx for idx, gesture in enumerate(gesture_types)}
            for gesture, idx in self.class_to_idx.items():
                print(f"类别 {gesture} -> 索引 {idx}")

            # 按类别组织正采样文件
            samples_by_class = {gesture: [] for gesture in gesture_types}

            for file in all_files:
                gesture_type, sample_type = self._extract_gesture_type(file)
                if gesture_type in self.class_to_idx and sample_type == 'positive':
                    samples_by_class[gesture_type].append(file)

            # 处理每个类别的数据
            for gesture_type, files in samples_by_class.items():
                if self.is_training:
                    # 训练时使用所有正采样数据
                    target_files = files
                    print(f"类别 {gesture_type}: 使用 {len(target_files)} 个正采样样本进行训练")
                else:
                    # 验证时使用20%的数据
                    num_val = max(1, int(len(files) * self.val_split))
                    target_files = files[-num_val:]
                    print(f"类别 {gesture_type}: 使用 {len(target_files)} 个正采样样本进行验证")

                for file in target_files:
                    self.samples.append(os.path.join(self.data_path, file))
                    self.labels.append(self.class_to_idx[gesture_type])

            # 在训练时添加负采样数据
            if self.is_training and negative_samples:
                # 随机选择与正采样总数相当的负采样
                np.random.shuffle(negative_samples)
                num_negative = min(len(negative_samples), len(self.samples))
                selected_negative = negative_samples[:num_negative]

                print(f"添加 {len(selected_negative)} 个负采样作为干扰样本")

                for file in selected_negative:
                    self.samples.append(os.path.join(self.data_path, file))
                    # 负采样样本的标签设为-1，表示干扰样本
                    self.labels.append(-1)

        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise

    def __len__(self):
        return len(self.samples)

    def _preprocess_data(self, radar_data):
        """数据预处理"""
        try:
            print(f"原始数据形状: {radar_data.shape}")

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

            # 数据增强（仅在训练时）
            if self.is_training:
                # 随机噪声
                if np.random.random() < 0.3:
                    noise = np.random.normal(0, 0.1, radar_data.shape)
                    radar_data = radar_data + noise

                # 随机翻转（使用copy确保连续内存）
                if np.random.random() < 0.5:
                    radar_data = np.flip(radar_data, axis=0).copy()

            return radar_data

        except Exception as e:
            print(f"数据预处理错误: {str(e)}")
            return np.zeros((32, 32, 4), dtype=np.float32)

    def __getitem__(self, idx):
        try:
            # 加载雷达数据
            radar_data = np.load(self.samples[idx])
            label = self.labels[idx]

            # 数据预处理
            radar_data = self._preprocess_data(radar_data)

            # 确保数据是连续的
            radar_data = np.ascontiguousarray(radar_data)

            # 检查数据形状
            if radar_data.shape != (32, 32, 4):
                print(f"警告：样本 {self.samples[idx]} 的形状不正确: {radar_data.shape}")
                radar_data = np.zeros((32, 32, 4), dtype=np.float32)

            # 转换为张量并调整维度顺序
            radar_data = torch.FloatTensor(radar_data)
            radar_data = radar_data.permute(2, 0, 1)  # 从(32, 32, 4)变为(4, 32, 32)

            # 应用变换
            if self.transform:
                radar_data = self.transform(radar_data)

            # 转换标签
            label = torch.LongTensor([label])

            return radar_data, label

        except Exception as e:
            print(f"处理样本 {self.samples[idx]} 时出错: {str(e)}")
            default_data = torch.zeros((4, 32, 32), dtype=torch.float32)
            default_label = torch.LongTensor([0])
            return default_data, default_label


# ========================== 可视化函数 ==========================
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.figure(figsize=(15, 6))
    epochs = range(1, len(train_losses) + 1)

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    plt.title('训练过程中的损失变化', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='训练准确率', linewidth=2)
    plt.plot(epochs, val_accs, 'r-', label='验证准确率', linewidth=2)
    plt.title('训练过程中的准确率变化', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('准确率 (%)', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(model, val_loader, device, class_names=None):
    """绘制混淆矩阵"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.squeeze().cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)

    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 绘制混淆矩阵
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names if class_names else None,
                yticklabels=class_names if class_names else None)
    plt.title('混淆矩阵 (%)', fontsize=14)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)

    # 调整标签文字大小和旋转角度
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_targets, all_preds,
                                target_names=class_names if class_names else None,
                                digits=4))


def plot_range_doppler_map(doppler_fft, velocities, ranges, velocity_scale, num_chirps, range_scale, num_samples):
    """ 绘制距离-多普勒图 """
    power_db = 10 * np.log10(np.abs(doppler_fft[:, :, 0, 0]) ** 2 + 1e-10)

    plt.figure(figsize=(12, 8))
    plt.imshow(power_db, aspect='auto', cmap='jet',
               extent=[-velocity_scale * num_chirps / 2, velocity_scale * num_chirps / 2,
                       range_scale * num_samples, 0])
    plt.scatter(velocities, ranges, c='red', s=20, marker='x', label='检测目标')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Range (m)')
    plt.colorbar(label='Power (dB)')
    plt.title('距离-多普勒图')
    plt.legend()
    plt.tight_layout()
    plt.savefig('range_doppler_map.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_music_spectrum(range_idx, doppler_idx, doppler_fft, theta_scan, phi_scan, d, lambda_, num_tx, num_rx,
                        range_scale, velocity_scale, targets):
    """ 绘制指定单元的MUSIC谱 """
    snapshots = doppler_fft[range_idx, doppler_idx, :, :].reshape(num_tx * num_rx, -1)
    spectrum = music_algorithm(snapshots, theta_scan, phi_scan, d, lambda_, num_tx, num_rx)

    # 3D MUSIC谱
    theta_grid, phi_grid = np.meshgrid(theta_scan, phi_scan)

    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(theta_grid, phi_grid, spectrum.T, cmap='jet')
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Elevation (deg)')
    ax.set_zlabel('Power')

    # 2D投影
    plt.subplot(132)
    plt.contourf(theta_scan, phi_scan, spectrum.T, 20, cmap='jet')
    for t in targets:
        plt.plot(t[2], t[3], 'wx', markersize=10)
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('Elevation (deg)')

    plt.subplot(133)
    plt.plot(theta_scan, np.max(spectrum, axis=1), label='Azimuth')
    plt.plot(phi_scan, np.max(spectrum, axis=0), label='Elevation')
    plt.legend()
    plt.xlabel('Angle (deg)')
    plt.ylabel('Normalized Power')
    plt.suptitle(f'MUSIC谱分析 (距离={range_idx * range_scale:.1f}m, 速度={doppler_idx * velocity_scale:.1f}m/s)')
    plt.tight_layout()
    plt.savefig(f'music_spectrum_{range_idx}_{doppler_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_3d_pointcloud(ranges, azimuths, elevations, velocities, targets):
    """ 3D点云可视化 """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(ranges, azimuths, elevations, c=velocities,
                    cmap='viridis', s=50, alpha=0.7)

    ax.set_xlabel('距离 (m)')
    ax.set_ylabel('方位角 (deg)')
    ax.set_zlabel('仰角 (deg)')
    plt.colorbar(sc, label='速度 (m/s)')
    plt.title('雷达点云')
    plt.tight_layout()
    plt.savefig('3d_pointcloud.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_polar_view(azimuths, ranges, intensities, targets):
    """ 极坐标显示 """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    azimuth_rad = np.deg2rad(azimuths)

    sc = ax.scatter(azimuth_rad, ranges, c=intensities,
                    cmap='viridis', s=50, alpha=0.7)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rmax(100)
    plt.colorbar(sc, label='信号强度')
    plt.title('极坐标视图（距离-方位角）')
    plt.tight_layout()
    plt.savefig('polar_view.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_visualizations(doppler_fft, velocities, ranges, velocity_scale, num_chirps, range_scale, num_samples,
                            range_bins, doppler_bins, theta_scan, phi_scan, d, lambda_, num_tx, num_rx,
                            azimuths, elevations, intensities, targets):
    """ 执行所有可视化 """
    plt.close('all')
    plot_range_doppler_map(doppler_fft, velocities, ranges, velocity_scale, num_chirps, range_scale, num_samples)
    plot_3d_pointcloud(ranges, azimuths, elevations, velocities, targets)
    plot_polar_view(azimuths, ranges, intensities, targets)

    # 显示前3个检测目标的MUSIC谱
    for i in range(min(3, len(ranges))):
        plot_music_spectrum(range_bins[i], doppler_bins[i], doppler_fft, theta_scan, phi_scan, d, lambda_,
                            num_tx, num_rx, range_scale, velocity_scale, targets)


# ========================== 雷达信号处理主函数 ==========================
def process_radar_data():
    """处理雷达数据并可视化"""
    print("开始处理雷达数据...")

    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 使用训练数据集
    data_path = "D:/桌面/大创111/MCD-Gesture-DRAI"  # 使用原始数据路径
    if not os.path.exists(data_path):
        print(f"错误：数据路径不存在: {data_path}")
        data_path = input("请输入正确的数据路径：")
        if not os.path.exists(data_path):
            print(f"错误：输入的路径 {data_path} 仍然不存在")
            return

    # 获取数据文件列表
    data_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    if not data_files:
        print(f"错误：在 {data_path} 中未找到任何.npy文件")
        print("请确保数据文件(.npy)在指定目录下")
        return

    print(f"找到 {len(data_files)} 个数据文件")

    # 处理多个数据文件
    num_files_to_process = min(10, len(data_files))  # 先处理前10个文件
    total_targets = 0
    processed_files = 0

    try:
        for file_idx in range(num_files_to_process):
            file_name = data_files[file_idx]
            print(f"\n处理文件 {file_idx + 1}/{num_files_to_process}: {file_name}")

            try:
                # 加载雷达数据
                data = np.load(os.path.join(data_path, file_name))
                print(f"数据形状: {data.shape}")

                # 数据预处理和重塑
                if len(data.shape) == 3 and data.shape[1:] == (32, 32):
                    # 使用更多统计特征和权重
                    max_data = np.max(data, axis=0)
                    mean_data = np.mean(data, axis=0)
                    std_data = np.std(data, axis=0)
                    min_data = np.min(data, axis=0)
                    median_data = np.median(data, axis=0)

                    # 使用加权组合
                    data_reshaped = (3 * max_data + 2 * mean_data + 2 * std_data + median_data + min_data) / 9
                else:
                    print(f"警告：文件 {file_name} 的数据格式不正确，跳过处理")
                    continue

                # 应用自适应高斯滤波
                sigma = np.std(data_reshaped) * 0.2  # 减小sigma以保留更多细节
                data_reshaped = gaussian_filter(data_reshaped, sigma=sigma)

                # 增强对比度
                p1, p99 = np.percentile(data_reshaped, (1, 99))  # 使用更极端的百分位数
                data_reshaped = np.clip(data_reshaped, p1, p99)
                data_reshaped = (data_reshaped - p1) / (p99 - p1)

                # CFAR目标检测（使用更宽松的参数）
                power = data_reshaped
                target_mask = cfar_2d(power, guard_win=1, train_win=2)  # 减小训练窗口
                intensity_threshold = 0.005  # 进一步降低阈值
                target_mask = target_mask & (power > intensity_threshold)

                # 使用更小的局部最大值窗口
                local_max = maximum_filter(power, size=1)
                target_mask = target_mask & (power >= local_max)  # 改为大于等于

                # 提取目标点
                target_indices = np.where(target_mask)
                num_targets = len(target_indices[0])
                total_targets += num_targets
                processed_files += 1

                print(f"检测到 {num_targets} 个目标点")

                # 如果是第一个文件，生成可视化结果
                if file_idx == 0:
                    ranges = target_indices[0]
                    velocities = target_indices[1]
                    intensities = power[target_mask]
                    azimuths = np.random.uniform(-30, 30, size=len(ranges))
                    elevations = np.random.uniform(-20, 20, size=len(ranges))

                    # 创建虚拟目标用于可视化
                    targets = [[0, r, a, e] for r, a, e in zip(ranges[:4], azimuths[:4], elevations[:4])]
                    if len(targets) < 4:
                        while len(targets) < 4:
                            targets.append([0, 45, 0, 0])

                    # 绘制数据处理过程的可视化
                    plt.figure(figsize=(15, 5))

                    # 原始数据
                    plt.subplot(131)
                    plt.imshow(np.max(data, axis=0), aspect='auto', cmap='jet')
                    plt.colorbar(label='原始信号强度')
                    plt.title('原始雷达信号')
                    plt.xlabel('多普勒频率')
                    plt.ylabel('距离')

                    # 处理后的数据
                    plt.subplot(132)
                    plt.imshow(data_reshaped, aspect='auto', cmap='jet')
                    plt.colorbar(label='处理后信号强度')
                    plt.title('处理后的信号')
                    plt.xlabel('多普勒频率')
                    plt.ylabel('距离')

                    # 检测结果
                    plt.subplot(133)
                    plt.imshow(power * target_mask, aspect='auto', cmap='jet')
                    plt.scatter(target_indices[1], target_indices[0], c='red', s=50, marker='x', label='检测目标')
                    plt.colorbar(label='检测到的目标')
                    plt.title('目标检测结果')
                    plt.xlabel('多普勒频率')
                    plt.ylabel('距离')
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig('radar_processing_steps.png', dpi=300, bbox_inches='tight')
                    plt.close()

                    # 3D点云可视化
                    plot_3d_pointcloud(ranges, azimuths, elevations, velocities, targets)

                    # 极坐标显示
                    plot_polar_view(azimuths, ranges, intensities, targets)

            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {str(e)}")
                continue

        # 显示统计信息
        print("\n处理统计信息:")
        print(f"总共处理文件数: {processed_files}")
        print(f"总检测目标点数: {total_targets}")
        if processed_files > 0:
            print(f"平均每个文件目标点数: {total_targets / processed_files:.2f}")

        print("\n雷达数据处理完成，可视化结果已保存")

        return None, None, None, ranges, velocities, azimuths, elevations, intensities

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        return None, None, None, None, None, None, None, None


# ========================== 神经网络训练主函数 ==========================
def train_gesture_model():
    """训练手势识别模型"""
    print("开始训练手势识别模型...")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据路径
    data_path = "D:/桌面/大创111/MCD-Gesture-DRAI"

    # 创建数据集
    train_dataset = GestureDataset(data_path, is_training=True)
    val_dataset = GestureDataset(data_path, is_training=False)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    # 创建模型
    model = GestureNet(num_classes=len(train_dataset.class_to_idx))
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=1e-4
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # 训练参数
    num_epochs = 10
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    # 记录训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            data, target = data.to(device), target.squeeze().to(device)

            # 分离正采样和负采样
            positive_mask = target != -1
            negative_mask = ~positive_mask

            optimizer.zero_grad()
            output = model(data)

            # 计算正采样的损失
            if positive_mask.any():
                positive_loss = criterion(output[positive_mask], target[positive_mask])
            else:
                positive_loss = torch.tensor(0.0, device=device)

            # 计算负采样的损失（使用所有类别的负对数似然）
            if negative_mask.any():
                # 对负采样，我们希望所有类别的概率都很低
                negative_output = output[negative_mask]
                # 使用 softmax 计算概率
                probs = F.softmax(negative_output, dim=1)
                # 计算负采样的损失（希望所有类别的概率都接近0）
                negative_loss = -torch.log(1 - probs + 1e-8).mean()
            else:
                negative_loss = torch.tensor(0.0, device=device)

            # 总损失
            loss = positive_loss + 0.5 * negative_loss  # 可以调整负采样的权重

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 只计算正采样的准确率
            if positive_mask.any():
                _, predicted = output[positive_mask].max(1)
                train_total += positive_mask.sum().item()
                train_correct += predicted.eq(target[positive_mask]).sum().item()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                data, target = data.to(device), target.squeeze().to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        # 计算平均损失和准确率
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0

        # 记录训练历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 更新学习率
        scheduler.step(val_loss)

        # 打印训练信息
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    print("训练完成！")

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # 绘制混淆矩阵
    class_names = list(train_dataset.class_to_idx.keys())
    plot_confusion_matrix(model, val_loader, device, class_names)

    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    with open('training_history.json', 'w') as f:
        json.dump(history, f)

    return model, train_dataset.class_to_idx


# ========================== 主程序 ==========================
if __name__ == "__main__":
    print("=" * 50)
    print("雷达手势识别系统")
    print("=" * 50)

    # 选择运行模式
    mode = input("请选择运行模式 (1: 雷达信号处理, 2: 手势识别训练, 3: 全部): ")

    if mode == "1" or mode == "3":
        print("\n开始雷达信号处理...")
        process_radar_data()

    if mode == "2" or mode == "3":
        print("\n开始手势识别训练...")
        model, class_to_idx = train_gesture_model()
        print(f"手势类别: {class_to_idx}")

    print("\n程序执行完毕!")