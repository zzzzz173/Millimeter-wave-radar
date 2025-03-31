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

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 雷达参数设置
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


def cfar_2d(matrix, guard_win=5, train_win=10, false_alarm=1e-6):
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

            # 使用严格的阈值
            threshold = train_mean + 8 * train_std
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
        """从文件名中提取手势类型"""
        parts = filename.split('_')
        if len(parts) >= 2:
            return parts[1]
        return None

    def _load_data(self):
        """加载数据集"""
        try:
            all_files = [f for f in os.listdir(self.data_path) if f.endswith('.npy')]

            if not all_files:
                raise ValueError(f"在 {self.data_path} 中未找到任何.npy文件")

            gesture_types = set()
            for file in all_files:
                gesture_type = self._extract_gesture_type(file)
                if gesture_type:
                    gesture_types.add(gesture_type)

            gesture_types = sorted(list(gesture_types))
            print(f"找到 {len(gesture_types)} 个手势类别：{gesture_types}")

            self.class_to_idx = {gesture: idx for idx, gesture in enumerate(gesture_types)}
            for gesture, idx in self.class_to_idx.items():
                print(f"类别 {gesture} -> 索引 {idx}")

            samples_by_class = {gesture: [] for gesture in gesture_types}
            for file in all_files:
                gesture_type = self._extract_gesture_type(file)
                if gesture_type in self.class_to_idx:
                    samples_by_class[gesture_type].append(file)

            for gesture_type, files in samples_by_class.items():
                num_samples = len(files)
                num_val = int(num_samples * self.val_split)

                if self.is_training:
                    target_files = files[:-num_val]
                else:
                    target_files = files[-num_val:]

                for file in target_files:
                    self.samples.append(os.path.join(self.data_path, file))
                    self.labels.append(self.class_to_idx[gesture_type])

                print(f"类别 {gesture_type}: 总共 {num_samples} 个样本，"
                      f"使用 {len(target_files)} 个样本用于{'训练' if self.is_training else '验证'}")

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


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练过程中的损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.title('训练过程中的准确率变化')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()


def plot_confusion_matrix(model, val_loader, device, class_names=None):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

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

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else None,
                yticklabels=class_names if class_names else None)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_targets, all_preds,
                                target_names=class_names if class_names else None))


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 数据路径
    data_path = "D:/桌面/大创111/MCD-Gesture-DRAI"

    # 创建数据集
    train_dataset = GestureDataset(data_path, is_training=True)
    val_dataset = GestureDataset(data_path, is_training=False)

    # 创建数据加载器（减小batch_size和num_workers）
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # 减小batch size
        shuffle=True,
        num_workers=2,  # 减少worker数量
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,  # 减小batch size
        shuffle=False,
        num_workers=2,  # 减少worker数量
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
    num_epochs = 150
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

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

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
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

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


if __name__ == "__main__":
    main() 