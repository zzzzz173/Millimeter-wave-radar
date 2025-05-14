import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import json
from tqdm import tqdm

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


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

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


# 定义数据集类
class GestureDataset(Dataset):
    def __init__(self, data_path, transform=None, is_training=True, val_split=0.3):
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

    def _extract_gesture_type(self, filepath):
        """从文件路径中提取手势类型（忽略数字）"""
        # 从路径中提取文件夹名称
        folder_name = os.path.basename(os.path.dirname(filepath))
        # 移除数字部分
        gesture_type = ''.join([c for c in folder_name if not c.isdigit()])
        return gesture_type

    def _load_data(self):
        """加载数据集"""
        try:
            # 遍历数据目录下的所有文件夹
            gesture_types = set()
            samples_by_class = {}

            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith('.npy'):
                        file_path = os.path.join(root, file)
                        gesture_type = self._extract_gesture_type(file_path)

                        if gesture_type not in gesture_types:
                            gesture_types.add(gesture_type)
                            samples_by_class[gesture_type] = []

                        samples_by_class[gesture_type].append(file_path)

            # 排序手势类型
            gesture_types = sorted(list(gesture_types))
            print(f"找到 {len(gesture_types)} 个手势类别：{gesture_types}")

            # 创建类别到索引的映射
            self.class_to_idx = {gesture: idx for idx, gesture in enumerate(gesture_types)}
            for gesture, idx in self.class_to_idx.items():
                print(f"类别 {gesture} -> 索引 {idx}")

            # 处理每个类别的数据
            for gesture_type, files in samples_by_class.items():
                # 随机打乱文件顺序
                np.random.shuffle(files)

                # 所有数据都加入训练集
                for file in files:
                    self.samples.append(file)
                    self.labels.append(self.class_to_idx[gesture_type])

            # 如果是验证模式，随机选择一部分数据作为验证集
            if not self.is_training:
                # 获取所有唯一的标签
                unique_labels = np.unique(self.labels)
                val_samples = []
                val_labels = []

                # 对每个类别随机选择验证集
                for label in unique_labels:
                    # 获取当前类别的所有样本索引
                    indices = np.where(np.array(self.labels) == label)[0]
                    # 随机选择验证集
                    num_val = max(1, int(len(indices) * self.val_split))
                    val_indices = np.random.choice(indices, num_val, replace=False)

                    # 将选中的样本移到验证集
                    val_samples.extend([self.samples[i] for i in val_indices])
                    val_labels.extend([self.labels[i] for i in val_indices])

                # 更新验证集的样本和标签
                self.samples = val_samples
                self.labels = val_labels

            # 打印每个类别的样本数量
            class_counts = {}
            for label in self.labels:
                class_name = list(self.class_to_idx.keys())[label]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            print("\n每个类别的样本数量:")
            for class_name, count in class_counts.items():
                print(f"{class_name}: {count} 个样本")

        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise

    def __len__(self):
        return len(self.samples)

    def _preprocess_data(self, radar_data):
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

            # 数据增强（仅在训练时）
            if self.is_training:
                # 1. 随机噪声（增加噪声强度）
                if np.random.random() < 0.4:  # 增加噪声概率
                    noise_level = np.random.uniform(0.05, 0.15)  # 随机噪声强度
                    noise = np.random.normal(0, noise_level, radar_data.shape)
                    radar_data = radar_data + noise

                # 2. 随机翻转（水平和垂直）
                if np.random.random() < 0.5:
                    radar_data = np.flip(radar_data, axis=0).copy()
                if np.random.random() < 0.5:
                    radar_data = np.flip(radar_data, axis=1).copy()

                # 3. 随机旋转（小角度）
                if np.random.random() < 0.3:
                    angle = np.random.uniform(-10, 10)
                    from scipy.ndimage import rotate
                    radar_data = rotate(radar_data, angle, axes=(0, 1), reshape=False)

                # 4. 随机缩放
                if np.random.random() < 0.3:
                    scale = np.random.uniform(0.9, 1.1)
                    radar_data = radar_data * scale

                # 5. 随机对比度调整
                if np.random.random() < 0.3:
                    contrast = np.random.uniform(0.8, 1.2)
                    radar_data = (radar_data - 0.5) * contrast + 0.5

                # 6. 随机遮挡（模拟部分数据缺失）
                if np.random.random() < 0.2:
                    mask_size = np.random.randint(2, 6)
                    x = np.random.randint(0, 32 - mask_size)
                    y = np.random.randint(0, 32 - mask_size)
                    radar_data[x:x + mask_size, y:y + mask_size] = 0

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
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

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
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

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


def train_gesture_model():
    """训练手势识别模型"""
    print("开始训练手势识别模型...")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据路径
    data_path = "D:/桌面/output"

    # 创建数据集
    train_dataset = GestureDataset(data_path, is_training=True)
    val_dataset = GestureDataset(data_path, is_training=False)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # 打印数据集大小和类别信息
    print(f"\n训练集大小: {len(train_dataset)} 个样本")
    print(f"验证集大小: {len(val_dataset)} 个样本")
    print("\n手势类别信息:")
    for gesture, idx in train_dataset.class_to_idx.items():
        print(f"{gesture}: {idx}")

    # 创建模型
    model = GestureNet(num_classes=len(train_dataset.class_to_idx))
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 使用AdamW优化器并增加权重衰减
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # 训练参数
    num_epochs = 100
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0

    # 记录训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 创建检查点目录
    checkpoint_dir = 'model_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.squeeze().to(device)

            # 应用Mixup数据增强
            if np.random.random() < 0.5:
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                batch_size = data.size(0)
                index = torch.randperm(batch_size).to(device)
                mixed_data = lam * data + (1 - lam) * data[index]
                data = mixed_data

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # 添加L2正则化
            l2_lambda = 0.01
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        scheduler.step()

        # 打印训练信息
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'class_to_idx': train_dataset.class_to_idx
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        # 保存最佳模型（基于验证准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'class_to_idx': train_dataset.class_to_idx
            }, 'best_model.pth')
            print(f'保存最佳模型，验证准确率: {val_acc:.2f}%')
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
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'class_to_idx': train_dataset.class_to_idx
    }
    with open('training_history.json', 'w') as f:
        json.dump(history, f)

    # 保存类别映射
    with open('class_mapping.json', 'w') as f:
        json.dump(train_dataset.class_to_idx, f)

    return model, train_dataset.class_to_idx


if __name__ == "__main__":
    print("=" * 50)
    print("手势识别训练程序")
    print("=" * 50)

    model, class_to_idx = train_gesture_model()
    print(f"手势类别: {class_to_idx}")
    print("\n程序执行完毕!") 