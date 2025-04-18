import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
    
class BoundaryDetector(nn.Module):
    def __init__(self, input_channels=4, hidden_size=32):
        super().__init__()
        # 时空特征提取
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 双向时序建模
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        # 边界预测
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x形状: [batch_size, channels, time_steps]
        x = self.conv_block(x)  # -> [batch, 16, T/2]
        x = x.permute(0, 2, 1)  # -> [batch, T/2, 16]
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # -> [batch, T/2, 2*hidden_size]
        
        # 时间维度聚合
        last_state = lstm_out[:, -1, :]
        return self.fc(last_state)  # -> [batch, 2]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data: 输入数据，形状为 [num_samples, channels, time_steps]
            labels: 标签，形状为 [num_samples, 2]（二分类任务）
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 打印每个 epoch 的损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 数据生成（示例数据）
def generate_dummy_data(num_samples=1000, channels=4, time_steps=128):
    """
    生成随机时间序列数据和标签
    """
    data = np.random.rand(num_samples, channels, time_steps).astype(np.float32)
    labels = np.random.randint(0, 2, size=(num_samples, 2)).astype(np.float32)  # 二分类标签
    return data, labels

# 主函数
if __name__ == "__main__":
    # 超参数
    input_channels = 4
    hidden_size = 32
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # 数据准备
    data, labels = generate_dummy_data(num_samples=1000, channels=input_channels, time_steps=128)
    dataset = TimeSeriesDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型、损失函数和优化器
    model = BoundaryDetector(input_channels=input_channels, hidden_size=hidden_size)
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device='cpu')
