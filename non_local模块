import torch
from torch import nn
from torch.nn import functional as F

class _NonLocalBlockND(nn.Module):
    """
    非局部神经网络模块基类（支持1D/2D/3D数据）
    
    与Transformer的主要区别：
    1. 使用卷积代替线性投影生成Q/K/V，保留空间结构
    2. 采用MaxPool下采样代替序列长度缩减
    3. 简单标量归一化代替Softmax
    4. 内置残差连接和BatchNorm
    """
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        
        # 维度验证（1D/2D/3D）
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample  # 是否使用空间下采样
        self.in_channels = in_channels
        
        # 中间通道数设置（默认为输入通道数的一半）
        self.inter_channels = inter_channels or in_channels // 2
        self.inter_channels = max(self.inter_channels, 1)  # 保证至少1个通道

        # 根据维度选择对应的卷积和池化操作
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))  # 时间维度保持，空间下采样
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))     # 空间维度减半
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)         # 长度减半
            bn = nn.BatchNorm1d

        # 值（Value）路径：g卷积 + 可选的池化
        self.g = conv_nd(in_channels, self.inter_channels, kernel_size=1)
        
        # 输出变换层（包含BatchNorm和最后的1x1卷积）
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(self.inter_channels, in_channels, kernel_size=1),
                bn(in_channels)
            )
            # 初始化BatchNorm权重为零，保证初始状态等效残差连接
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(self.inter_channels, in_channels, kernel_size=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # 查询（Query）和键（Key）路径
        self.theta = conv_nd(in_channels, self.inter_channels, kernel_size=1)  # 保持原始分辨率
        self.phi = conv_nd(in_channels, self.inter_channels, kernel_size=1)    # 可能进行下采样

        # 应用空间下采样
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        """
        前向传播流程（以3D输入为例）
        输入形状：(batch_size, in_channels, t, h, w)
        输出形状：与输入相同
        """
        batch_size = x.size(0)
        
        # 值路径处理 [B, C, ...] -> [B, N, C']
        g_x = self.g(x)
        g_x = g_x.view(batch_size, self.inter_channels, -1)  # 展平空间维度
        g_x = g_x.permute(0, 2, 1)  # [B, N, C']  N=空间位置数
        
        # 查询路径处理 [B, C, ...] -> [B, N, C']
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # [B, N, C']
        
        # 键路径处理 [B, C, ...] -> [B, C', M]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [B, C', M]

        # 注意力矩阵计算（与Transformer不同，无缩放和Softmax）
        f = torch.matmul(theta_x, phi_x)  # [B, N, M]
        f_div_C = f / f.size(-1)  # 简单标量归一化
        
        # 特征聚合
        y = torch.matmul(f_div_C, g_x)  # [B, N, C']
        y = y.permute(0, 2, 1).contiguous()  # [B, C', N]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # 恢复空间维度
        
        # 输出变换 + 残差连接
        W_y = self.W(y)
        return W_y + x  # 残差连接保证模块可插入任何位置

# 维度特化实现 ----------------------------------------------------------------
class NONLocalBlock1D(_NonLocalBlockND):
    """1D版本（适用于时序信号/文本）"""
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels, inter_channels=inter_channels,
                       dimension=1, sub_sample=sub_sample, bn_layer=bn_layer)

class NONLocalBlock2D(_NonLocalBlockND):
    """2D版本（适用于图像数据）"""
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels, inter_channels=inter_channels,
                       dimension=2, sub_sample=sub_sample, bn_layer=bn_layer)

class NONLocalBlock3D(_NonLocalBlockND):
    """3D版本（适用于视频/体数据）"""
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels, inter_channels=inter_channels,
                       dimension=3, sub_sample=sub_sample, bn_layer=bn_layer)

if __name__ == '__main__':
    # 测试用例验证不同配置下的维度变化
    for (sub_sample, bn_layer) in [(True, True), (False, False)]:
        print(f"\nTesting config: sub_sample={sub_sample}, bn_layer={bn_layer}")
        
        # 1D测试
        img = torch.zeros(2, 3, 20)
        net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(f"1D output shape: {out.shape}")  # 应保持形状（若sub_sample为False）或减半
        
        # 2D测试
        img = torch.zeros(2, 3, 20, 20)
        net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(f"2D output shape: {out.shape}")  # sub_sample时H/W减半
        
        # 3D测试
        img = torch.randn(2, 3, 8, 20, 20)
        net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(f"3D output shape: {out.shape}")  # sub_sample时H/W减半，时间维度保持
