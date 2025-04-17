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

# ======== 新增NonLocal模块 ========
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super().__init__()
        self.inter_channels = inter_channels or in_channels // 2
        self.inter_channels = max(self.inter_channels, 1)
        
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=2))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=2))
        
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = f / f.size(-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        return W_y + x

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, **kwargs):
        super().__init__(in_channels, dimension=2, **kwargs)

# ======== 原final version.py代码（在GestureNet中添加NonLocal模块） ========
class GestureNet(nn.Module):
    def __init__(self, num_classes):
        super(GestureNet, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.cbam1 = CBAM(32)
        self.nonlocal1 = NONLocalBlock2D(32)  # 新增非局部模块

        # 后续层保持不变，在每层CBAM后添加NonLocal模块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)
        self.nonlocal2 = NONLocalBlock2D(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)
        self.nonlocal3 = NONLocalBlock2D(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.cbam4 = CBAM(256)
        self.nonlocal4 = NONLocalBlock2D(256)

        # 后续代码保持不变...
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = self.nonlocal1(x)  # 添加非局部模块
        x = F.max_pool2d(x, 2)
        
        # 其他层保持相同修改模式...
        return x