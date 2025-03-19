import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========================== 参数设置 ==========================
fc = 77e9          # 雷达载频 77 GHz
c = 3e8            # 光速
lambda_ = c / fc   # 波长
d = lambda_ / 2    # 阵元间距 (均匀线阵)
num_tx = 2         # 发射天线数
num_rx = 4         # 接收天线数
num_chirps = 128   # 每帧啁啾数
num_samples = 128  # 每啁啾采样点数
S = 99.987e12       # 频率斜率
theta_scan = np.linspace(-90, 90, 181)  # 方位角搜索范围
phi_scan = np.linspace(-90, 90, 181)    # 仰角搜索范围
sample_rate = 4e6  # 采样率
T_chirp = 40e-6    # chirp持续时间
T_idle = 340e-6    # chirp间隔时间

# ========================== 生成模拟数据 ==========================

def generate_moving_target_data(num_tx, num_rx, num_chirps, num_samples, fc, c, velocity, distance, azimuth, elevation):
    """
    生成模拟在一定位置以一定速率运动的物体的雷达数据
    """
    lambda_ = c / fc
    t = np.arange(num_samples) * T_chirp / num_samples  # 时间轴
    f_doppler = 2 * velocity / lambda_  # 多普勒频率
    f_range = 2 * distance * S / c # 距离频率
    sample_data = np.zeros((num_tx, num_rx, num_chirps, num_samples), dtype=complex)
    
    for tx in range(num_tx):
        for rx in range(num_rx):
            for chirp in range(num_chirps):
                phase_shift = 2 * np.pi * f_doppler * chirp * T_chirp
                azimuth_shift = 2 * np.pi * d * np.sin(np.deg2rad(azimuth)) * tx / lambda_
                elevation_shift = 2 * np.pi * d * np.sin(np.deg2rad(elevation)) * rx / lambda_
                sample_data[tx, rx, chirp, :] = np.exp(1j * (2 * np.pi * f_range * t + phase_shift + azimuth_shift + elevation_shift))
    return sample_data
# 示例数据
velocity = 3  # 目标速度 (m/s)
distance = 1  # 目标距离 (m)
azimuth = 23   # 目标方位角 (degree)
elevation = 15  # 目标仰角 (degree)
data = generate_moving_target_data(num_tx, num_rx, num_chirps, num_samples, fc, c, velocity, distance, azimuth, elevation)

# ========================== 数据预处理 ==========================
data_reshaped = np.transpose(data, (3, 2, 0, 1))  # (距离, 多普勒, 发射天线, 接收天线)

# ========================== 距离-多普勒处理 ==========================
range_fft = np.fft.fft(data_reshaped, axis=0)
doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1), axes=1)

# ========================== CFAR目标检测 ==========================

def cfar_2d(matrix, guard_win=2, train_win=4, false_alarm=1e-6):
    """
    2D CFAR检测
    """
    num_range, num_doppler = matrix.shape
    mask = np.zeros((num_range, num_doppler), dtype=bool)
    
    for r in range(num_range):
        for d in range(num_doppler):
            # 排除保护单元
            r_start = max(0, r - guard_win)
            r_end = min(num_range, r + guard_win + 1)
            d_start = max(0, d - guard_win)
            d_end = min(num_doppler, d + guard_win + 1)
            
            # 计算训练单元均值
            train_cells = matrix[r_start:r_end, d_start:d_end]
            train_mean = np.mean(train_cells)
            
            # 阈值判断
            threshold = train_mean * (-np.log(false_alarm))
            if matrix[r, d] > threshold:
                mask[r, d] = True
    return mask
power = np.abs(doppler_fft[:, :, 0, 0]) ** 2
target_mask = cfar_2d(power)

# ========================== MUSIC角度估计 ==========================
def music_algorithm(snapshots, theta_scan, phi_scan, d, lambda_, num_tx, num_rx, K=1):
    """
    输入:snapshots (num_tx * num_rx, num_snapshots)
    输出:music_spectrum (方位角扫描点数, 仰角扫描点数)
    """
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

# 遍历所有距离-多普勒单元，检测目标并估计方位角和仰角
angle_map = np.zeros((num_samples, num_chirps, 2))  # 存储每个单元的方位角和仰角估计值
intensity_map = np.zeros((num_samples, num_chirps)) # 存储每个单元的强度值

for range_bin in range(num_samples):
    for doppler_bin in range(num_chirps):
        if target_mask[range_bin, doppler_bin]:
            # 提取当前单元的MIMO虚拟阵列信号
            snapshots = doppler_fft[range_bin, doppler_bin, :, :].reshape(num_tx * num_rx, -1)  # (8, 128)
            
            # 计算MUSIC谱
            spectrum = music_algorithm(snapshots, theta_scan, phi_scan, d, lambda_, num_tx, num_rx, K=1)
            
            # 提取主峰方位角和仰角
            peak_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
            angle_map[range_bin, doppler_bin, 0] = - theta_scan[peak_idx[0]]
            angle_map[range_bin, doppler_bin, 1] = - phi_scan[peak_idx[1]]

            # 计算强度
            intensity_map[range_bin, doppler_bin] = np.max(spectrum)

# ========================== 结果可视化 ==========================
range_idx, doppler_idx = np.where(target_mask)
range_frequency = (range_idx) * (num_samples / T_chirp) / num_samples
ranges = c * range_frequency / (2 * S)

T = (T_chirp+T_idle)
doppler_frequency = (doppler_idx) / T / num_chirps
velocities = doppler_frequency * lambda_ / 2

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(ranges, angle_map[range_idx, doppler_idx, 1], angle_map[range_idx, doppler_idx, 0],
                c=velocities, cmap='viridis')
ax.set_xlabel('Range (m)')
ax.set_ylabel('Elevation (degree)')
ax.set_zlabel('Azimuth (degree)')
plt.colorbar(sc, label='Velocity (m/s)')
plt.show()