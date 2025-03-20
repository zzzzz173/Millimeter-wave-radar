import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.ndimage import maximum_filter

# ========================== 参数设置 ==========================
fc = 77e9          # 雷达载频 77 GHz
c = 3e8            # 光速
lambda_ = c / fc   # 波长
d = lambda_ / 2    # 阵元间距 (均匀线阵)
num_tx = 2         # 发射天线数
num_rx = 4         # 接收天线数
num_chirps = 128   # 每帧啁啾数
num_samples = 128  # 每啁啾采样点数
S = 99.987e9       # 频率斜率
theta_scan = np.linspace(-90, 90, 181)  # 方位角搜索范围
phi_scan = np.linspace(-90, 90, 181)    # 仰角搜索范围
sample_rate = 4e6  # 采样率
chirp_time = 1 / (sample_rate / num_samples)  # 啁啾周期
frame_time = chirp_time * num_chirps  # 帧周期

# ========================== 生成模拟数据 ==========================
def generate_moving_target_data(num_tx, num_rx, num_chirps, num_samples, fc, c, targets):
    """
    生成多个运动目标的模拟雷达数据
    targets: 包含多个目标参数的列表，每个目标包含 [velocity, distance, azimuth, elevation]
    """
    lambda_ = c / fc
    t = np.arange(num_samples) / sample_rate  # 时间轴
    sample_data = np.zeros((num_tx, num_rx, num_chirps, num_samples), dtype=complex)
    
    for velocity, distance, azimuth, elevation in targets:
        f_doppler = 2 * velocity / lambda_  # 多普勒频率
        f_range = 2 * distance * S / c  # 距离频率
        
        for tx in range(num_tx):
            for rx in range(num_rx):
                for chirp in range(num_chirps):
                    phase_shift = 2 * np.pi * f_doppler * chirp * chirp_time
                    azimuth_shift = 2 * np.pi * d * np.sin(np.deg2rad(azimuth)) * tx / lambda_
                    elevation_shift = 2 * np.pi * d * np.sin(np.deg2rad(elevation)) * rx / lambda_
                    sample_data[tx, rx, chirp, :] += np.exp(1j * (2 * np.pi * f_range * t + phase_shift + azimuth_shift + elevation_shift))
    
    return sample_data

# 示例数据：模拟4个不同位置和速度的运动目标
targets = [
    [5, 45, 20, -20],     # 目标1：较慢速度
    [-3, 46, -15, -22],   # 目标2：较慢速度
    [8, 47, 30, -18],     # 目标3：中等速度
    [-6, 48, -25, -16]    # 目标4：中等速度
]
data = generate_moving_target_data(num_tx, num_rx, num_chirps, num_samples, fc, c, targets)

# ========================== 数据预处理 ==========================
# 重组数据为 (距离, 多普勒, 发射天线, 接收天线) = (128, 128, 2, 4)
data_reshaped = np.transpose(data, (3, 2, 0, 1))

# ========================== 距离-多普勒处理 ==========================
# 距离FFT (快时间维)
range_fft = np.fft.fft(data_reshaped, axis=0)

# 多普勒FFT (慢时间维)
doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1), axes=1)

# 功率谱预处理
power = np.abs(doppler_fft[:, :, 0, 0]) ** 2
power = power / np.max(power)

# ========================== CFAR目标检测 ==========================
def cfar_2d(matrix, guard_win=5, train_win=10, false_alarm=1e-6):
    """
    2D CFAR检测，使用严格的阈值和较大的窗口
    """
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
            training_cells[r-r_start:r-r_start+guard_cells.shape[0], 
                         d-d_start:d-d_start+guard_cells.shape[1]] = 0
            
            # 计算训练单元均值和标准差
            train_mean = np.mean(training_cells[training_cells != 0])
            train_std = np.std(training_cells[training_cells != 0])
            
            # 使用严格的阈值
            threshold = train_mean + 8 * train_std
            if matrix[r, d] > threshold:
                mask[r, d] = True
    
    return mask

# 对功率谱进行CFAR检测
target_mask = cfar_2d(power)

# 添加强度阈值过滤
intensity_threshold = 0.3  # 使用较高的强度阈值
target_mask = target_mask & (power > intensity_threshold)

# 添加距离和多普勒的局部最大值检测
local_max = maximum_filter(power, size=7)  # 使用较大的窗口
target_mask = target_mask & (power == local_max)

# 计算距离和速度的bin到实际值的转换因子
range_scale = c / (2 * S * num_samples / sample_rate)  # 距离分辨率
velocity_scale = lambda_ / (2 * frame_time)  # 速度分辨率

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
            snapshots = doppler_fft[range_bin, doppler_bin, :, :].reshape(num_tx * num_rx, -1)
            
            # 计算MUSIC谱
            spectrum = music_algorithm(snapshots, theta_scan, phi_scan, d, lambda_, num_tx, num_rx, K=1)
            
            # 提取主峰方位角和仰角
            peak_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
            angle_map[range_bin, doppler_bin, 0] = theta_scan[peak_idx[0]]
            angle_map[range_bin, doppler_bin, 1] = phi_scan[peak_idx[1]]

            # 计算强度
            intensity_map[range_bin, doppler_bin] = np.max(spectrum)

# ========================== 结果可视化 ==========================
# 提取目标点
range_bins, doppler_bins = np.where(target_mask)

# 计算实际距离和速度
ranges = range_bins * range_scale
velocities = (doppler_bins - num_chirps // 2) * velocity_scale

# 提取角度和强度
azimuths = angle_map[range_bins, doppler_bins, 0]
elevations = angle_map[range_bins, doppler_bins, 1]
intensities = intensity_map[range_bins, doppler_bins]

# 过滤有效目标（在合理范围内的目标）
valid_range_mask = (ranges >= 40) & (ranges <= 55)
valid_velocity_mask = (velocities >= -10) & (velocities <= 10)
valid_mask = valid_range_mask & valid_velocity_mask

ranges = ranges[valid_mask]
velocities = velocities[valid_mask]
azimuths = azimuths[valid_mask]
elevations = elevations[valid_mask]
intensities = intensities[valid_mask]

print(f"检测到的目标数量: {len(ranges)}")
print("\n目标信息:")
for i in range(len(ranges)):
    print(f"目标 {i+1}:")
    print(f"  距离: {ranges[i]:.2f} m")
    print(f"  速度: {velocities[i]:.2f} m/s")
    print(f"  方位角: {azimuths[i]:.2f}°")
    print(f"  仰角: {elevations[i]:.2f}°")
    print(f"  强度: {intensities[i]:.2f}")
    print()

# 绘制点云
plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
scatter = ax.scatter(ranges, elevations, azimuths, c=velocities, cmap='viridis', s=100)
ax.set_xlabel('Range (m)')
ax.set_ylabel('Elevation (degree)')
ax.set_zlabel('Azimuth (degree)')
plt.colorbar(scatter, label='Velocity (m/s)')
plt.title('4D Radar Point Cloud')
plt.show()

# 绘制RDAI热力图
plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
scatter = ax.scatter(ranges, azimuths, elevations, c=intensities, cmap='viridis', s=100)
ax.set_xlabel('Range (m)')
ax.set_ylabel('Azimuth (degree)')
ax.set_zlabel('Elevation (degree)')
plt.colorbar(scatter, label='Intensity')
plt.title('RDAI Heatmap')
plt.show()
