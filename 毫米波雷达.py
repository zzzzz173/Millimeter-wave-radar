import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d

# ========================== 参数设置 ==========================
fc = 77e9  # 雷达载频 77 GHz
c = 3e8  # 光速
lambda_ = c / fc  # 波长
d = lambda_ / 2  # 阵元间距 (均匀线阵)
num_tx = 2  # 发射天线数
num_rx = 4  # 接收天线数
num_chirps = 128  # 每帧啁啾数
num_samples = 128  # 每啁啾采样点数
S = 99.987e9  # 频率斜率
theta_scan = np.linspace(-90, 90, 181)  # 方位角搜索范围
phi_scan = np.linspace(-90, 90, 181)  # 仰角搜索范围
sample_rate = 4e6  # 采样率


# ========================== 生成模拟数据 ==========================
def generate_moving_target_data(num_tx, num_rx, num_chirps, num_samples, fc, c, velocity, distance, azimuth, elevation):
    """
    生成模拟数据，添加信噪比控制
    """
    lambda_ = c / fc
    t = np.arange(num_samples) / sample_rate
    f_doppler = 2 * velocity / lambda_
    f_range = 2 * distance * S / c
    sample_data = np.zeros((num_tx, num_rx, num_chirps, num_samples), dtype=complex)

    # 添加信号
    for tx in range(num_tx):
        for rx in range(num_rx):
            for chirp in range(num_chirps):
                phase_shift = 2 * np.pi * f_doppler * chirp / num_chirps
                azimuth_shift = 2 * np.pi * d * np.sin(np.deg2rad(azimuth)) * tx / lambda_
                elevation_shift = 2 * np.pi * d * np.sin(np.deg2rad(elevation)) * rx / lambda_
                signal = np.exp(1j * (2 * np.pi * f_range * t + phase_shift + azimuth_shift + elevation_shift))

                # 添加噪声，SNR=20dB
                noise = np.random.normal(0, 0.1, num_samples) + 1j * np.random.normal(0, 0.1, num_samples)
                sample_data[tx, rx, chirp, :] = signal + noise

    return sample_data


# 示例数据：模拟在一定位置以一定速率运动的物体
velocity = 30  # 目标速度 (m/s)
distance = 50  # 目标距离 (m)
azimuth = 23  # 目标方位角 (degree)
elevation = 15  # 目标仰角 (degree)
data = generate_moving_target_data(num_tx, num_rx, num_chirps, num_samples, fc, c, velocity, distance, azimuth,
                                   elevation)

# ========================== 数据预处理 ==========================
# 重组数据为 (距离, 多普勒, 发射天线, 接收天线) = (128, 128, 2, 4)
data_reshaped = np.transpose(data, (3, 2, 0, 1))

# ========================== 距离-多普勒处理 ==========================
# 距离FFT (快时间维)
range_fft = np.fft.fft(data_reshaped, axis=0)

# 多普勒FFT (慢时间维)
doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1), axes=1)


# ========================== CFAR目标检测 ==========================

def cfar_2d(matrix, guard_win=2, train_win=4, false_alarm=1e-3):  # 调整虚警率
    """
    优化的2D CFAR检测，调整检测灵敏度
    """
    num_range, num_doppler = matrix.shape
    mask = np.zeros((num_range, num_doppler), dtype=bool)

    # 使用对数变换增强动态范围
    matrix = np.log10(matrix + 1)

    # 使用滑动窗口计算局部均值和标准差
    kernel = np.ones((2 * train_win + 1, 2 * train_win + 1))
    kernel[train_win - guard_win:train_win + guard_win + 1,
    train_win - guard_win:train_win + guard_win + 1] = 0
    kernel = kernel / np.sum(kernel)

    local_mean = convolve2d(matrix, kernel, mode='same')
    local_std = np.sqrt(convolve2d((matrix - local_mean) ** 2, kernel, mode='same'))

    # 自适应阈值检测
    threshold = local_mean + 3 * local_std  # 使用3倍标准差作为阈值
    mask = matrix > threshold

    return mask


# 对某个天线（如第一个发射天线和第一个接收天线）的距离-多普勒图进行CFAR检测
power = np.abs(doppler_fft[:, :, 0, 0]) ** 2  # 取第一个发射天线和第一个接收天线的功率
target_mask = cfar_2d(power)


# ========================== MUSIC角度估计 ==========================
def music_algorithm(snapshots, theta_scan, phi_scan, d, lambda_, num_tx, num_rx, K=1):
    """
    优化的MUSIC算法
    添加SVD分解提高稳定性
    """
    M, N = snapshots.shape

    # 使用SVD替代特征值分解
    U, S, Vh = np.linalg.svd(snapshots @ snapshots.conj().T / N)
    Un = U[:, K:]

    # 预计算角度向量
    theta_rad = np.deg2rad(theta_scan)
    phi_rad = np.deg2rad(phi_scan)

    a_tx_matrix = np.exp(-1j * 2 * np.pi * d * np.outer(np.sin(theta_rad), np.arange(num_tx)) / lambda_)
    a_rx_matrix = np.exp(-1j * 2 * np.pi * d * np.outer(np.sin(phi_rad), np.arange(num_rx)) / lambda_)

    music_spectrum = np.zeros((len(theta_scan), len(phi_scan)))

    # 向量化计算MUSIC谱
    for i, a_tx in enumerate(a_tx_matrix):
        for j, a_rx in enumerate(a_rx_matrix):
            a = np.kron(a_tx, a_rx).reshape(-1, 1)
            music_spectrum[i, j] = 1 / np.abs(a.conj().T @ Un @ Un.conj().T @ a).squeeze()

    return music_spectrum / np.max(music_spectrum)


# 遍历所有距离-多普勒单元，检测目标并估计方位角和仰角
angle_map = np.zeros((num_samples, num_chirps, 2))  # 存储每个单元的方位角和仰角估计值
intensity_map = np.zeros((num_samples, num_chirps))  # 存储每个单元的强度值

for range_bin in range(num_samples):
    for doppler_bin in range(num_chirps):
        if target_mask[range_bin, doppler_bin]:
            # 提取当前单元的MIMO虚拟阵列信号
            snapshots = doppler_fft[range_bin, doppler_bin, :, :].reshape(num_tx * num_rx, -1)  # (8, 128)

            # 计算MUSIC谱
            spectrum = music_algorithm(snapshots, theta_scan, phi_scan, d, lambda_, num_tx, num_rx, K=1)

            # 提取主峰方位角和仰角
            peak_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
            angle_map[range_bin, doppler_bin, 0] = theta_scan[peak_idx[0]]
            angle_map[range_bin, doppler_bin, 1] = phi_scan[peak_idx[1]]

            # 计算强度
            intensity_map[range_bin, doppler_bin] = np.max(spectrum)

# ========================== 结果可视化 ==========================
# 绘制距离-多普勒-方位角-仰角点云
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 提取目标点
range_idx, doppler_idx = np.where(target_mask)
azimuths = angle_map[range_idx, doppler_idx, 0]
elevations = angle_map[range_idx, doppler_idx, 1]
intensities = intensity_map[range_idx, doppler_idx]

range_frequency = range_idx * sample_rate / num_samples
doppler_frequency = (doppler_idx - num_chirps // 2) * sample_rate / num_chirps

ranges = c * range_frequency / (2 * S)
velocities = doppler_frequency * lambda_ / 2


# 修改可视化部分
def plot_results(ranges, elevations, azimuths, velocities, intensities):
    """
    改进的可视化函数 - 移除强度阈值过滤
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 设置合理的显示范围
    ax.set_xlim([0, 100])  # 距离范围0-100米
    ax.set_ylim([-90, 90])  # 仰角范围-90到90度
    ax.set_zlim([-90, 90])  # 方位角范围-90到90度

    # 绘制所有检测到的点
    sc = ax.scatter(ranges,
                    elevations,
                    azimuths,
                    c=velocities,
                    s=100,  # 增大点的大小
                    cmap='viridis',
                    alpha=0.6)  # 添加透明度

    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Elevation (degree)')
    ax.set_zlabel('Azimuth (degree)')
    plt.colorbar(sc, label='Velocity (m/s)')

    # 添加目标真实位置的标记
    ax.scatter([distance], [elevation], [azimuth],
               color='red', s=200, marker='*',
               label='True Target')
    ax.legend()

    plt.title(f'Detected Points (Total: {len(ranges)})')
    plt.show()


def plot_rdai_heatmap(ranges, azimuths, elevations, intensities):
    """
    改进的RDAI热力图函数
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 设置合理的显示范围
    ax.set_xlim([0, 100])
    ax.set_ylim([-90, 90])
    ax.set_zlim([-90, 90])

    # 绘制所有点
    sc = ax.scatter(ranges,
                    azimuths,
                    elevations,
                    c=intensities,
                    s=100,  # 增大点的大小
                    cmap='viridis',
                    alpha=0.6)

    # 添加目标真实位置的标记
    ax.scatter([distance], [azimuth], [elevation],
               color='red', s=200, marker='*',
               label='True Target')
    ax.legend()

    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Azimuth (degree)')
    ax.set_zlabel('Elevation (degree)')
    plt.colorbar(sc, label='Intensity')
    plt.title('RDAI Heatmap')
    plt.show()


# 在调用可视化函数之前，添加速度值的限制
velocity_limit = 50  # 限制速度范围为±50m/s
valid_velocity_mask = (np.abs(velocities) <= velocity_limit)
ranges = ranges[valid_velocity_mask]
elevations = elevations[valid_velocity_mask]
azimuths = azimuths[valid_velocity_mask]
velocities = velocities[valid_velocity_mask]
intensities = intensities[valid_velocity_mask]

# 调用修改后的函数
plot_results(ranges, elevations, azimuths, velocities, intensities)
plot_rdai_heatmap(ranges, azimuths, elevations, intensities)

# 更新调试信息
print("检测到的目标点数量:", len(ranges))
print("速度范围（过滤后）:", np.min(velocities), "到", np.max(velocities))
print("目标真实参数:")
print(f"距离: {distance}m")
print(f"速度: {velocity}m/s")
print(f"方位角: {azimuth}度")
print(f"仰角: {elevation}度")
