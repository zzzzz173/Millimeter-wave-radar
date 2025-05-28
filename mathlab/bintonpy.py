import numpy as np
import matplotlib.pyplot as plt
import os

# 参数设置
frame_num = 50
T = 0.08
c = 3.0e8
freqSlope = 99.987e12
Tc = 160e-6
Fs = 4e6
f0 = 77e9
_lambda = c / f0
d = _lambda / 2
NSample = 128
Range_Number = 128

Chirp = 64
Doppler_Number = 64
NChirp = frame_num * Chirp
Rx_Number = 4
Tx_Number = 2
TR_x_Number = Tx_Number * Rx_Number
Angle_bin = 32
numADCBits = 16
readframe = 25

def noise_elimination(RDIs, doppler_bin_threshold, scale_factor, Angle_bin):
    """
    RDIs: numpy array, shape [K, L, N]
    doppler_bin_threshold: int, 多普勒频率阈值
    scale_factor: float, 多普勒功率阈值的比例因子
    Angle_bin: int, 角度FFT点数
    返回:
        DRAI: 动态范围角图像
        RDIs: 处理后的RDIs
        m: 峰值与噪声比值的对数
    """
    K, L, N = RDIs.shape
    # 去除静止杂波
    for i in range(N):
        RDIs[:, int(L/2)-doppler_bin_threshold:int(L/2)+doppler_bin_threshold+1, i] = 0

    avRD = np.mean(np.abs(RDIs), axis=2)
    speed_profile_max = np.sum(np.abs(avRD), axis=0)
    T = scale_factor * np.max(speed_profile_max)

    Drai = np.zeros((K, Angle_bin), dtype=np.complex128)
    Rda = np.zeros((K, L, Angle_bin), dtype=np.complex128)
    for n in range(K):
        for m in range(L):
            temp = RDIs[n, m, :]
            temp_fft = np.fft.fftshift(np.fft.fft(temp, Angle_bin))
            Rda[n, m, :] = temp_fft

    for i in range(L):
        if speed_profile_max[i] >= T:
            Drai += np.abs(Rda[:, i, :])

    # 只取前32行
    Drai_mm = np.zeros((32, Angle_bin), dtype=np.complex128)
    Drai_mm[:32, :] = Drai[:32, :]

    # 1. 计算幅度矩阵，并找到峰值坐标
    abs_Drai_mm = np.abs(Drai_mm)
    max_val = np.max(abs_Drai_mm)
    max_idx = np.argmax(abs_Drai_mm)
    y_peak, x_peak = np.unravel_index(max_idx, (32, 32))

    # 2. 定义噪声计算区域（排除 x±2 和 y±4 的邻域）
    x_mask = np.ones(32, dtype=bool)
    x_mask[max(0, x_peak-2):min(32, x_peak+3)] = False  # 注意Python切片右开
    y_mask = np.ones(32, dtype=bool)
    y_mask[max(0, y_peak-4):min(32, y_peak+5)] = False

    # 3. 提取噪声区域的数据
    noise_region = abs_Drai_mm[np.ix_(y_mask, x_mask)]
    noise_mean = np.mean(np.abs(noise_region))

    # 4. 计算 m = log10((峰值 + 噪声) / 噪声)
    m = np.log10((max_val + noise_mean) / noise_mean)

    DRAI = np.abs(Drai)
    return DRAI, RDIs, m

Filename = r"F:\gesture_datas\big1\data__1.bin"
adcDataRow = np.fromfile(Filename, dtype=np.int16)

lvds_data = adcDataRow[0::2] + 1j * adcDataRow[1::2]
expected_size = Range_Number * TR_x_Number * NChirp
actual_size = lvds_data.size

if expected_size != actual_size:
    raise ValueError(f'数据大小不匹配！预期 {expected_size} 个元素，实际有 {actual_size} 个元素')

ADC_data = lvds_data.reshape((Range_Number, TR_x_Number, NChirp), order='F')
ADC_data = np.transpose(ADC_data, (0,2,1)) #[Range_Number, NChirp, TR_x_Number]

m_values = np.zeros(frame_num)
motion_frames = []
motion_threshold = 1
data = np.zeros((frame_num, 32, 32))

for readframe in range(frame_num):
    ADC_Data_frame = ADC_data[:, readframe*Chirp:(readframe+1)*Chirp, :]
    range_win = np.hamming(Range_Number + 2)
    range_profile = np.zeros((Range_Number, Chirp, TR_x_Number), dtype=complex)
    for k in range(TR_x_Number):
        for m in range(Chirp):
            inputMat = ADC_Data_frame[:, m, k]
            inputMat = inputMat - np.mean(inputMat)
            inputMat = inputMat * range_win[1:Range_Number+1]
            range_profile[:, m, k] = np.fft.fft(inputMat, Range_Number)
    doppler_win = np.hamming(Chirp + 2)
    speed_profile = np.zeros((Range_Number, Doppler_Number, TR_x_Number), dtype=complex)
    for k in range(TR_x_Number):
        for n in range(Range_Number):
            temp = range_profile[n, :, k] * doppler_win[1:Chirp+1]
            speed_profile[n, :, k] = np.fft.fftshift(np.fft.fft(temp, Doppler_Number))

    speed_profile_temp = speed_profile[:32, :, :]

    angle_profile_display, speed_profile, m = noise_elimination(speed_profile_temp, 4, 0.8, Angle_bin)

    m_values[readframe] = m

    # 得到Range-Azimuth热力图
    angle_profile_display_32x32 = angle_profile_display[:32, :]
    data[readframe, :, :] = angle_profile_display_32x32

    # 检测运动
    if m > motion_threshold:
        motion_frames.append(readframe)

# 保存有运动的帧数据
if motion_frames:
    motion_data = data[motion_frames, :, :]
    t = len(motion_frames)
    reshaped_data = np.zeros((t, 32, 32))
    for i in range(t):
        current_frame = motion_data[i, :, :]
        reshaped_data[i, :, :] = current_frame

    output_dir = r'F:\gesture_datas\big1'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, 'motion_frames.npy')
    np.save(save_path, reshaped_data)

    print(f'检测到运动的帧数: {t}')
    print(f'运动帧号: {motion_frames}')
    print(f'文件已保存到: {save_path}')
    print(f'数据维度: {reshaped_data.shape}')
else:
    print('未检测到运动')

# 绘制 m 值的柱状图
plt.figure()
plt.bar(np.arange(1, frame_num+1), m_values)
plt.title('Bar Chart of m Values for Each Frame')
plt.xlabel('Frame Number')
plt.ylabel('m Value')
plt.grid(True)
plt.show()