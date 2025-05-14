clc; clear; close all;
%% =========================================================================
%% 参数设置
frame_num=50;           %帧数
T=0.08;                  %帧周期
c=3.0e8;                %光速
freqSlope=99.987e12;        %调频斜率
Tc=160e-6;              %chirp总周期
Fs=4e6;                 %采样率
f0=77e9;                %初始频率
lambda=c/f0;            %雷达信号波长
d=lambda/2 ;            %天线阵列间距
NSample=128;            %距离向FFT点数
Range_Number=128;       %采样点数/脉冲

Chirp=64;              %每帧脉冲数
Doppler_Number=64;     %速度向FFT点数
NChirp=frame_num*Chirp;  %总脉冲数
Rx_Number=4;            %RX天线通道数
Tx_Number=2;            %TX天线通道数
TR_x_Number=Tx_Number*Rx_Number; %等效通道数
Angle_bin= 32;                %角度FFT点数
numADCBits = 16;
readframe=25;
%0.047m 的距离分辨率和 0.039m/s 的速度分辨率

%% 读取Bin文件
Filename ="E:\gesture\push4\data__8.bin";
fid = fopen(Filename, 'r');
adcDataRow = fread(fid, 'int16');
fclose(fid);

% 数据重组（IQ信号）
lvds_data = adcDataRow(1:2:end) + 1i * adcDataRow(2:2:end);

% 计算预期的数组大小
expected_size = Range_Number * TR_x_Number * NChirp;
actual_size = length(lvds_data);
fprintf('预期数组大小: %d\n', expected_size);
fprintf('实际数组大小: %d\n', actual_size);

% 检查数组大小是否匹配
if actual_size ~= expected_size
    error('数据大小不匹配！预期 %d 个元素，实际有 %d 个元素', expected_size, actual_size);
end

ADC_Data = reshape(lvds_data, [Range_Number, TR_x_Number, NChirp]);
ADC_Data = permute(ADC_Data, [1, 3, 2]); % [Range_Number × NChirp × TR_x_Number]

m_values = zeros(frame_num, 1); % Array to store the values of 'm'
motion_frames = []; % 存储有运动的帧号
motion_threshold = 1; % 运动检测阈值
data = zeros(frame_num, 32, 32); % 预分配数据数组

for readframe=1:frame_num
    %% 提取当前帧数据
    ADC_Data_frame = ADC_Data(:, (readframe-1)*Chirp+1 : readframe*Chirp, :);

    %% 距离FFT（加海明窗）
    range_win = hamming(Range_Number+2);
    range_profile = zeros(Range_Number, Chirp, TR_x_Number);
    for k = 1:TR_x_Number
        for m = 1:Chirp
            % Range FFT - 增加均值去除
            inputMat = ADC_Data_frame(:, m, k);
            inputMat = inputMat - mean(inputMat);
            inputMat = inputMat .* range_win(2:Range_Number+1);
            range_profile(:, m, k) = fft(inputMat, Range_Number);
        end
    end

    %% 多普勒FFT（加海明窗）
    doppler_win = hamming(Chirp+2);
    speed_profile = zeros(Range_Number, Doppler_Number, TR_x_Number);
    for k = 1:TR_x_Number
        for n = 1:Range_Number
            temp = range_profile(n, :, k) .* doppler_win(2:Chirp+1)';
            speed_profile(n, :, k) = fftshift(fft(temp, Doppler_Number));
        end
    end

    speed_profile_temp=speed_profile(1:32,:,:);
    [angle_profile_display, speed_profile, m] = noise_elimination(speed_profile_temp,4, 0.8, Angle_bin);

    % Store the 'm' value for the current frame
    m_values(readframe) = m;

    %% 得到Range-Azimuth热力图
    angle_profile_display_32x32 = angle_profile_display(1:32, :);
    data(readframe,:,:)=angle_profile_display_32x32;

    % 检测运动
    if m > motion_threshold
        motion_frames = [motion_frames, readframe];
    end
end

% 保存有运动的帧数据
if ~isempty(motion_frames)
    % 提取有运动的帧数据
    motion_data = data(motion_frames, :, :);
    
    % 重新组合成t*32*32格式
    t = length(motion_frames); % 运动帧的数量
    reshaped_data = zeros(t, 32, 32);
    
    for i = 1:t
        % 提取当前帧的32x32数据
        current_frame = squeeze(motion_data(i, :, :));
        reshaped_data(i, :, :) = current_frame;
    end
    
    % 指定保存路径
    output_dir = 'D:\桌面\output';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % 保存为npy格式
    save_path = fullfile(output_dir, 'motion_frames.npy');
    writeNPY(reshaped_data, save_path);
    
    fprintf('检测到运动的帧数: %d\n', t);
    fprintf('运动帧号: %s\n', mat2str(motion_frames));
    fprintf('文件已保存到: %s\n', save_path);
    fprintf('数据维度: %d x %d x %d\n', size(reshaped_data));
else
    fprintf('未检测到运动\n');
end

% Plot the bar chart of 'm' values
figure;
bar(m_values);
title('Bar Chart of m Values for Each Frame');
xlabel('Frame Number');
ylabel('m Value');
grid on;

