function [DRAI,RDIs,m] = noise_elimination(RDIs, doppler_bin_threshold, ...
    scale_factor,Angle_bin)
    % 输入参数：
    % RDIs: 接收通道的 Range-Doppler 图像矩阵，大小为 [K x L]，其中 K 是范围（range）维度，L 是多普勒（Doppler）维度
    % doppler_bin_threshold: 多普勒频率阈值，用于去除静止杂波
    % scale_factor: 多普勒功率阈值的比例因子

    % 输出：
    % DRAI: 动态范围角图像（Dynamic Range Angle Image）

    [K, L,N] = size(RDIs);
    for i=1:N
        RDIs(:,L/2-doppler_bin_threshold-1:L/2+doppler_bin_threshold,i)=0;
        % RDIs(:,28:36,i)=0;
    end

    % figure;
    % speed_profile_temp = reshape(RDIs(:,:,1),K,L);   
    % speed_profile_Temp = speed_profile_temp';
    % [X,Y]=meshgrid((0:K-1),(-L/2:L/2-1));
    % mesh(X,Y,(abs(speed_profile_Temp))); 
    % xlabel('距离(m)');ylabel('速度(m/s)');zlabel('信号幅值');

    avRD=mean(abs(RDIs),3);
    
    speed_profile_max = sum(abs(avRD),1);
    T=scale_factor*max(speed_profile_max);
    
    Drai=zeros(K,Angle_bin);
    Rda=zeros(K,L,Angle_bin);
    for n=1:K   %range
        for m=1:L   %chirp
          temp=RDIs(n,m,:);   
          temp_fft=fftshift(fft(temp,Angle_bin));    %对2D FFT结果进行Q点FFT
          Rda(n,m,:)=temp_fft;  
        end
    end
    for i=1:L
        if speed_profile_max(i)>=T
            Drai=squeeze(abs(Rda(:,i,:)))+Drai;
        end
    end
    Drai_mm(1:32,:)=Drai(1:32,:);
    % 1. 计算幅度矩阵，并找到峰值坐标 (x_peak, y_peak)
    abs_Drai_mm = abs(Drai_mm);  % 幅度矩阵
    [max_val, max_idx] = max(abs_Drai_mm(:));  % 找到最大值及其线性索引
    [y_peak, x_peak] = ind2sub([32, 32], max_idx);  % 转换为二维坐标 (x, y)
    
    % 2. 定义噪声计算区域（排除 x±2 和 y±4 的邻域）
    x_mask = true(1, 32);  % 初始化 x 范围
    x_mask(max(1, x_peak-2) : min(32, x_peak+2)) = false;  % 排除 x-2 ≤ x ≤ x+2
    
    y_mask = true(1, 32);  % 初始化 y 范围
    y_mask(max(1, y_peak-4) : min(32, y_peak+4)) = false;  % 排除 y-4 ≤ y ≤ y+4
    
    % 3. 提取噪声区域的数据
    noise_region = Drai_mm(y_mask, x_mask);
    noise_mean = mean(abs(noise_region(:)));  % 计算噪声均值
    
    % 4. 计算 m = log10((峰值 + 噪声) / 噪声)
    m = log10((max_val + noise_mean) / noise_mean);
    

    % 输出结果
    % fprintf('m = %.4f\n', m);
    DRAI=abs(Drai);
    
end