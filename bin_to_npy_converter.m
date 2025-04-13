function bin_to_npy_converter()
    % 选择bin文件
    [filename, pathname] = uigetfile('*.bin', '选择bin文件');
    if isequal(filename, 0)
        disp('用户取消了文件选择');
        return;
    end
    
    % 构建完整的文件路径
    fullpath = fullfile(pathname, filename);
    
    % 读取bin文件
    fid = fopen(fullpath, 'rb');
    if fid == -1
        error('无法打开文件');
    end
    
    % 读取数据
    data = fread(fid, 'float32');
    fclose(fid);
    
    % 重塑数据（根据您的数据格式调整）
    % 假设数据是NxM的矩阵，其中N是样本数，M是特征数
    % 您需要根据实际数据格式调整这些参数
    num_samples = length(data) / 128; % 假设每个样本有128个特征
    data = reshape(data, 128, num_samples)';
    
    % 创建桌面dataset文件夹（如果不存在）
    desktop_path = [getenv('USERPROFILE') '\Desktop'];
    dataset_path = fullfile(desktop_path, 'dataset');
    if ~exist(dataset_path, 'dir')
        mkdir(dataset_path);
    end
    
    % 构建npy文件的保存路径
    npy_filename = strrep(filename, '.bin', '.npy');
    save_path = fullfile(dataset_path, npy_filename);
    
    % 保存为npy文件
    save_npy_custom(data, save_path);
    
    disp(['转换完成！文件已保存到: ' save_path]);
end

function save_npy_custom(data, filename)
    % 自定义的npy文件保存函数
    fid = fopen(filename, 'wb');
    if fid == -1
        error('无法创建文件');
    end
    
    % 写入npy文件头
    % 版本信息
    fwrite(fid, uint8([147 'NUMPY']), 'uint8');
    fwrite(fid, uint8([1 0]), 'uint8');  % 版本号 1.0
    
    % 构建头部信息
    shape = size(data);
    header = sprintf('{"descr": "<f4", "fortran_order": False, "shape": (%d, %d), }', shape(1), shape(2));
    header_len = length(header);
    padding = 64 - mod(header_len + 10, 64);
    header = [header repmat(' ', 1, padding)];
    header_len = length(header);
    
    % 写入头部长度（小端序）
    header_len_bytes = typecast(uint16(header_len), 'uint8');
    if ~isequal(header_len_bytes(1), uint8(header_len))
        header_len_bytes = fliplr(header_len_bytes);
    end
    fwrite(fid, header_len_bytes, 'uint8');
    
    % 写入头部信息
    fwrite(fid, uint8(header), 'uint8');
    
    % 写入数据
    fwrite(fid, data', 'float32');
    
    fclose(fid);
end 