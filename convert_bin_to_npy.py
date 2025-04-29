import numpy as np
import os

bin_folder = r"c:\Users\Albert\Desktop\push1"
npy_folder = r"c:\Users\Albert\Desktop\push1_npy"
os.makedirs(npy_folder, exist_ok=True)

for file in os.listdir(bin_folder):
    if file.endswith('.bin'):
        bin_path = os.path.join(bin_folder, file)
        # 假设每个bin文件存储float32，且每帧为32*32
        data = np.fromfile(bin_path, dtype=np.float32)
        # 自动推断帧数N
        N = data.size // (32 * 32)
        if N == 0:
            print(f"{file} 数据量不足，跳过")
            continue
        data = data[:N*32*32].reshape((N, 32, 32))
        # 保存为npy文件
        npy_path = os.path.join(npy_folder, file.replace('.bin', '.npy'))
        np.save(npy_path, data)
        print(f"已保存: {npy_path}")