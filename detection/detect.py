import socket
import numpy as np
import threading
import queue
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numba import jit

# ----------------------
# 硬件配置模块
# ----------------------
class DCA1000Config:
    def __init__(self):
        self.ip = "192.168.33.180"
        self.ctrl_port = 4096
        self.data_port = 4098
        self.frame_config = [
            "sensorStop",
            "flushCfg",
            "dfeDataOutputMode 1",
            "channelCfg 15 3 0",
            "adcCfg 2 1",
            "adcbufCfg -1 0 1 1 1",
            "profileCfg 0 77 100 7 57.14 0 0 70 1 256 5209 0 0 30",
            "frameCfg 0 1 16 0 100 1 0"
        ]

    def configure_radar(self):
        ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for cmd in self.frame_config:
            ctrl_sock.sendto(f"{cmd}\n".encode(), (self.ip, self.ctrl_port))
        ctrl_sock.sendto(b"sensorStart\n", (self.ip, self.ctrl_port))
        ctrl_sock.close()

# ----------------------
# 数据采集模块
# ----------------------
class RadarDataCollector:
    def __init__(self, buffer_size=1000):
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.config = DCA1000Config()
        
    def _data_receiver(self):
        data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_sock.bind(("0.0.0.0", self.config.data_port))
        
        while self.running:
            packet, _ = data_sock.recvfrom(1500)
            if self.buffer.full():
                self.buffer.get()  # 丢弃旧数据
            self.buffer.put(packet)
        
    def start(self):
        self.config.configure_radar()
        self.running = True
        self.thread = threading.Thread(target=self._data_receiver)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

# ----------------------
# 信号处理模块
# ----------------------
class SignalProcessor:
    def __init__(self, frame_length=256, num_chirps=128):
        self.frame_length = frame_length
        self.num_chirps = num_chirps
        self.hanning = np.hanning(frame_length)
        
    @jit(nopython=True)
    def parse_packet(self, packet):
        adc_data = np.frombuffer(packet[40:], dtype=np.int16)
        return adc_data.astype(np.float32).view(np.complex64).reshape(-1, 4)
    
    def range_fft(self, frame):
        return np.fft.fftshift(np.fft.fft(frame * self.hanning[:, None], axis=0))
    
    def doppler_fft(self, range_data):
        return np.fft.fftshift(np.fft.fft(range_data * self.hanning[None, :], axis=1), axes=1)

# ----------------------
# 动作检测模型
# ----------------------
class BoundaryDetector(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.lstm = nn.LSTM(16, 32, bidirectional=True)
        self.fc = nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        return torch.sigmoid(x[-1])

# ----------------------
# 实时处理引擎
# ----------------------
class GestureRecognizer:
    def __init__(self):
        self.collector = RadarDataCollector()
        self.processor = SignalProcessor()
        self.model = self.load_model()
        self.buffer = []
        self.window_size = 30  # 300ms窗口（假设100fps）
        
        # 多普勒触发参数
        self.entropy_threshold = 0.4
        self.min_gesture_length = 10  # 100ms
        
    def load_model(self, model_path="boundary_model.pth"):
        model = BoundaryDetector()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    def compute_entropy(self, spectrum):
        prob = spectrum / (np.sum(spectrum) + 1e-9)
        return -np.sum(prob * np.log(prob + 1e-9))
    
    def realtime_processing(self):
        self.collector.start()
        current_state = "idle"
        gesture_buffer = []
        num_frames = 20  # 手势收集状态时额外收集的帧数

        try:
            while True:
                if not self.collector.buffer.empty():
                    packet = self.collector.buffer.get()
                    frame = self.processor.parse_packet(packet)

                    # 特征提取
                    range_fft = self.processor.range_fft(frame)
                    doppler_fft = self.processor.doppler_fft(range_fft)
                    spectrum = np.mean(np.abs(doppler_fft), axis=0)

                    # 初级触发检测
                    self.buffer.append(spectrum)
                    if len(self.buffer) > self.window_size:
                        self.buffer.pop(0)

                        # 计算熵变化
                        current_entropy = self.compute_entropy(np.array(self.buffer))
                        if len(self.buffer) == self.window_size:
                            prev_entropy = self.compute_entropy(self.buffer[:-1])
                            delta = current_entropy - prev_entropy

                            if delta > self.entropy_threshold and current_state == "idle":
                                print("Trigger detected!")
                                current_state = "detecting"
                                gesture_buffer = self.buffer.copy()

                            elif current_state == "detecting":
                                gesture_buffer.append(spectrum)

                                # 转换为模型输入
                                input_data = np.array(gesture_buffer)[-self.window_size:]
                                input_tensor = torch.FloatTensor(input_data).permute(1, 0).unsqueeze(0)

                                # 模型预测
                                with torch.no_grad():
                                    pred = self.model(input_tensor).numpy()[0]

                                # 状态转换逻辑
                                if pred[0] > 0.7 and pred[1] < 0.3:
                                    current_state = "gesturing"
                                elif pred[1] > 0.6 and current_state == "gesturing":
                                    print(f"Gesture detected! Duration: {len(gesture_buffer)} frames")

                                    # 继续收集 num_frames 帧数据
                                    for _ in range(num_frames):
                                        if not self.collector.buffer.empty():
                                            packet = self.collector.buffer.get()
                                            gesture_buffer.append(packet)
                                            #继续数据处理。。。。

                                    # 调用手势分类逻辑
                                    self.classify_gesture(gesture_buffer)

                                    # 重置状态
                                    current_state = "idle"
                                    gesture_buffer = []

        except KeyboardInterrupt:
            self.collector.stop()

def classify_gesture(self, data):
    """
    手势分类逻辑。。。。。
    """
    print("Performing gesture classification...")
    # 示例可视化
    plt.specgram(np.array(data).mean(axis=1), Fs=100)
    plt.title("Gesture Spectrogram")
    plt.show()

# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    recognizer = GestureRecognizer()
    recognizer.realtime_processing()