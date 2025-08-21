import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import numpy as np
from sysv_ipc import SharedMemory, IPC_CREAT
import time
import cv2
import os
import yaml


class TransistorRM2026Net(nn.Module):
    def __init__(self, num_classes=8):
        super(TransistorRM2026Net, self).__init__()

        self.activate = nn.ReLU()
        self.pooling = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.conv1x1 = nn.Conv2d(256, 512, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(256)
        
        self.bn1x1 = nn.BatchNorm2d(512)

        self.dropout2d_1 = nn.Dropout2d(0.4)
        self.dropout2d_2 = nn.Dropout2d(0.3)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 512)

        self.head1 = nn.Linear(512, 1)
        self.head2 = nn.Linear(512, 1)
        self.head3 = nn.Linear(512, 1)
        self.head4 = nn.Linear(512, 1)
        self.head5 = nn.Linear(512, num_classes)
        
    def forward(self, x):                                   #64*48*3
        x = self.activate(self.bn1(self.conv1(x)))          #64*48*32
        x = self.pooling(x)                                 #32*24*32
        x = self.activate(self.bn2(self.conv2(x)))          #32*24*64
        x = self.pooling(x)                                 #16*12*64
        x = self.activate(self.bn3(self.conv3(x)))          #16*12*128
        x = self.pooling(x)                                 #8*6*128
        x = self.activate(self.bn4(self.conv4(x)))          #8*6*128
        x = self.dropout2d_1(x)                             #8*6*128
        x = self.activate(self.bn5(self.conv5(x)))          #8*6*256
        x = self.pooling(x)                                 #4*3*256
        x = self.activate(self.bn6(self.conv6(x)))          #4*3*256
        x = self.dropout2d_2(x)                             #4*3*256
        x = self.activate(self.bn7(self.conv7(x)))          #4*3*256
        x = self.activate(self.bn1x1(self.conv1x1(x)))      #4*3*512
        x = self.gap(x).squeeze(-1).squeeze(-1)             #512
        x = self.activate(self.fc(x))                       #512
        x = self.dropout(x)                                 #512
        result1 = self.head1(x)                             #1
        result2 = self.head2(x)                             #1
        result3 = self.head3(x)                             #1
        result4 = self.head4(x)                             #1
        result5 = self.head5(x)                             #num_classes
        return (result1, result2, result3, result4, result5)


class ShmPytorchProcessorNode(Node):
    def __init__(self, name):
        super().__init__(name)

        # 获取路径及配置文件内容
        script_file_path = os.path.abspath(__file__)
        self.get_logger().info(f"info from Python | Path: {script_file_path}")
        ws_dir_name = "transistor_rm2026_algorithm_visual_ws"
        ws_dir_path = script_file_path[:script_file_path.find(ws_dir_name)+len(ws_dir_name)]

        config_file_relatvie_path = "src/shared_files/config.yaml"
        config_file_path = os.path.join(ws_dir_path, config_file_relatvie_path)
        with open(config_file_path, 'r', encoding='utf-8') as config_file:
            config_data = yaml.safe_load(config_file)

        self.SHM_KEY = int(config_data["SHM_KEY"])
        model_relative_path = config_data["model_relative_path"]
        model_path = os.path.join(ws_dir_path, model_relative_path)
        
        # 定义与C++完全一致的内存结构
        self.shm = None
        self.MAX_IMAGES = 100
        self.attach_shared_memory()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransistorRM2026Net()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.run()
    
    def __del__(self):
        if self.shm:
            self.shm.detach()
    
    def attach_shared_memory(self):
        try:
            self.shm = SharedMemory(self.SHM_KEY, flags=0, size=0)
        except:
            # 首次运行时创建共享内存
            shm_size = (
                4  # num_images (int32)
                + 4  # is_processed + reserved (4*bool=4字节)
                + 100 * 12 * 4  # results (100*12*float32)
                + 100 * 64 * 48 * 3  # images (100*64*48*3 uint8)
            )
            self.shm = SharedMemory(self.SHM_KEY, IPC_CREAT, size=shm_size)

    def run(self):
        """持续监控共享内存并处理图像，添加可视化功能"""
        with torch.no_grad():
            while True:
                # 读取控制信息
                control_data = self.shm.read(8, offset=0)
                num_images = np.frombuffer(control_data[:4], dtype=np.int32)[0]
                is_processed = control_data[4]  # 第5字节是is_processed
                
                if not is_processed and num_images > 0:
                    t_start = time.time()

                    # 1. 准备数据
                    num_images = min(num_images, self.MAX_IMAGES)
                    
                    # 2. 读取图像数据
                    img_offset = 8 + 100*12*4  # 控制信息+结果区
                    img_data = self.shm.read(
                        num_images * 64*48*3, 
                        offset=img_offset
                    )
                    images = np.frombuffer(img_data, dtype=np.uint8)

                    # 初始化图像列表
                    image_list = []

                    
                    
                    # 处理每张图像
                    for i in range(num_images):
                        # 提取当前图像数据 (64宽x48高x3通道)
                        img_start = i * 64 * 48 * 3
                        img_end = img_start + 64 * 48 * 3
                        img_flat = images[img_start:img_end]
                        
                        # 转换为OpenCV图像格式 (高度, 宽度, 通道)
                        img_np = img_flat.reshape((48, 64, 3))  # 注意: 高度48，宽度64 #(48, 64, 3)

                        # 将图像加入图像列表
                        image_list.append(img_np)


                    # ============== 可视化部分 ==============
                    # ============== 可视化部分 ==============
                    # 创建窗口用于显示图像
                    #cv2.namedWindow("Shared Memory Image", cv2.WINDOW_NORMAL)
                    #cv2.resizeWindow("Shared Memory Image", 320, 240)  # 放大显示
                    #cv2.imshow("Shared Memory Image", image_list[0])  # 显示图像
                    #key = cv2.waitKey(1)  # 图像显示1ms  # 等待按键或短暂延迟

                    
                    t_1 = time.time()

                    # 将所有图像叠加为一个tensor，并行处理
                    images_np = np.stack(image_list, axis=0)
                    images_tensor = torch.from_numpy(images_np).float().to(self.device)
                    images_tensor = images_tensor.permute(0, 3, 1, 2) # 调整维度顺序 (N, H, W, C) → (N, C, H, W)
                    images_tensor = (images_tensor / 127.5) - 1.0 # 归一化到[-1, 1]

                    t_2 = time.time()
                    
                    # 使用PyTorch处理
                    results_tuple = self.model(images_tensor)

                    t_3 = time.time()

                    results1 = torch.sigmoid(results_tuple[0]).cpu()
                    results2 = torch.sigmoid(results_tuple[1]).cpu()
                    results3 = torch.sigmoid(results_tuple[2]).cpu()
                    results4 = torch.sigmoid(results_tuple[3]).cpu()
                    results5 = torch.softmax(results_tuple[4], dim=1).cpu()
                    results_np = np.zeros(12 * num_images)
                    for i in range(num_images):
                        results_np[12*i] = results1[i].numpy()
                        results_np[12*i+1] = results2[i].numpy()
                        results_np[12*i+2] = results3[i].numpy()
                        results_np[12*i+3] = results4[i].numpy()
                        results_np[12*i+4 : 12*i+12] = results5[i].numpy()

                    t_4 = time.time()

                    # 4. 写入结果
                    results_bytes = results_np.astype(np.float32).tobytes()
                    self.shm.write(results_bytes, offset=8)  # 控制信息后紧跟结果区
                    
                    # 5. 设置处理完成标志
                    processed_flag = bytearray([1])
                    self.shm.write(processed_flag, offset=4)

                    t_5 = time.time()

                    self.get_logger().debug(f"python process time: {(time.time()-t_start)*1000:.2f}ms\n{(t_1-t_start)*1000:.2f} | {(t_2-t_1)*1000:.2f} | {(t_3-t_2)*1000:.2f} | {(t_4-t_3)*1000:.2f} | {(t_5-t_4)*1000:.2f}")

                time.sleep(0.001)  # 减少CPU占用
                

def main(args=None):
    rclpy.init(args=args)
    node = ShmPytorchProcessorNode("shm_pytorch_processor_node")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
