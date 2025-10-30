import rclpy
from rclpy.node import Node
import torch
import numpy as np
from sysv_ipc import SharedMemory, IPC_CREAT
import time
import cv2
import os
import yaml
from ultralytics import YOLO

class FrameRateCounter:
    def __init__(self, window_size = 30):
        self.window_size = window_size
        self.tick_time_history = []
    def tick(self):
        self.tick_time_history.append(time.time())
        if len(self.tick_time_history) > self.window_size:
            self.tick_time_history = self.tick_time_history[-self.window_size:]
    def avg_frame_time(self):
        if len(self.tick_time_history) <= 1:
            return 0.0
        else:
            return (self.tick_time_history[-1] - self.tick_time_history[0]) / (len(self.tick_time_history) - 1)
    def fps(self):
        avg_frame_time = self.avg_frame_time()
        if avg_frame_time == 0.0:
            return 0.0
        else:
            return 1 / avg_frame_time

class ShmYOLOPoseProcessorNode(Node):
    def __init__(self, name):
        super().__init__(name)

        # 获取路径及配置文件内容
        script_file_path = os.path.abspath(__file__)
        self.get_logger().info(f"Python YOLO Pose Processor | Path: {script_file_path}")
        
        ws_dir_name = "transistor_rm2026_algorithm_visual_ws"  # 修改为您的workspace名称
        ws_dir_path = script_file_path[:script_file_path.find(ws_dir_name) + len(ws_dir_name)]

        config_file_relative_path = "src/shared_files/config.yaml"
        config_file_path = os.path.join(ws_dir_path, config_file_relative_path)
        
        with open(config_file_path, 'r', encoding='utf-8') as config_file:
            config_data = yaml.safe_load(config_file)

        self.YOLO_POSE_SHM_KEY = int(config_data["YOLO_POSE_SHM_KEY"])
        yolo_pose_model_relative_path = config_data["yolo_pose_model_relative_path"]
        model_path = os.path.join(ws_dir_path, yolo_pose_model_relative_path)
        
        # 定义与C++完全一致的内存结构
        self.shm = None
        self.MAX_DETECTIONS = 50
        
        # 计算各个区域的偏移量
        self.CONTROL_DATA_SIZE = 8  # is_processed(1) + reserved1(1) + reserved2(1) + reserved3(1) + reserved4(4)
        self.IMAGE_DATA_SIZE = 640 * 640 * 3
        self.INPUT_DATA_SIZE = self.IMAGE_DATA_SIZE + 4
        self.RETURN_DATA_HEADER_SIZE = 8  # num_detections(4)
        self.PER_DETECTION_SIZE = 8 * 4 + 4 + 4 + 4  # keypoints(8*float32) + confidence(4) + class_id(4) + reserved(4)
        self.TOTAL_RETURN_DATA_SIZE = self.RETURN_DATA_HEADER_SIZE + self.MAX_DETECTIONS * self.PER_DETECTION_SIZE
        
        self.IMAGE_DATA_OFFSET = self.CONTROL_DATA_SIZE
        self.RETURN_DATA_OFFSET = self.CONTROL_DATA_SIZE + self.INPUT_DATA_SIZE
        
        self.attach_shared_memory()

        # 加载YOLO姿态估计模型
        self.model = YOLO(model_path)
        
        self.get_logger().info("YOLO Pose model loaded successfully")

        self.frame_rate_counter = FrameRateCounter()

        self.run()
    
    def __del__(self):
        if self.shm:
            self.shm.detach()
    
    def attach_shared_memory(self):
        try:
            total_size = self.CONTROL_DATA_SIZE + self.INPUT_DATA_SIZE + self.TOTAL_RETURN_DATA_SIZE
            self.shm = SharedMemory(self.YOLO_POSE_SHM_KEY, flags=0, size=0)
            # 验证共享内存大小是否足够
            if self.shm.size < total_size:
                self.get_logger().error(f"Shared memory too small: {self.shm.size} < {total_size}")
                raise ValueError("Shared memory size mismatch")
        except:
            # 首次运行时创建共享内存
            total_size = self.CONTROL_DATA_SIZE + self.INPUT_DATA_SIZE + self.TOTAL_RETURN_DATA_SIZE
            self.shm = SharedMemory(self.YOLO_POSE_SHM_KEY, IPC_CREAT, size=total_size)
            self.get_logger().info(f"Shared memory created successfully, size: {total_size}")

    def run(self):
        """持续监控共享内存并处理图像"""
        while True:
            try:
                # 读取控制信息 - 只检查is_processed标志
                control_data = self.shm.read(1, offset=0)  # 只读取第一个字节(is_processed)
                is_processed = control_data[0]
                
                if not is_processed:
                    t_start = time.time()

                    # 1. 读取图像数据
                    img_data = self.shm.read(self.IMAGE_DATA_SIZE, offset=self.IMAGE_DATA_OFFSET)
                    
                    # 转换为numpy数组并reshape为图像
                    image_np = np.frombuffer(img_data, dtype=np.uint8).reshape((640, 640, 3))

                    history_frame_identifier = np.frombuffer(self.shm.read(4, offset=self.IMAGE_DATA_OFFSET + self.IMAGE_DATA_SIZE), dtype=np.int32)
                    
                    # 2. 使用YOLO进行姿态估计
                    results = self.model(image_np, verbose=False)
                    
                    # 3. 处理检测结果
                    all_detections = []
                    
                    for result in results:
                        if result.keypoints is not None and len(result.keypoints) > 0:
                            # 获取关键点数据
                            keypoints = result.keypoints.data.cpu().numpy()
                            boxes = result.boxes.data.cpu().numpy()
                            
                            for i, (box, kpt) in enumerate(zip(boxes, keypoints)):
                                if i >= self.MAX_DETECTIONS:
                                    break
                                
                                confidence = box[4]
                                class_id = int(box[5])
                                
                                # 提取4个关键点的归一化坐标
                                if kpt.shape[0] >= 4:
                                    # 取前4个关键点
                                    selected_keypoints = kpt[:4]
                                    normalized_keypoints = []
                                    
                                    for j in range(4):
                                        # 归一化到[0,1]范围
                                        x_norm = selected_keypoints[j, 0] / 640.0
                                        y_norm = selected_keypoints[j, 1] / 640.0
                                        normalized_keypoints.extend([x_norm, y_norm])
                                    
                                    # 如果关键点数量不足4个，用0填充
                                    while len(normalized_keypoints) < 8:
                                        normalized_keypoints.extend([0.0, 0.0])
                                    
                                    detection_data = {
                                        'keypoints': normalized_keypoints[:8],  # 确保正好8个值
                                        'confidence': float(confidence),
                                        'class_id': class_id
                                    }
                                    all_detections.append(detection_data)

                    
                    # ============== 可视化部分 ==============
                    # 创建窗口用于显示图像
                    cv2.namedWindow("YOLO Pose Image", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("YOLO Pose Image", 640, 640)
                    cv2.imshow("YOLO Pose Image", results[0].plot())
                    key = cv2.waitKey(1)
                    
                    # 4. 准备写入返回数据
                    num_detections = len(all_detections)
                    
                    # 写入检测数量到返回数据区开头
                    num_detections_bytes = np.int32(num_detections).tobytes()
                    self.shm.write(num_detections_bytes, offset=self.RETURN_DATA_OFFSET)
                    self.get_logger().debug(f"python history_frame_identifier: {history_frame_identifier}")
                    self.shm.write(history_frame_identifier.tobytes(), offset=self.RETURN_DATA_OFFSET + 4)
                    
                    # 写入每个检测结果
                    results_offset = self.RETURN_DATA_OFFSET + self.RETURN_DATA_HEADER_SIZE
                    for i, detection in enumerate(all_detections):
                        if i >= self.MAX_DETECTIONS:
                            break
                            
                        # 写入关键点坐标 (8个float32)
                        keypoints_bytes = np.array(detection['keypoints'], dtype=np.float32).tobytes()
                        self.shm.write(keypoints_bytes, offset=results_offset)
                        results_offset += 8 * 4
                        
                        # 写入置信度 (1个float32)
                        conf_bytes = np.float32(detection['confidence']).tobytes()
                        self.shm.write(conf_bytes, offset=results_offset)
                        results_offset += 4
                        
                        # 写入类别ID (1个int32)
                        class_bytes = np.int32(detection['class_id']).tobytes()
                        self.shm.write(class_bytes, offset=results_offset)
                        results_offset += 4
                        
                        # 写入保留字段 (1个int32)
                        reserved_bytes = np.int32(0).tobytes()
                        self.shm.write(reserved_bytes, offset=results_offset)
                        results_offset += 4
                    
                    # 5. 设置处理完成标志
                    processed_flag = bytearray([1])
                    self.shm.write(processed_flag, offset=0)

                    processing_time = (time.time() - t_start) * 1000
                    self.get_logger().debug(f"YOLO Pose processing time: {processing_time:.2f}ms, Detections: {num_detections}")

                    self.frame_rate_counter.tick()
                    self.get_logger().info(f"YOLO Pose processing frame rate: {self.frame_rate_counter.fps():.2f} fps")

                time.sleep(0.001)  # 减少CPU占用
            
            except Exception as e:
                self.get_logger().error(f"Error in processing loop: {str(e)}")
                time.sleep(0.1)  # 出错时稍作等待


def main(args=None):
    rclpy.init(args=args)
    node = ShmYOLOPoseProcessorNode("shm_yolo_pose_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()