import cv2
from ultralytics import YOLO
import numpy as np
import time

def process_and_save_video(model_path, input_video_path, output_video_path, conf_threshold=0.5):
    """
    使用YOLO Seg模型处理视频并保存分割结果
    
    参数:
    model_path: YOLO模型路径
    input_video_path: 输入视频路径
    output_video_path: 输出视频路径
    conf_threshold: 置信度阈值
    """
    # 加载YOLO模型
    model = YOLO(model_path)
    
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return False
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 定义视频编码器和输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"开始处理视频，总帧数: {total_frames}")
    
    while True:
        #time.sleep(0.5)
        ret, frame = cap.read()
        if not ret:
            break
        
        #cv2.imshow("frame read", frame)
        #key = cv2.waitKey(1)

        # 使用模型进行预测
        results = model(frame, conf=conf_threshold, device="intel:gpu")
        
        # 处理预测结果
        annotated_frame = results[0].plot()  # 绘制检测框和分割掩码
        
        # 写入输出视频
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧")
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"视频处理完成，结果已保存至: {output_video_path}")
    return True

# 使用示例
if __name__ == "__main__":
    # 设置路径和参数
    #model_path = "armor-oneclass-yolo11n-pose-best.pt"  # YOLO分割模型路径
    #model_path = "armor-oneclass-yolo11n-pose-best_openvino_model"
    model_path = "armor-oneclass-yolo11n-pose-best_int8_openvino_model"
    input_video = "outpost1_8m.mp4"  # 输入视频路径
    output_video = "output_video.mp4"  # 输出视频路径
    
    # 处理并保存视频
    t_start = time.time()
    success = process_and_save_video(model_path, input_video, output_video)
    t_end = time.time()
    print(f"Total Time: {t_end-t_start:.1f}s")
    
    if success:
        print("处理成功!")
    else:
        print("处理失败!") 
