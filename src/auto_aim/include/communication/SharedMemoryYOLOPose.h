#ifndef SHARED_MEMORY_YOLO_POSE_H
#define SHARED_MEMORY_YOLO_POSE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <cstring>
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#include "macro/AutoAimMacro.h"

struct DetectionResult {
    float keypoints[8];  // 4个关键点，每个关键点(x,y)归一化坐标 (x1,y1,x2,y2,x3,y3,x4,y4)
    float confidence;    // 检测置信度
    int class_id;       // 类别ID
    int history_frame_identifier;
};

class SharedMemoryYOLOPose {
public:
    SharedMemoryYOLOPose(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node);
    ~SharedMemoryYOLOPose();
    
    // 处理单张图像，返回检测结果
    std::vector<DetectionResult> processImage(const cv::Mat& image, bool block, int now_history_frame_identifier);

private:
    rclcpp::Node* node;
    // 共享内存数据结构
    #pragma pack(push, 1)
    struct SharedData {
        // 控制数据区 (8字节，对齐到8字节边界)
        bool is_processed;     // 处理状态标志 (1字节)
        bool show_windows;     // 显示图像
        bool reserved2;        // 备用标志2 (1字节) - 用于对齐  
        bool reserved3;        // 备用标志3 (1字节) - 用于对齐
        int reserved4;         // 备用标志4 (4字节) - 用于填充到8字节
        
        // 传入数据区: 单张640x640 RGB图像
        unsigned char image_data[640 * 640 * 3];
        int input_history_frame_identifier;
        
        // 返回数据区
        struct {
            int num_detections;    // 检测到的目标数量 (4字节)
            int return_history_frame_identifier;
            // 结果存储区：最多50个目标，每个目标包含4个关键点(8个坐标)+置信度+类别ID
            struct {
                float keypoints[8];  // 4个关键点的归一化坐标 (x1,y1,x2,y2,x3,y3,x4,y4)
                float confidence;    // 置信度
                int class_id;        // 类别ID
                int reserved;        // 保留字段用于对齐
            } results[50];
        } return_data;
    };
    #pragma pack(pop)
    
    int shm_id_;
    SharedData* shared_data_;
    int YOLO_POSE_SHM_KEY;
    
    void attachSharedMemory();
    void detachSharedMemory();
    void waitForProcessing();
    cv::Mat preprocessImage(const cv::Mat& image);
};

#endif // SHARED_MEMORY_YOLO_POSE_H