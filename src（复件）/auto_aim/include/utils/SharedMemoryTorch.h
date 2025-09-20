// SharedMemoryTorch.h
#ifndef SHARED_MEMORY_TORCH_H
#define SHARED_MEMORY_TORCH_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <cstring>
#include <yaml-cpp/yaml.h>

class SharedMemoryTorch {
public:
    SharedMemoryTorch(std::shared_ptr<YAML::Node> config_file_ptr);
    ~SharedMemoryTorch();
    
    // 向共享内存写入图像并等待处理结果
    std::vector<std::vector<float>> processImages(const std::vector<cv::Mat>& images);

private:
    // 共享内存数据结构（必须与Python端完全匹配）
    #pragma pack(push, 1)
    struct SharedData {
        int num_images;         // 当前批次图像数量
        bool is_processed;      // 处理状态标志
        bool reserved1;         // 备用标志1
        bool reserved2;         // 备用标志2
        bool reserved3;         // 备用标志3
        
        float results[100][12]; // 结果存储区 (100x12 float)
        unsigned char image_data[100][64*48*3]; // 图像存储区 (100张64x48 RGB图像)
    };
    #pragma pack(pop)
    
    int shm_id_;
    SharedData* shared_data_;
    const size_t MAX_IMAGES = 100;
    int SHM_KEY; // 共享内存键值
    
    void attachSharedMemory();
    void detachSharedMemory();
    void waitForProcessing();
};

#endif // SHARED_MEMORY_TORCH_H