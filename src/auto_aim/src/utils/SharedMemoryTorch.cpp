// SharedMemoryTorch.cpp
#include "utils/SharedMemoryTorch.h"

SharedMemoryTorch::SharedMemoryTorch(std::shared_ptr<YAML::Node> config_file_ptr) {
    SHM_KEY = (*config_file_ptr)["SHM_KEY"].as<int>();
    // 创建或附加共享内存
    size_t shm_size = sizeof(SharedData);
    shm_id_ = shmget(SHM_KEY, shm_size, IPC_CREAT | 0666);
    if (shm_id_ == -1) {
        throw std::runtime_error("Failed to create shared memory");
    }
    
    attachSharedMemory();
}

SharedMemoryTorch::~SharedMemoryTorch() {
    detachSharedMemory();
}

void SharedMemoryTorch::attachSharedMemory() {
    shared_data_ = static_cast<SharedData*>(shmat(shm_id_, nullptr, 0));
    if (shared_data_ == reinterpret_cast<void*>(-1)) {
        throw std::runtime_error("Failed to attach shared memory");
    }
}

void SharedMemoryTorch::detachSharedMemory() {
    if (shmdt(shared_data_) == -1) {
        // 处理错误但不抛出异常，防止在析构中产生问题
    }
}

void SharedMemoryTorch::waitForProcessing() {
    while (!shared_data_->is_processed) {
        usleep(1000); // 1ms休眠减少CPU占用
    }
}

std::vector<std::vector<float>> SharedMemoryTorch::processImages(const std::vector<cv::Mat>& images) {
    // 1. 准备数据
    int num_images = std::min(images.size(), MAX_IMAGES);
    shared_data_->num_images = num_images;
    
    // 2. 复制图像数据到共享内存
    for (int i = 0; i < num_images; ++i) {
        // 验证图像格式和尺寸
        if (images[i].cols != 64 || images[i].rows != 48 || images[i].channels() != 3) {
            throw std::invalid_argument("Invalid image format. Expected 64x48 RGB");
        }
        
        // 复制连续的内存块
        size_t data_size = 64 * 48 * 3;
        if (images[i].isContinuous()) {
            std::memcpy(shared_data_->image_data[i], images[i].data, data_size);
        } else {
            cv::Mat continuous_img = images[i].clone();
            std::memcpy(shared_data_->image_data[i], continuous_img.data, data_size);
        }
    }
    
    // 3. 通知Python端开始处理
    shared_data_->is_processed = false;
    
    // 4. 等待处理完成
    waitForProcessing();
    
    // 5. 读取处理结果
    std::vector<std::vector<float>> results;
    for (int i = 0; i < num_images; ++i) {
        results.emplace_back(
            shared_data_->results[i],
            shared_data_->results[i] + 12
        );
    }
    
    return results;
}