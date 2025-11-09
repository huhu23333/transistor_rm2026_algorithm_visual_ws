// SharedMemoryYOLOPose.cpp
#include "communication/SharedMemoryYOLOPose.h"

SharedMemoryYOLOPose::SharedMemoryYOLOPose(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node) : node(node) {
    YOLO_POSE_SHM_KEY = (*config_file_ptr)["YOLO_POSE_SHM_KEY"].as<int>();
    
    // 创建或附加共享内存
    size_t shm_size = sizeof(SharedData);
    shm_id_ = shmget(YOLO_POSE_SHM_KEY, shm_size, IPC_CREAT | 0666);
    if (shm_id_ == -1) {
        throw std::runtime_error("Failed to create shared memory for YOLO Pose");
    }
    
    attachSharedMemory();
    
    // 初始化共享内存
    shared_data_->is_processed = true;  // 初始状态为已处理
    shared_data_->reserved1 = false;
    shared_data_->reserved2 = false;
    shared_data_->reserved3 = false;
    shared_data_->reserved4 = 0;
    shared_data_->return_data.num_detections = 0;
}

SharedMemoryYOLOPose::~SharedMemoryYOLOPose() {
    detachSharedMemory();
}

void SharedMemoryYOLOPose::attachSharedMemory() {
    shared_data_ = static_cast<SharedData*>(shmat(shm_id_, nullptr, 0));
    if (shared_data_ == reinterpret_cast<void*>(-1)) {
        throw std::runtime_error("Failed to attach shared memory");
    }
}

void SharedMemoryYOLOPose::detachSharedMemory() {
    if (shmdt(shared_data_) == -1) {
        // 记录错误但不抛出异常
    }
}

void SharedMemoryYOLOPose::waitForProcessing() {
    while (!shared_data_->is_processed) {
        usleep(1000); // 1ms休眠
    }
}

cv::Mat SharedMemoryYOLOPose::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    
    // 调整尺寸到640x640
    if (image.size() != cv::Size(640, 640)) {
        cv::resize(image, processed, cv::Size(640, 640));
    } else {
        processed = image.clone();
    }
    
    // 确保图像是RGB格式
    if (processed.channels() == 1) {
        cv::cvtColor(processed, processed, cv::COLOR_GRAY2BGR);
    } else if (processed.channels() == 4) {
        cv::cvtColor(processed, processed, cv::COLOR_BGRA2BGR);
    }
    
    return processed;
}

std::vector<DetectionResult> SharedMemoryYOLOPose::processImage(const cv::Mat& image, bool block, int now_history_frame_identifier) {
    // 1. 预处理图像
    cv::Mat processed_image = preprocessImage(image);
    
    // 验证图像格式和尺寸
    if (processed_image.cols != 640 || processed_image.rows != 640 || processed_image.channels() != 3) {
        throw std::invalid_argument("Image must be 640x640 RGB format after preprocessing");
    }
    
    // 2. 复制图像数据到共享内存
    size_t data_size = 640 * 640 * 3;
    if (processed_image.isContinuous()) {
        std::memcpy(shared_data_->image_data, processed_image.data, data_size);
    } else {
        cv::Mat continuous_img = processed_image.clone();
        std::memcpy(shared_data_->image_data, continuous_img.data, data_size);
    }
    
    shared_data_->input_history_frame_identifier = now_history_frame_identifier;

    // 3. 重置处理状态和检测数量
    shared_data_->is_processed = false;
    //shared_data_->return_data.num_detections = 0;
    
    // 4. 等待Python端处理完成
    if (block) {
        waitForProcessing();
    }
    // 5. 读取处理结果
    std::vector<DetectionResult> results;
    int num_detections = shared_data_->return_data.num_detections;
    
    for (int i = 0; i < num_detections && i < 50; ++i) {
        DetectionResult det;
        std::memcpy(det.keypoints, shared_data_->return_data.results[i].keypoints, 8 * sizeof(float));
        det.confidence = shared_data_->return_data.results[i].confidence;
        det.class_id = shared_data_->return_data.results[i].class_id;
        det.history_frame_identifier = shared_data_->return_data.return_history_frame_identifier;
        RCLCPP_DEBUG(node->get_logger(), "return_history_frame_identifier: %d", shared_data_->return_data.return_history_frame_identifier);
        results.push_back(det);
    }

    RCLCPP_INFO(node->get_logger(), "yolo_data_num: %d", num_detections);
    
    return results;
}