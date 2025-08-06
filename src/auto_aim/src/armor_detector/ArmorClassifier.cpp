//ArmorCLassifier.cpp
#include "armor_detector/ArmorClassifier.h"
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

ArmorClassifier::ArmorClassifier(const std::string& model_path, bool use_cuda) 
    : device(use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    try {
        model = std::make_shared<NumberNet>();
        torch::load(model, model_path);
        model->to(device);
        model->eval();
        
        std::cout << "Model loaded successfully to " 
                  << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        throw;
    }
}

cv::Mat ArmorClassifier::preprocessROI(const cv::Mat& img, const cv::Rect& roi) {
    static int save_count = 0;
    const int MAX_SAVE_COUNT = 2000;  // 最大保存数量
    cv::Mat normalized;  // 将声明移到函数开始
    
    // 如果已经保存了1000张图片，直接返回处理后的图像而不保存
    if (save_count >= MAX_SAVE_COUNT) {
        cv::Rect safe_roi = roi & cv::Rect(0, 0, img.cols, img.rows);
        if (safe_roi.area() == 0) {
            return cv::Mat();
        }
        
        // 扩大ROI区域
        const float MARGIN_RATIO = 0.1f;
        int margin_w = static_cast<int>(safe_roi.width * MARGIN_RATIO);
        int margin_h = static_cast<int>(safe_roi.height * MARGIN_RATIO);
        
        safe_roi.x = std::max(0, safe_roi.x - margin_w);
        safe_roi.y = std::max(0, safe_roi.y - margin_h);
        safe_roi.width = std::min(img.cols - safe_roi.x, safe_roi.width + 2 * margin_w);
        safe_roi.height = std::min(img.rows - safe_roi.y, safe_roi.height + 2 * margin_h);
        
        // 提取ROI
        cv::Mat roi_img = img(safe_roi);
        
        // 图像预处理
        cv::Mat gray;
        cv::cvtColor(roi_img, gray, cv::COLOR_BGR2GRAY);
        
        cv::Mat enhanced;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(3, 3));
        clahe->apply(gray, enhanced);
        
        cv::Mat blurred;
        cv::GaussianBlur(enhanced, blurred, cv::Size(3, 3), 0);
        
        cv::Mat padded;
        int padding = 2;
        cv::copyMakeBorder(blurred, padded, padding, padding, padding, padding, 
                          cv::BORDER_REPLICATE);
        
        cv::Mat resized;
        cv::resize(padded, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
        
        resized.convertTo(normalized, CV_32F, 1.0/255.0);
        normalized = (normalized - 0.5f) / 0.5f;
        
        return normalized;
    }
    
    cv::Rect safe_roi = roi & cv::Rect(0, 0, img.cols, img.rows);
    if (safe_roi.area() == 0) {
        return cv::Mat();
    }
    
    // 扩大ROI区域
    const float MARGIN_RATIO = 0.1f;
    int margin_w = static_cast<int>(safe_roi.width * MARGIN_RATIO);
    int margin_h = static_cast<int>(safe_roi.height * MARGIN_RATIO);
    
    safe_roi.x = std::max(0, safe_roi.x - margin_w);
    safe_roi.y = std::max(0, safe_roi.y - margin_h);
    safe_roi.width = std::min(img.cols - safe_roi.x, safe_roi.width + 2 * margin_w);
    safe_roi.height = std::min(img.rows - safe_roi.y, safe_roi.height + 2 * margin_h);
    
    // 提取ROI
    cv::Mat roi_img = img(safe_roi);
    
    // 图像预处理
    cv::Mat gray;
    cv::cvtColor(roi_img, gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat enhanced;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(3, 3));
    clahe->apply(gray, enhanced);
    
    cv::Mat blurred;
    cv::GaussianBlur(enhanced, blurred, cv::Size(3, 3), 0);
    
    cv::Mat padded;
    int padding = 2;
    cv::copyMakeBorder(blurred, padded, padding, padding, padding, padding, 
                      cv::BORDER_REPLICATE);
    
    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    
    resized.convertTo(normalized, CV_32F, 1.0/255.0);
    normalized = (normalized - 0.5f) / 0.5f;
    
    // 保存处理后的图像（用于神经网络输入的标准化图像）
    if (save_count < MAX_SAVE_COUNT) {
        // 创建保存目录
        fs::create_directories("network_input_images");
        
        // 生成文件名（0001.jpg 格式）
        std::ostringstream filename;
        filename << "network_input_images/"
                << std::setw(4) << std::setfill('0') << (save_count + 1)
                << ".jpg";
                
        // 将浮点图像转换回8位图像用于保存
        cv::Mat save_img;
        cv::Mat temp = (normalized * 0.5f + 0.5f) * 255.0f;
        temp.convertTo(save_img, CV_8U);
        
        cv::imwrite(filename.str(), save_img);
        save_count++;
        
        if (save_count == MAX_SAVE_COUNT) {
            std::cout << "Reached maximum number of saved images (2000)" << std::endl;
        }
    }
    
    return normalized;
}

bool ArmorClassifier::isNearPreviousCenter(const cv::Point2f& current, 
                                         const cv::Point2f& previous, 
                                         float max_dist) {
    float dist = cv::norm(current - previous);
    return dist <= max_dist;
}

std::vector<ArmorResult> ArmorClassifier::classify(
    const cv::Mat& img, const std::vector<Armor>& armors) {
    
    std::vector<ArmorResult> results;
    torch::NoGradGuard no_grad;
    auto current_time = std::chrono::steady_clock::now();
    
    // 清理过期的跟踪目标
    for (auto it = tracked_armors.begin(); it != tracked_armors.end();) {
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - it->second.last_seen).count();
        if (age > MAX_TRACKING_AGE_MS) {
            it = tracked_armors.erase(it);
        } else {
            ++it;
        }
    }
    
    for (const auto& armor : armors) {
        cv::Mat roi = preprocessROI(img, armor.roi);
        if (roi.empty()) continue;
        
        try {
            torch::Tensor tensor_image = torch::from_blob(
                roi.data, 
                {1, 1, INPUT_HEIGHT, INPUT_WIDTH},
                torch::kFloat32
            ).clone();
            
            tensor_image = tensor_image.to(device);
            
            auto output = model->forward(tensor_image);
            auto probabilities = torch::exp(output);
            
            auto max_result = probabilities.max(1);
            int number = std::get<1>(max_result).item<int>();
            float confidence = std::get<0>(max_result).item<float>();
            
            // 计算当前装甲板中心点
            cv::Point2f current_center = (armor.corners[0] + armor.corners[2]) * 0.5f;
            
            if (number != 0 && confidence >= CONFIDENCE_THRESHOLD) {
                auto& tracked = tracked_armors[number];
                
                // 检查是否与上一次检测位置接近
                if (tracked.tracking_count > 0 && 
                    !isNearPreviousCenter(current_center, tracked.center)) {
                    continue;  // 如果位置跳变太大，忽略此次检测
                }
                
                tracked.number = number;
                tracked.confidence = confidence;
                tracked.last_seen = current_time;
                tracked.tracking_count++;
                tracked.center = current_center;
                
                if (tracked.tracking_count >= MIN_TRACKING_COUNT) {
                    results.emplace_back(armor, number, confidence);
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error in classify: " << e.what() << std::endl;
            continue;
        }
    }
    
    // 更新未检测到的目标
    for (auto& pair : tracked_armors) {
        if (pair.second.last_seen != current_time && pair.second.tracking_count > 0) {
            pair.second.tracking_count = std::max(0, pair.second.tracking_count - 1);
        }
    }
    
    return results;
}
