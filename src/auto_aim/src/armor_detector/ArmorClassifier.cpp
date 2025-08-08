// ArmorCLassifier.cpp
#include "armor_detector/ArmorClassifier.h"

/* #include <iostream>
#include <sstream>
#include <string>
// DEBUG */

namespace fs = std::filesystem;

ArmorClassifier::ArmorClassifier(const std::string& model_path, bool use_cuda, rclcpp::Node* node) 
    : device(use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU), node(node) {
    try {
        model = std::make_shared<TransistorRM2026Net>(8);
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

cv::Mat ArmorClassifier::preprocessROI(const cv::Mat& img, const Armor& armor) {
    static int save_count = 0;
    const int MAX_SAVE_COUNT = 2000;  // 最大保存数量
    cv::Mat normalized;  // 将声明移到函数开始

    // 提取ROI
    cv::Mat roi_img = UnwarpUtils::unwarpQuadrilateral(img, armor.corners_expanded);
    
    
    // 图像预处理
    cv::Mat blurred;
    cv::GaussianBlur(roi_img, blurred, cv::Size(3, 3), 0);
    
    cv::Mat padded;
    int padding = 2;
    cv::copyMakeBorder(blurred, padded, padding, padding, padding, padding, 
                      cv::BORDER_REPLICATE);

    cv::imshow("Classifier DEBUG", padded);
    
    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    
    resized.convertTo(normalized, CV_32F, 1.0/255.0);
    normalized = (normalized - 0.5f) / 0.5f;
    
    // 如果已经保存了1000张图片，直接返回处理后的图像而不保存
    if (save_count >= MAX_SAVE_COUNT) {
        return normalized;
    }
    
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
    
    
    // 更新所有目标并清理过期的跟踪目标
    for (size_t i = 0; i < tracked_armors.size(); ++i) {
        tracked_armors[i].is_tracked_now = false;
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - tracked_armors[i].last_seen).count();
        if (age > MAX_TRACKING_AGE_MS) {
            tracked_armors.erase(tracked_armors.begin() + i);
            --i;
        }
    }
    
    std::vector<torch::Tensor> tensor_images;
    std::vector<torch::Tensor> model_outputs;
    for (size_t i = 0; i < armors.size(); ++i) {
        auto armor = armors[i];
        cv::Mat roi = preprocessROI(img, armor);
        torch::Tensor tensor_image = torch::from_blob(
            roi.data, 
            {3, INPUT_HEIGHT, INPUT_WIDTH},
            torch::kFloat32
        ).clone();
        tensor_images.push_back(tensor_image);
    }
    if (armors.size() > 0) {
        torch::Tensor stacked_tensor_images = torch::stack(tensor_images, 0).to(device);
        model_outputs = model->forward(stacked_tensor_images);
        /* std::ostringstream toPrint;
        toPrint << "model_outputs[i].sizes()" << model_outputs[0][0].cpu().item<float>() << model_outputs[1].sizes()
        << model_outputs[2].sizes() << model_outputs[3].sizes() 
        << std::get<1>(torch::softmax(model_outputs[4][0], 0).cpu().max(0)).item<int>()
        << "\n";
        RCLCPP_DEBUG(node->get_logger(), toPrint.str().c_str()); // DEBUG */
    }
    
    for (size_t i = 0; i < armors.size(); ++i) {

        auto armor = armors[i];
        // 获取多输出头结果
        float is_armor_probability = torch::sigmoid(model_outputs[0][i]).cpu().item<float>();
        float is_large_probability = torch::sigmoid(model_outputs[1][i]).cpu().item<float>();
        float not_screen_probability = torch::sigmoid(model_outputs[2][i]).cpu().item<float>();
        float not_slant_probability = torch::sigmoid(model_outputs[3][i]).cpu().item<float>();
        auto classify_probabilities = torch::softmax(model_outputs[4][i], 0).cpu();
        
        auto max_result = classify_probabilities.max(0);
        int current_number = std::get<1>(max_result).item<int>();
        float classify_confidence = std::get<0>(max_result).item<float>();
        
        RCLCPP_DEBUG(node->get_logger(), "----------ArmorClassifier Debug Flag----------");

        // 计算当前装甲板中心点
        cv::Point2f current_center = armor.center;

        is_armor_probability = 1.0; // DEBUG
        is_large_probability = 0.0;
        not_screen_probability = 1.0;
        not_slant_probability = 1.0;
        current_number = 1;
        classify_confidence = 1.0;

        bool is_ture_armor = (is_armor_probability >= IS_ARMOR_THRESHOLD) &&
                                (not_screen_probability >= NOT_SCREEN_THRESHOLD) &&
                                (classify_confidence >= CLASSIFY_THRESHOLD);
        
        if (is_ture_armor) {
            bool is_large = is_large_probability > IS_LARGE_THRESHOLD;
            float armor_type_confidence = 1.0 - is_large_probability;
            if (is_large)
            {
                armor_type_confidence = is_large_probability;
            }
            float confidence = std::pow(
                is_armor_probability * armor_type_confidence * not_screen_probability * classify_confidence, 
                1.0 / 4.0
            );
            bool not_slant = not_slant_probability > NOT_SLANT_THRESHOLD; // TODO：倾斜目标纠正网络

            // 检测是否正在跟踪当前装甲板
            bool is_tracked = false;
            for (size_t j = 0; j < tracked_armors.size(); ++j) {
                if (current_number == tracked_armors[j].number && 
                    is_large == tracked_armors[j].is_large &&
                    isNearPreviousCenter(current_center, tracked_armors[j].center_last_seen)) {
                    // 若正在跟踪则更新
                    tracked_armors[j].tracking_count += 1;
                    tracked_armors[j].last_seen = current_time;
                    tracked_armors[j].center_last_seen = current_center;
                    tracked_armors[j].is_tracked_now = true;
                    tracked_armors[j].armor_last_seen = armor;
                    tracked_armors[j].confidence = confidence;
                    tracked_armors[j].not_slant = not_slant;
                    is_tracked = true;
                    break;
                }
            }
            // 若未在跟踪则添加至跟踪列表
            if(!is_tracked) {
                TrackedArmor new_tracked_armor(current_number, current_time, current_center, 
                    armor, confidence, is_large, not_slant);
                tracked_armors.push_back(new_tracked_armor);
            }
        }
    }
    // 更新所有目标
    for (size_t i = 0; i < tracked_armors.size(); ++i) {
        if (tracked_armors[i].last_seen != current_time && tracked_armors[i].tracking_count > 0) {
            tracked_armors[i].tracking_count -= 1;
        }        
        if (tracked_armors[i].tracking_count >= MIN_TRACKING_COUNT) {
            tracked_armors[i].is_steady_tracked = true;
        } else {
            tracked_armors[i].is_steady_tracked = false;
        }
    }
    // 输出
    for (size_t i = 0; i < tracked_armors.size(); ++i) {        
        if (tracked_armors[i].is_steady_tracked) {
            results.emplace_back(tracked_armors[i].armor_last_seen, tracked_armors[i].number, tracked_armors[i].confidence, 
            tracked_armors[i].is_tracked_now, tracked_armors[i].is_large, tracked_armors[i].not_slant);
        }
    }
    
    return results;
}
