// ArmorCLassifier.cpp
#include "armor_detector/ArmorClassifier.h"

/* #include <iostream>
#include <sstream>
#include <string>
// DEBUG */

namespace fs = std::filesystem;

ArmorClassifier::ArmorClassifier(std::shared_ptr<YAML::Node> config_file_ptr, bool use_cuda, rclcpp::Node* node) 
    : node(node) {
    
    MAX_ROI_SAVE_COUNT = (*config_file_ptr)["MAX_ROI_SAVE_COUNT"].as<int>();
    classify_classes = (*config_file_ptr)["classify_classes"].as<int>();

    IS_ARMOR_THRESHOLD = (*config_file_ptr)["IS_ARMOR_THRESHOLD"].as<float>();
    IS_LARGE_THRESHOLD = (*config_file_ptr)["IS_LARGE_THRESHOLD"].as<float>();
    NOT_SCREEN_THRESHOLD = (*config_file_ptr)["NOT_SCREEN_THRESHOLD"].as<float>();
    NOT_SLANT_THRESHOLD = (*config_file_ptr)["NOT_SLANT_THRESHOLD"].as<float>();
    CLASSIFY_THRESHOLD = (*config_file_ptr)["CLASSIFY_THRESHOLD"].as<float>();
    INPUT_HEIGHT = (*config_file_ptr)["INPUT_HEIGHT"].as<int>();
    INPUT_WIDTH = (*config_file_ptr)["INPUT_WIDTH"].as<int>();
    MAX_TRACKING_AGE_MS = (*config_file_ptr)["MAX_TRACKING_AGE_MS"].as<int>();
    MIN_TRACKING_COUNT = (*config_file_ptr)["MIN_TRACKING_COUNT"].as<int>();
    IS_NEAR_MAX_DIST_RATIO = (*config_file_ptr)["IS_NEAR_MAX_DIST_RATIO"].as<float>();
    fit_step = (*config_file_ptr)["fit_step"].as<int>();
    predict_step = (*config_file_ptr)["predict_step"].as<int>();
    fourier_fit_step = (*config_file_ptr)["fourier_fit_step"].as<int>();
    fourier_fit_order = (*config_file_ptr)["fourier_fit_order"].as<int>();
    fourier_predict_step = (*config_file_ptr)["fourier_predict_step"].as<int>();
    MAX_FOURIER_TRACKING_AGE_MS = (*config_file_ptr)["MAX_FOURIER_TRACKING_AGE_MS"].as<int>();

    auto current_time = std::chrono::steady_clock::now();
    for (int number = 0; number < classify_classes; ++number)
    {
        classified_latest_tracked_armors.emplace_back(number, current_time, cv::Point2f(0, 0), 
                                                      Armor(), 0.0, false, false, fourier_fit_step, cv::Point2f(0, 0));
    }
    
    shm_pytorch_processor = std::make_shared<SharedMemoryTorch>(config_file_ptr);
}

cv::Mat ArmorClassifier::preprocessROI(const cv::Mat& img, const Armor& armor) {
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

    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    
    // cv::imshow("Classifier DEBUG", resized);
    // 如果已经保存了1000张图片，直接返回处理后的图像而不保存
    if (roi_save_count >= MAX_ROI_SAVE_COUNT) {
        return resized;
    }
    
    // 保存处理后的图像（用于神经网络输入的标准化图像）
    if (roi_save_count < MAX_ROI_SAVE_COUNT) {
        // 创建保存目录
        fs::create_directories("network_input_images");
        
        // 生成文件名（00001.jpg 格式）
        std::ostringstream filename;
        filename << "network_input_images/"
                << std::setw(5) << std::setfill('0') << (roi_save_count.fetch_add(1) + 1)
                << ".jpg";
        
        cv::imwrite(filename.str(), resized);
        
        if (roi_save_count == MAX_ROI_SAVE_COUNT) {
            std::cout << "Reached maximum number of saved images (2000)" << std::endl;
        }
    }
    
    return resized;
}

bool ArmorClassifier::isNearPreviousCenter(const Armor& current_armor, 
                                           const cv::Point2f& ground_stable_point,
                                           const TrackedArmor& previous_tracked_armor, 
                                           float max_dist_ratio) {
    cv::Point2f current_center_ground_stable = current_armor.center - ground_stable_point;
    cv::Point2f previous_center_ground_stable = previous_tracked_armor.center_last_seen - previous_tracked_armor.last_ground_stable_point;
    if (max_dist_ratio < 0)
    {
        max_dist_ratio = IS_NEAR_MAX_DIST_RATIO;
    }
    // 根据装甲板最远两角点距离确定距离基础值
    float corners_max_dist = 0.0;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
            float corners_dist = cv::norm(current_armor.corners[i] - current_armor.corners[j]);
            if (corners_dist > corners_max_dist) {
                corners_max_dist = corners_dist;
            }
        }
    }
    // 根据系数参数修正
    float max_dist = corners_max_dist * max_dist_ratio;
    float dist = cv::norm(current_center_ground_stable - previous_center_ground_stable);
    return dist <= max_dist;
}

struct alignas(64) RoiImageThreadInfo { // 64字节对齐
    const Armor* armor;
    size_t armor_index;
};

std::vector<std::vector<ArmorResult>> ArmorClassifier::classify(
    const cv::Mat& img, const std::vector<Armor>& armors, const cv::Point2f ground_stable_point) {
    
    std::vector<std::vector<ArmorResult>> results;
    results.push_back(std::vector<ArmorResult>());
    results.push_back(std::vector<ArmorResult>());
    auto current_time = std::chrono::steady_clock::now();
    int process_armors_count = armors.size();
    process_armors_count = std::min(process_armors_count, 100);
    std::vector<cv::Mat> roi_images(process_armors_count);
    std::vector<RoiImageThreadInfo> roiImageThreadInfos(process_armors_count);
    std::vector<std::vector<float>> pytorch_results;
    
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
    /* for (size_t i = 0; i < classified_latest_tracked_armors.size(); ++i) {
        classified_latest_tracked_armors[i].is_tracked_now = false;
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - classified_latest_tracked_armors[i].last_seen).count();
        if (age > MAX_FOURIER_TRACKING_AGE_MS) {
            classified_latest_tracked_armors[i].tracking_count = 0;
            classified_latest_tracked_armors[i].center_last_seen = cv::Point2f(0, 0);
            classified_latest_tracked_armors[i].is_steady_tracked = false;
            classified_latest_tracked_armors[i].confidence = 0.0;
            classified_latest_tracked_armors[i].predictor.clearHistory(); 
            classified_latest_tracked_armors[i].predictions.clear();
            classified_latest_tracked_armors[i].center_predicted = cv::Point2f(0, 0);
            classified_latest_tracked_armors[i].prediction_index = 0;
            classified_latest_tracked_armors[i].last_ground_stable_point = cv::Point2f(0, 0);
        }
    } */
    for (size_t i = 0; i < process_armors_count; ++i) {
        roiImageThreadInfos[i].armor = &armors[i];
        roiImageThreadInfos[i].armor_index = i;
    }
    // 进行多线程优化
    std::for_each(std::execution::par, roiImageThreadInfos.begin(), roiImageThreadInfos.end(), 
    [&](RoiImageThreadInfo& roiImageThreadInfo) {
        roi_images[roiImageThreadInfo.armor_index] = preprocessROI(img, *roiImageThreadInfo.armor);
    });
    if (process_armors_count > 0) {
        pytorch_results = shm_pytorch_processor->processImages(roi_images);
    }
    
    for (size_t i = 0; i < process_armors_count; ++i) {

        auto armor = armors[i];
        // 计算当前装甲板中心点
        cv::Point2f current_center = armor.center;
        
        // 获取多输出头结果
        float is_armor_probability;
        float is_large_probability;
        float not_screen_probability;
        float not_slant_probability;
        std::vector<float> classify_probabilities(8);
        int current_number;
        float classify_confidence;
        
        is_armor_probability = pytorch_results[i][0];
        is_large_probability = pytorch_results[i][1];
        not_screen_probability = pytorch_results[i][2];
        not_slant_probability = pytorch_results[i][3];
        std::copy(pytorch_results[i].begin() + 4, pytorch_results[i].begin() + 12, classify_probabilities.begin());
        
        auto classify_max_it = std::max_element(classify_probabilities.begin(), classify_probabilities.end());
        if (classify_max_it != classify_probabilities.end()) {
            classify_confidence = *classify_max_it;
            current_number = std::distance(classify_probabilities.begin(), classify_max_it);
        }

        RCLCPP_DEBUG(node->get_logger(), "ArmorClassifier Debug:\n %.2f | %.2f | %.2f | %.2f | %.2f | %d", 
            is_armor_probability, is_large_probability, not_screen_probability, not_slant_probability, classify_confidence, current_number
        );

        is_armor_probability = 1.0; // DEBUG
        is_large_probability = 0.0;
        not_screen_probability = 1.0;
        not_slant_probability = 1.0;
        current_number = 1;
        classify_confidence = 1.0;

        not_screen_probability = 1.0;

        bool is_ture_armor = (is_armor_probability >= IS_ARMOR_THRESHOLD) &&
                                (not_screen_probability >= NOT_SCREEN_THRESHOLD) &&
                                (classify_confidence >= CLASSIFY_THRESHOLD);
        
        bool not_slant = not_slant_probability > NOT_SLANT_THRESHOLD; // TODO：倾斜目标纠正网络

        if (is_ture_armor && not_slant) {
            bool is_large = is_large_probability > IS_LARGE_THRESHOLD;
            float armor_type_confidence = 1.0 - is_large_probability;
            if (is_large)
            {
                armor_type_confidence = is_large_probability;
                armor.corners = armor.corners_large;
            }
            float confidence = std::pow(
                std::abs(is_armor_probability * armor_type_confidence * not_screen_probability * classify_confidence * not_slant) + 1e-6, 
                1.0 / 5.0
            );


            // 检测是否正在跟踪当前装甲板
            bool is_tracked = false;
            for (size_t j = 0; j < tracked_armors.size(); ++j) {
                if (current_number == tracked_armors[j].number && 
                    is_large == tracked_armors[j].is_large &&
                    //isNearPreviousCenter(current_center, tracked_armors[j].center_last_seen)) {
                    isNearPreviousCenter(armor, ground_stable_point, tracked_armors[j])) {
                    // 若正在跟踪则更新
                    tracked_armors[j].tracking_count += 1;
                    tracked_armors[j].last_seen = current_time;
                    tracked_armors[j].center_last_seen = current_center;
                    tracked_armors[j].is_tracked_now = true;
                    tracked_armors[j].armor_last_seen = armor;
                    tracked_armors[j].confidence = confidence;
                    tracked_armors[j].not_slant = not_slant;
                    tracked_armors[j].last_ground_stable_point = ground_stable_point;
                    is_tracked = true;
                    break;
                }
            }
            // 若未在跟踪则添加至跟踪列表
            if(!is_tracked) {
                tracked_armors.emplace_back(current_number, current_time, current_center, 
                    armor, confidence, is_large, not_slant, fit_step, ground_stable_point);
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

        // RCLCPP_DEBUG(node->get_logger(), "----------ArmorClassifier Debug Flag----------");

        if (tracked_armors[i].is_tracked_now) {
            for (int j = 0; j < tracked_armors[i].prediction_index-1; ++j)
            {
                tracked_armors[i].predictor.addPoint(tracked_armors[i].predictions[j] - tracked_armors[i].last_ground_stable_point);
            }
            tracked_armors[i].predictor.addPoint(tracked_armors[i].center_last_seen - tracked_armors[i].last_ground_stable_point);
            tracked_armors[i].predictor.fitLinear(fit_step);
            tracked_armors[i].predictions = tracked_armors[i].predictor.predictLinear(predict_step, tracked_armors[i].last_ground_stable_point);
            tracked_armors[i].prediction_index = 0;
        } else if (tracked_armors[i].prediction_index < predict_step-1) {
            tracked_armors[i].prediction_index += 1;
        }
        if (tracked_armors[i].is_steady_tracked) {
            tracked_armors[i].center_predicted = tracked_armors[i].predictions[tracked_armors[i].prediction_index];
            // 将某一类中最新稳定跟踪的装甲板赋值给classified_latest_tracked_armors，用于傅里叶预测
            classified_latest_tracked_armors[tracked_armors[i].number].tracking_count += 1;
            classified_latest_tracked_armors[tracked_armors[i].number].last_seen = current_time;
            classified_latest_tracked_armors[tracked_armors[i].number].center_last_seen = tracked_armors[i].center_last_seen;
            classified_latest_tracked_armors[tracked_armors[i].number].is_steady_tracked = true;
            classified_latest_tracked_armors[tracked_armors[i].number].is_tracked_now = true;
            classified_latest_tracked_armors[tracked_armors[i].number].armor_last_seen = tracked_armors[i].armor_last_seen;
            classified_latest_tracked_armors[tracked_armors[i].number].confidence = 1.0;
            classified_latest_tracked_armors[tracked_armors[i].number].is_large = tracked_armors[i].is_large;
            classified_latest_tracked_armors[tracked_armors[i].number].not_slant = tracked_armors[i].not_slant;
            classified_latest_tracked_armors[tracked_armors[i].number].predictor.addPoint(tracked_armors[i].center_last_seen - tracked_armors[i].last_ground_stable_point);
            classified_latest_tracked_armors[tracked_armors[i].number].last_ground_stable_point = tracked_armors[i].last_ground_stable_point;
        } else {
            tracked_armors[i].center_predicted = tracked_armors[i].center_last_seen;
        }
    }
    // 更新每一类装甲板傅里叶预测结果
    for (size_t i = 0; i < classified_latest_tracked_armors.size(); ++i) {;
        if (!classified_latest_tracked_armors[i].is_tracked_now) {
            classified_latest_tracked_armors[i].predictor.addPoint(classified_latest_tracked_armors[i].center_predicted - classified_latest_tracked_armors[i].last_ground_stable_point);
        }
        classified_latest_tracked_armors[i].predictor.fitFourier(fourier_fit_step, fourier_fit_order);
        classified_latest_tracked_armors[i].predictions = classified_latest_tracked_armors[i].predictor.predictFourier(fourier_predict_step, classified_latest_tracked_armors[i].last_ground_stable_point);
        classified_latest_tracked_armors[i].center_predicted = classified_latest_tracked_armors[i].predictions[0];
    }
    // 输出
    for (size_t i = 0; i < tracked_armors.size(); ++i) {
        if (tracked_armors[i].is_steady_tracked) {
            results[0].emplace_back(tracked_armors[i].armor_last_seen, tracked_armors[i].number, tracked_armors[i].confidence, 
                tracked_armors[i].is_tracked_now, tracked_armors[i].is_large, tracked_armors[i].not_slant, 
                tracked_armors[i].predictions, tracked_armors[i].center_predicted);
        }
    }
    for (size_t i = 0; i < classified_latest_tracked_armors.size(); ++i) {
        if (classified_latest_tracked_armors[i].is_steady_tracked) {
            results[1].emplace_back(
                classified_latest_tracked_armors[i].armor_last_seen, 
                classified_latest_tracked_armors[i].number, 
                classified_latest_tracked_armors[i].confidence, 
                classified_latest_tracked_armors[i].is_tracked_now, 
                classified_latest_tracked_armors[i].is_large, 
                classified_latest_tracked_armors[i].not_slant, 
                classified_latest_tracked_armors[i].predictions, 
                classified_latest_tracked_armors[i].center_predicted);
        }
    }
    
    return results;
}
