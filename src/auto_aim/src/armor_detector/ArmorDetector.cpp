// ArmorDetector.cpp
#include "armor_detector/ArmorDetector.h"
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>

void ArmorDetector::initCameraMatrix(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node) {
    const YAML::Node& camera_matrix_Node = (*config_file_ptr)["camera_matrix"];
    RCLCPP_DEBUG(node->get_logger(), "camera_matrix_config: \n[%f, %f, %f,\n %f, %f, %f,\n %f, %f, %f]\n", 
        camera_matrix_Node[0][0].as<double>(), camera_matrix_Node[0][1].as<double>(), camera_matrix_Node[0][2].as<double>(), 
        camera_matrix_Node[1][0].as<double>(), camera_matrix_Node[1][1].as<double>(), camera_matrix_Node[1][2].as<double>(), 
        camera_matrix_Node[2][0].as<double>(), camera_matrix_Node[2][1].as<double>(), camera_matrix_Node[2][2].as<double>());
    // 相机内参矩阵
    camera_matrix = (cv::Mat_<double>(3, 3) << 
        camera_matrix_Node[0][0].as<double>(), camera_matrix_Node[0][1].as<double>(), camera_matrix_Node[0][2].as<double>(), 
        camera_matrix_Node[1][0].as<double>(), camera_matrix_Node[1][1].as<double>(), camera_matrix_Node[1][2].as<double>(), 
        camera_matrix_Node[2][0].as<double>(), camera_matrix_Node[2][1].as<double>(), camera_matrix_Node[2][2].as<double>());
    
    const YAML::Node& dist_coeffs_Node = (*config_file_ptr)["dist_coeffs"];
    RCLCPP_DEBUG(node->get_logger(), "dist_coeffs_config: %f, %f, %f, %f, %f\n", 
        dist_coeffs_Node[0].as<double>(), dist_coeffs_Node[1].as<double>(), dist_coeffs_Node[2].as<double>(), 
        dist_coeffs_Node[3].as<double>(), dist_coeffs_Node[4].as<double>());
    // 畸变系数
    dist_coeffs = (cv::Mat_<double>(1, 5) << 
        dist_coeffs_Node[0].as<double>(), dist_coeffs_Node[1].as<double>(), dist_coeffs_Node[2].as<double>(), 
        dist_coeffs_Node[3].as<double>(), dist_coeffs_Node[4].as<double>());
}

void ArmorDetector::initArmorPoints() {
    // 使用小装甲板尺寸初始化（因为在初始化阶段我们还不知道具体是哪种装甲板）
    const float HALF_WIDTH = ArmorConstants::SMALL_ARMOR_WIDTH / 2.0f;   // 67.5mm
    const float HALF_HEIGHT = ArmorConstants::SMALL_ARMOR_HEIGHT / 2.0f; // 62.5mm
    
    armor_points_3d = {
        cv::Point3f(-HALF_WIDTH, -HALF_HEIGHT, 0.0f),  // 左上
        cv::Point3f(HALF_WIDTH, -HALF_HEIGHT, 0.0f),   // 右上
        cv::Point3f(HALF_WIDTH, HALF_HEIGHT, 0.0f),    // 右下
        cv::Point3f(-HALF_WIDTH, HALF_HEIGHT, 0.0f)    // 左下
    };
}

cv::Point2f ArmorDetector::project3DToPixel(const cv::Point3f& world_point) const {
    // 确保相机参数已初始化
    if (camera_matrix.empty() || dist_coeffs.empty()) {
        throw std::runtime_error("Camera parameters not initialized!");
    }

    // 将3D点转换为OpenCV输入格式
    std::vector<cv::Point3f> object_points = {world_point};
    std::vector<cv::Point2f> image_points;

    // 使用solvePnP投影
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);  // 假设无旋转
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);  // 假设无平移
    
    // 直接使用projectPoints进行投影
    cv::projectPoints(object_points, rvec, tvec, 
                     camera_matrix, dist_coeffs, 
                     image_points);

    return image_points[0];
}

// 修改solveArmor函数实现
AimResult ArmorDetector::solveArmor(const Armor& armor, int number) const {
    AimResult result;
    result.valid = false;
    
    try {
        bool is_large_armor = (number == 1 || number == 5);
        
        float half_width = is_large_armor ? 
            ArmorConstants::LARGE_ARMOR_WIDTH / 2.0f :
            ArmorConstants::SMALL_ARMOR_WIDTH / 2.0f;
            
        float half_height = is_large_armor ? 
            ArmorConstants::LARGE_ARMOR_HEIGHT / 2.0f :
            ArmorConstants::SMALL_ARMOR_HEIGHT / 2.0f;
            
        std::vector<cv::Point3f> armor_points_3d = {
            cv::Point3f(-half_width, -half_height, 0.0f),
            cv::Point3f(half_width, -half_height, 0.0f),
            cv::Point3f(half_width, half_height, 0.0f),
            cv::Point3f(-half_width, half_height, 0.0f)
        };

        cv::Mat rvec, tvec;
        bool solve_success = cv::solvePnP(armor_points_3d, armor.corners, 
                                        camera_matrix, dist_coeffs, 
                                        rvec, tvec, false, cv::SOLVEPNP_IPPE);
        
        if (!solve_success) {
            std::cerr << "PnP solve failed!" << std::endl;
            return result;
        }
        
        // 设置位置信息（相机坐标系下的三维位置）
        result.position = cv::Point3f(tvec.at<double>(0),
                                    tvec.at<double>(1),
                                    tvec.at<double>(2));
        
        // 计算距离
        result.distance = cv::norm(result.position);
        
        // 标记解算成功
        result.valid = true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in solveArmor: " << e.what() << std::endl;
    }
    
    return result;
}

std::vector<Armor> ArmorDetector::detectArmors(const std::vector<Light>& lights) {
    std::vector<Armor> armors;


    // 2. 遍历所有可能的灯条对
    for (size_t i = 0; i < lights.size(); i++) {
        for (size_t j = i + 1; j < lights.size(); j++) {
            const Light& l1 = lights[i];
            const Light& l2 = lights[j];
            
            const Light& leftLight = (l1.el.center.x < l2.el.center.x) ? l1 : l2;
            const Light& rightLight = (l1.el.center.x < l2.el.center.x) ? l2 : l1;
            
            if (isArmorPair(leftLight, rightLight)) {
                Armor armor(leftLight.el, rightLight.el, corners_expand_ratio, node);
                armor.confidence = getArmorConfidence(leftLight, rightLight);
                
                // 只添加置信度足够高的装甲板
                if (armor.confidence >= min_armor_confidence) {
                    armors.push_back(armor);
                }
            }
        }
    }

    // 根据置信度排序
    std::sort(armors.begin(), armors.end(),
        [](const Armor& a1, const Armor& a2) {
            return a1.confidence > a2.confidence;
        });

    return armors;
}


// 修改装甲板配对条件
bool ArmorDetector::isArmorPair(const Light& l1, const Light& l2) {
    // 1. 检查灯条间距
    float distance = cv::norm(l1.el.center - l2.el.center);
    float avg_light_height = (l1.length + l2.length) / 2.0f;
    
    float expected_small_distance = (ArmorConstants::SMALL_ARMOR_WIDTH / ArmorConstants::LIGHT_HEIGHT) * avg_light_height;
    float expected_large_distance = (ArmorConstants::LARGE_ARMOR_WIDTH / ArmorConstants::LIGHT_HEIGHT) * avg_light_height;
    
    bool distance_match = false;
    if (std::abs(distance - expected_small_distance) / expected_small_distance <= max_expected_small_distance_mismatch_ratio ||
        std::abs(distance - expected_large_distance) / expected_large_distance <= max_expected_large_distance_mismatch_ratio) {
        distance_match = true;
    }
    
    if (!distance_match) return false;
    
    // 2. 检查灯条平行度
    float angleDiff = std::abs(l1.angle - l2.angle);
    if (angleDiff > max_angle_diff) {
        return false;
    }
    
    // 3. 检查灯条高度比例
    float heightDiff = std::abs(l1.length - l2.length);
    if (heightDiff / std::min(l1.length, l2.length) > max_height_diff_ratio) {
        return false;
    }
    
    return true;
}

float ArmorDetector::getArmorConfidence(const Light& l1, const Light& l2) {
    // 1. 角度差得分
    float angleDiff = std::abs(l1.angle - l2.angle);
    float angleScore = 1.0f - (angleDiff / max_angle_diff * 1.5f);
    
    // 2. 高度差得分
    float heightDiff = std::abs(l1.length - l2.length);
    float averageHeight = (l1.length + l2.length) / 2;
    float heightScore = 1.0f - (heightDiff / averageHeight);
    
    // 3. 距离得分
    float distance = cv::norm(l1.el.center - l2.el.center);
    float expectedDistance = averageHeight * 1.5f; // 理想的灯条间距
    float distanceScore = 1.0f - std::abs(distance - expectedDistance) / expectedDistance;
    
    // 综合评分
    return (angleScore + heightScore + distanceScore) / 3.0f;
}