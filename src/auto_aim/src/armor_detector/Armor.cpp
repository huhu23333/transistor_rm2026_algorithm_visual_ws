// Armor.cpp
#include "armor_detector/Armor.h"


// 带参数的构造函数
Armor::Armor(const cv::RotatedRect& left, const cv::RotatedRect& right, std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node) 
    : leftLight(left), rightLight(right), confidence(0.0f), node(node) {
    corners_expand_ratio = (*config_file_ptr)["corners_expand_ratio"].as<float>();
    height_correct_ratio_small = (*config_file_ptr)["height_correct_ratio_small"].as<float>();
    width_correct_ratio_small = (*config_file_ptr)["width_correct_ratio_small"].as<float>();
    height_correct_ratio_large = (*config_file_ptr)["height_correct_ratio_large"].as<float>();
    width_correct_ratio_large = (*config_file_ptr)["width_correct_ratio_large"].as<float>();
    calculateROI();
}

// ROI计算函数
void Armor::calculateROI() {
    // 更新corners向量
    corners.clear();
    calculateCorners();
    // RCLCPP_DEBUG(node->get_logger(), "----------Armor Debug Flag----------");

    // 计算ROI
    roi = cv::boundingRect(corners);
}

void Armor::calculateCorners() {
    // 获取左右灯条的顶点
    cv::Point2f left_vertices[4], right_vertices[4];
    leftLight.points(left_vertices);
    rightLight.points(right_vertices);
    
    // 找到左右灯条的中心点
    cv::Point2f left_center = leftLight.center;
    cv::Point2f right_center = rightLight.center;

    // 获取左右灯条的顶点转换为相对中心点的坐标
    cv::Vec2f relative_left_vertices[4], relative_right_vertices[4];
    for (int i = 0; i < 4; i+=1) {
        relative_left_vertices[i] = pointToVec(left_vertices[i] - left_center);
        relative_right_vertices[i] = pointToVec(right_vertices[i] - right_center);
    }

    // 获取左灯条中心点指向右灯条中心点的向量及垂直其向上的向量，并单位化
    cv::Vec2f d_center_vector = right_center - left_center;
    cv::Vec2f vertical_d_center_vector = cv::Vec2f(d_center_vector[1], -d_center_vector[0]);
    d_center_vector = cv::normalize(d_center_vector);
    vertical_d_center_vector = cv::normalize(vertical_d_center_vector);

    // 获取左右灯条长边及短边的单位方向向量，并调整方向为向右或向上
    float rad_left = -leftLight.angle * M_PI / 180.0;
    float rad_right = -rightLight.angle * M_PI / 180.0;
    cv::Vec2f left_length_direction = cv::Vec2f(std::sin(rad_left), std::cos(rad_left));
    cv::Vec2f left_width_direction = cv::Vec2f(-left_length_direction[1], left_length_direction[0]);
    cv::Vec2f right_length_direction = cv::Vec2f(std::sin(rad_right), std::cos(rad_right));
    cv::Vec2f right_width_direction = cv::Vec2f(-right_length_direction[1], right_length_direction[0]);
    if (left_length_direction.dot(vertical_d_center_vector) < 0) left_length_direction = -left_length_direction;
    if (left_width_direction.dot(d_center_vector) < 0) left_width_direction = -left_width_direction;
    if (right_length_direction.dot(vertical_d_center_vector) < 0) right_length_direction = -right_length_direction;
    if (right_width_direction.dot(d_center_vector) < 0) right_width_direction = -right_width_direction;

    // 将顶点相对坐标拆分为沿长边和短边两部分
    cv::Vec2f horizontal_relative_left_vertices[4], horizontal_relative_right_vertices[4],
                vertical_relative_left_vertices[4], vertical_relative_right_vertices[4];
    for (int i = 0; i < 4; i+=1) {
        horizontal_relative_left_vertices[i] = left_width_direction * relative_left_vertices[i].dot(left_width_direction);
        horizontal_relative_right_vertices[i] = right_width_direction * relative_right_vertices[i].dot(right_width_direction);
        vertical_relative_left_vertices[i] = left_length_direction * relative_left_vertices[i].dot(left_length_direction);
        vertical_relative_right_vertices[i] = right_length_direction * relative_right_vertices[i].dot(right_length_direction);
    }

    // 获取灯条顶点坐标中靠近装甲板中心的四个点的相对中心点坐标的沿长边和短边两部分
    cv::Vec2f left_up_horizontal, left_up_vertical, left_down_horizontal, left_down_vertical, 
                right_up_horizontal, right_up_vertical, right_down_horizontal, right_down_vertical; 
    for (int i = 0; i < 4; i+=1) {
        if (horizontal_relative_left_vertices[i].dot(left_width_direction) >= 0)
        {
            if (vertical_relative_left_vertices[i].dot(left_length_direction) >= 0)
            {
                left_up_horizontal = horizontal_relative_left_vertices[i];
                left_up_vertical = vertical_relative_left_vertices[i];
            }
            else
            {
                left_down_horizontal = horizontal_relative_left_vertices[i];
                left_down_vertical = vertical_relative_left_vertices[i];
            }
        }
        if (horizontal_relative_right_vertices[i].dot(right_width_direction) <= 0)
        {
            if (vertical_relative_right_vertices[i].dot(right_length_direction) >= 0)
            {
                right_up_horizontal = horizontal_relative_right_vertices[i];
                right_up_vertical = vertical_relative_right_vertices[i];
            }
            else
            {
                right_down_horizontal = horizontal_relative_right_vertices[i];
                right_down_vertical = vertical_relative_right_vertices[i];
            }
        }
    }

    // 沿长边部分使用小装甲板比例获得装甲板高度相对坐标
    left_up_vertical *= ArmorConstants::SMALL_HEIGHT_RATIO;
    left_down_vertical *= ArmorConstants::SMALL_HEIGHT_RATIO;
    right_up_vertical *= ArmorConstants::SMALL_HEIGHT_RATIO;
    right_down_vertical *= ArmorConstants::SMALL_HEIGHT_RATIO;

    std::vector<cv::Point2f> corners_biased;

    // 按从左上角开始逆时针排序
    corners_biased.push_back(left_center + vecToPoint(left_up_horizontal + left_up_vertical));
    corners_biased.push_back(left_center + vecToPoint(left_down_horizontal + left_down_vertical));
    corners_biased.push_back(right_center + vecToPoint(right_down_horizontal + right_down_vertical));
    corners_biased.push_back(right_center + vecToPoint(right_up_horizontal + right_up_vertical));

    // 计算中心
    center = computeIntersection(corners_biased);

    // 拆分为装甲板中心坐标下纵向和横向坐标，修正坐标并输出
    for (int i = 0; i < 4; i+=1) {
        cv::Vec2f center_to_corner = pointToVec(corners_biased[i] - center);
        cv::Vec2f center_to_corner_height = vertical_d_center_vector * center_to_corner.dot(vertical_d_center_vector);
        cv::Vec2f center_to_corner_width = d_center_vector * center_to_corner.dot(d_center_vector);
        corners.push_back(center + 
            vecToPoint(center_to_corner_height * height_correct_ratio_small + center_to_corner_width * width_correct_ratio_small));
        corners_large.push_back(center + 
            vecToPoint(center_to_corner_height * height_correct_ratio_large + center_to_corner_width * width_correct_ratio_large));
    }

    // 计算扩大后角点坐标
    for (int i = 0; i < 4; i+=1) {
        corners_expanded.push_back(center + corners_expand_ratio * (corners[i] - center));
    }
}

cv::Point2f Armor::vecToPoint(const cv::Vec2f& vec) {
    return cv::Point2f(vec[0], vec[1]);
}

cv::Vec2f Armor::pointToVec(const cv::Point2f& point) {
    return cv::Vec2f(point.x, point.y);
}

cv::Point2f Armor::computeIntersection(const std::vector<cv::Point2f>& corners) {
    // 提取对角线端点
    cv::Point2f A1 = corners[0]; // 左上角
    cv::Point2f A2 = corners[2]; // 右下角
    cv::Point2f B1 = corners[1]; // 左下角
    cv::Point2f B2 = corners[3]; // 右上角

    // 计算向量
    cv::Point2f a = A2 - A1; // 对角线1的向量
    cv::Point2f b = B2 - B1; // 对角线2的向量
    cv::Point2f c = B1 - A1; // 从A1指向B1的向量

    // 计算叉积
    float cross_ab = a.cross(b); // a × b
    float cross_cb = c.cross(b); // c × b

    // 检查对角线是否平行
    if (std::fabs(cross_ab) < 1e-6) {
        throw std::runtime_error("Diagonals are parallel, no intersection.");
    }

    // 计算参数t
    float t = cross_cb / cross_ab;

    // 计算交点坐标
    cv::Point2f P = A1 + t * a;

    return P;
}