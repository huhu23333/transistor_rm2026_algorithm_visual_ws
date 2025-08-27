// ArmorSolver.h
#ifndef ARMOR_SOLVER_H
#define ARMOR_SOLVER_H
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include "LightBar.h"
#include "armor_detector/Armor.h"
#include <vector>
#include <yaml-cpp/yaml.h>
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>
#include <opencv2/core/eigen.hpp> // 用于Eigen转换

#include "rclcpp/rclcpp.hpp"
#include <fstream> // <-- 添加文件流头文件
class ArmorSolver {
public:
    ArmorSolver(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node)
    : node(node) {
        // 初始化相机参数
        initCameraMatrix(config_file_ptr, node);
        initArmorPoints();
    }
    // 新增3D到像素坐标投影函数
    cv::Point2f project3DToPixel(const cv::Point3f& world_point) const;
    AimResult solveArmor(const ArmorResult& armor_result) const; // 增加number参数
    static double getYawFromRvec(const cv::Mat& rvec);
private:
    // 相机参数
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    // 装甲板3D点(单位：mm)
    std::vector<cv::Point3f> armor_points_3d;
    
    void initCameraMatrix(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node);
    void initArmorPoints();
    rclcpp::Node* node;
};

#endif // ARMOR_SOLVER_H