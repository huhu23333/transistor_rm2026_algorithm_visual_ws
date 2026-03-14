// ArmorSolver.h
#ifndef ARMOR_SOLVER_H
#define ARMOR_SOLVER_H
#include <Eigen/Dense>


#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <vector>
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>
#include <opencv2/core/eigen.hpp> // 用于Eigen转换
#include "rclcpp/rclcpp.hpp"
#include <fstream> // <-- 添加文件流头文件
#include <memory>


#include "2d_armor_detector/LightBar.h"
#include "2d_armor_detector/Armor.h"
#include "ba_solver/ba_solver.hpp"
#include "ba_solver/utils.hpp"

#include <Eigen/Geometry> // For Quaternion and rotation matrix math


double getYawFromRvec(const cv::Mat& rvec);

std::vector<double> getNormalYawPitchRollFromRvec(const cv::Mat& rvec);

class ArmorSolver {
    
public:
    ArmorSolver(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node)
    : node(node) {
        // 初始化相机参数
        initCameraMatrix(config_file_ptr, node);
        initArmorPoints();
        delta_x_ = (*config_file_ptr)["delta_x_"].as<float>();
        delta_y_ = (*config_file_ptr)["delta_y_"].as<float>();
        delta_z_ = (*config_file_ptr)["delta_z_"].as<float>();
    }
    // 新增3D到像素坐标投影函数
    cv::Point2f project3DToPixel(const cv::Point3f& world_point) const;

    AimResult solveArmor(const ArmorResult& armor_result, const double last_pitch_rad_, const double last_yaw_rad_); // 增加number参数

private:
    // 相机参数
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    // 装甲板3D点(单位：mm)
    std::vector<cv::Point3f> armor_points_3d;

    std::vector<fyt::auto_aim::BaFrameInput> frames;// newBA实际传vector
    
    void initCameraMatrix(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node);
    void initArmorPoints();
    rclcpp::Node* node;

    std::unique_ptr<fyt::auto_aim::BaSolver> ba_;

        // 设置logger
    rclcpp::Logger logger_p = rclcpp::get_logger("armor_solver");
    
    float delta_x_;
    float delta_y_;
    float delta_z_;
};

#endif // ARMOR_SOLVER_H