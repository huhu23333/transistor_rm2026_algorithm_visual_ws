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


#include "LightBar.h"
#include "armor_detector/Armor.h"
#include "ba_solver/ba_solver.hpp"
#include "ba_solver/utils.hpp"



double getYawFromRvec(const cv::Mat& rvec) {
    if (rvec.empty()) return 0.0;
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat); // 从旋转向量得到旋转矩阵

    // 根据OpenCV相机坐标系从旋转矩阵直接计算Yaw角
    // Yaw是绕Y轴的旋转，一个稳健的计算方法如下：
    // yaw = atan2(-R(2,0), sqrt(R(0,0)^2 + R(1,0)^2))
    double yaw = std::atan2(-rmat.at<double>(2, 0),
                           std::sqrt(std::pow(rmat.at<double>(0, 0), 2) +
                                     std::pow(rmat.at<double>(1, 0), 2)));
    return yaw;
}
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

    AimResult solveArmor(const ArmorResult& armor_resul, const double last_pitch_rad_, const double last_yaw_rad_) const; // 增加number参数
    
    
 

private:
    // 相机参数
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    // 装甲板3D点(单位：mm)
    std::vector<cv::Point3f> armor_points_3d;
    
    void initCameraMatrix(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node);
    void initArmorPoints();
    rclcpp::Node* node;

    std::unique_ptr<fyt::auto_aim::BaSolver> ba_;

        // 设置logger
    rclcpp::Logger logger_p = rclcpp::get_logger("armor_solver");
    
};

#endif // ARMOR_SOLVER_H