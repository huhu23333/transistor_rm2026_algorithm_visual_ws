

#ifndef ARMOR_DETECTOR_BA_SOLVER_HPP_
#define ARMOR_DETECTOR_BA_SOLVER_HPP_

// std
#include <array>
#include <cstddef>
#include <tuple>
#include <vector>
// 3rd party
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <sophus/so3.hpp>
#include <std_msgs/msg/float32.hpp>
// g2o
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/sparse_optimizer.h>
// project
#include "ba_solver/graph_optimizer.hpp" 
#include "armor_detector/Armor.h"
#include <yaml-cpp/yaml.h>


namespace fyt::auto_aim {

// BA algorithm based Optimizer for the armor pose estimation (Particularly for
// the Yaw angle)
// 基于BA算法的对于装甲板位姿估计（尤其是对于yaw轴角度）的优化
class BaSolver {
public:
  BaSolver(const std::array<double, 9> &camera_matrix, //3x3的内参矩阵
           const std::vector<double> &dist_coeffs); // 畸变系数

  // Solve the armor pose using the BA algorithm, return the optimized rotation
  // 用BA算法解算装甲板位姿，并返回优化后的矩阵
  Eigen::Matrix3d solveBa(const ArmorResult &armor,       // 一个struct 在type里面
                          const Eigen::Vector3d &t_camera_armor, // 相机系下的平移
                          const Eigen::Matrix3d &R_camera_armor, // 相机系下的旋转
                          const Eigen::Matrix3d &R_imu_camera) noexcept; //外参矩阵
  
    // 小工具：将旋转矩阵转化为ROoll、Roll、Yaw、Pitch。
  Eigen::Vector3d rotationMatrixToRPY(const Eigen::Matrix3d& R) {
    Eigen::Vector3d rpy;

    // 数值稳定：对 asin 的输入做夹取
    double sp = -R(2,0);
    if (sp >  1.0) sp =  1.0;
    if (sp < -1.0) sp = -1.0;

    rpy[0] = std::asin(sp);                // pitch
    rpy[1] = std::atan2(R(2,1), R(2,2));   // yaw
    rpy[2] = std::atan2(R(1,0), R(0,0));   // roll
    return rpy; 
  }

  Eigen::Matrix3d RPYTorotationMatrix(const Eigen::Vector3d& rpy) {
    // 改后约定：rpy = [rollZ, pitchX, yawY] ，单位：rad
    const double yawZ  = rpy[0]; // 绕 Z（偏航）
    const double pitchX = rpy[1]; // 绕 X（俯仰）
    const double rollY   = rpy[2]; // 绕 Y（滚转）

    const Eigen::AngleAxisd Rz(rollY,  Eigen::Vector3d::UnitZ());
    const Eigen::AngleAxisd Rx(pitchX, Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd Ry(yawZ,   Eigen::Vector3d::UnitY());

    // 组合顺序：先 yaw，再 pitch，最后 roll
    return (Rz * Rx * Ry).toRotationMatrix();
}  


  template<typename TPoint3>
  inline std::vector<TPoint3> buildObjectPoints(double w, double h) noexcept {
    auto make = [](double x, double y, double z) {
        // 适配 Eigen::Vector3d / cv::Point3f 的 (x,y,z) 构造
        return TPoint3(static_cast<typename std::remove_reference_t<TPoint3>::value_type>(x),
                       static_cast<typename std::remove_reference_t<TPoint3>::value_type>(y),
                       static_cast<typename std::remove_reference_t<TPoint3>::value_type>(z));
    };
    // 注意顺序：左上开始逆时针
    return {
        make(-w/2, -h/2, 0),
        make(-w/2, h/2, 0),
        make(w/2, h/2, 0),
        make(w/2, -h/2, 0),
    };
}

private:
  Eigen::Matrix3d K_; //内参矩阵
  g2o::SparseOptimizer optimizer_; // 稀疏优化器，往里面塞入顶点（具体是图论的内容）
  g2o::OptimizationAlgorithmProperty solver_property_; // 求解器
  g2o::OptimizationAlgorithmLevenberg *lm_algorithm_; // LM算法对象

    // 设置logger
  rclcpp::Logger logger_b = rclcpp::get_logger("ba_solver");

  
  

};
}

// namespace fyt::auto_aim
#endif // ARMOR_DETECTOR_BAS_SOLVER_HPP_


