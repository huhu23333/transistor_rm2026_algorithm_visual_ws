#pragma once

#include "motion_model.hpp" 
#include <memory>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

// 新增：用于传递9D EKF参数的结构体
struct EKFParams {
    // 过程噪声 Q
    double s2qx, s2qy, s2qz, s2qyaw, s2qr;
    // 测量噪声 R
    double r_x, r_y, r_z, r_yaw;
    // 初始协方差 P0
    double p0;
};

class Tracker {
public:
    using State = RobotEKF::MatrixX1;
    using Measurement = RobotEKF::MatrixZ1;

    // 追踪器状态机
    enum StateType {
        LOST,       // 完全丢失
        DETECTING,  // 刚检测到，待稳定
        TRACKING,   // 稳定追踪
        TEMP_LOST   // 短暂丢失
    } state;

    // 构造函数，传入时间步长dt和EKF参数
    explicit Tracker(double dt, const EKFParams& params);

    // 使用第一个装甲板测量值来初始化或重置滤波器
    void reset(const Measurement& z);

    State predict();
    State update(const Measurement& z);

    // 获取当前滤波后的状态（机器人中心状态）
    State getTargetState() const;

    // 提前预测机器人中心在未来t_ahead秒后的状态
    State predictAhead(double t_ahead) const;

    // 从机器人中心状态计算出当前装甲板的预测位置
    Eigen::Vector3d getArmorPosition() const;

    double r_init_;               // 初始估计半径
private:
    std::unique_ptr<RobotEKF> ekf_; // EKF滤波器实例
    double dt_;                   // 时间步长
    int tracking_thres_ = 5;      // 进入稳定追踪状态的帧数阈值
    int lost_thres_ = 10;         // 判断为完全丢失的帧数阈值
    int detect_count_ = 0;        // 稳定追踪计数
    int lost_count_ = 0;          // 丢失计数
};
