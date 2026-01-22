#pragma once

#include "motion_model.hpp"
#include <Eigen/Dense>
#include <memory>

namespace armor_ekf
{

// EKF 噪声参数（可以从 yaml 里读，也可以直接在这里调）

struct EKFParams
{
    // 过程噪声（s2 越小，滤波越平滑，但也越滞后；s2 越大，越灵敏但抖动）
    double s2qx  = 20.0;
    double s2qy  = 20.0;   
    double s2qz  = 10.0;
    double s2qyaw = 100.0;

    double s2qr   = 0.1;
    double s2qdz  = 0.1;

    // 测量噪声（r 越大，越不信任观测，滤波越平滑）
    // 如果预测点抖动厉害，适当调大 r_x, r_y
    double r_x   = 0.15; 
    double r_y   = 0.15;
    double r_z   = 0.15;
    double r_yaw = 0.05;

    // 初始协方差
    double p0 = 500.0;
};


class Tracker
{
public:
    enum class ArmorsNum { NORMAL_4 = 4, BALANCE_2 = 2, OUTPOST_3 = 3 };

    void setArmorsNum(ArmorsNum n) { armors_num_ = n; }
    void setMatchThresholds(double max_dist_mm, double max_yaw_diff_rad) {
        max_match_distance_ = max_dist_mm;
        max_match_yaw_diff_ = max_yaw_diff_rad;
    }

    using State       = RobotEKF::MatrixX1;
    using Measurement = RobotEKF::MatrixZ1;

    enum class TrackState
    {
        LOST,
        DETECTING,
        TRACKING,
        TEMP_LOST
    };

    explicit Tracker(double dt, const EKFParams &params);

    void setDt(double dt);
    void setMotionModel(MotionModel m);

    // 用第一帧装甲板测量初始化滤波器
    // init_r_mm: 该类型机器人中心到装甲板的水平半径
    // init_dz_mm: 该类型机器人中心到装甲板的高度偏置
    void resetFromArmor(const Measurement &z,
                        double init_r_mm,
                        double init_dz_mm);

    // 纯预测一步
    State predict();

    // 带量测更新
    State update(const Measurement &z);

    // 预测未来 t_ahead 秒后的【中心位置 + yaw】
    // 返回 [xc, yc, zc, yaw]^T
    Eigen::Matrix<double, 4, 1> predictAhead(double t_ahead) const;

    // 4装甲跳变处理（在 predict/update 前调用）
    void handleArmorJump(double measured_yaw,
                     const Eigen::Vector3d &measured_pos);
    // 由状态反解“当前这块装甲”的三维位置（便于匹配/调试）
    static Eigen::Vector3d armorPositionFromState(const State& x, int offset_sign = OFFSET_SIGN);

    const State &state() const { return x_; }
    TrackState trackState() const { return state_flag_; }

    

private:
    std::unique_ptr<RobotEKF> ekf_;
    EKFParams params_;

    State x_ {};              // 当前后验状态
    double dt_ {};            // 时间步长
    MotionModel model_;       // 当前使用的运动模型
    TrackState state_flag_;   // 状态机

    int detect_cnt_ = 0;      // 连续检测到的帧数
    int lost_cnt_   = 0;      // 连续丢失的帧数

    const int tracking_thres_ = 5;  // 从 DETECTING 进入 TRACKING 所需的连续检测帧数 
    const int lost_thres_     = 60; // 从 TRACKING 进入 LOST 所需的连续丢失帧数 

    // 四装甲相关与匹配阈值
    ArmorsNum armors_num_{ArmorsNum::NORMAL_4};

    // 另一块装甲的半径
    double another_r_{0.0};
    // 两层装甲的相对高度差幅值
    double d_za_{0.0};
    // “当前使用”的中心->装甲高度偏置
    double d_zc_{0.0};
    // 上一时刻的 yaw
    double last_yaw_{0.0};

    // 匹配门限
    double max_match_distance_{350.0}; // mm
    double max_match_yaw_diff_{0.40};  // rad

    };

} // namespace armor_ekf