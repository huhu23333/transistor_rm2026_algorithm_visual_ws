#include "EKF/Tracker.h"
#include <algorithm>
#include <cmath>

namespace armor_ekf
{

Tracker::Tracker(double dt, const EKFParams &params)
    : params_(params),
      dt_(dt),
      model_(MotionModel::CONSTANT_VEL_ROT),
      state_flag_(TrackState::LOST)
{
    // Q 更新函数
    RobotEKF::UpdateQFunc updateQ = [this]() {
        RobotEKF::MatrixXX Q = RobotEKF::MatrixXX::Zero();
        const double t = std::max(dt_, 1e-3); // 这里的 dt_ 是秒

        auto put_block = [&](int p, int v, double s2) {
            // 使用 t^3 和 t^2，而不是 t^4
            Q(p, p) += 0.3333 * t * t * t * s2;  // 位置方差 ~ 1/3 * dt^3 * sigma^2
            Q(p, v) += 0.5    * t * t * s2;      // 协方差   ~ 1/2 * dt^2 * sigma^2
            Q(v, p) += 0.5    * t * t * s2;
            Q(v, v) +=          t * s2;          // 速度方差 ~ dt * sigma^2
        };

        put_block(0, 1, params_.s2qx);   
        put_block(2, 3, params_.s2qy);   
        put_block(4, 5, params_.s2qz);   
        put_block(6, 7, params_.s2qyaw); 

        // r 和 dz 的动态通常较慢，用一阶积分即可
        Q(8, 8) = t * 1e-5;
        //Q(9, 9) = t * 1e-5;

        return Q;
    };

    // R 更新函数：使用固定测量噪声（params_ 里保存的是“标准差”）
    RobotEKF::UpdateRFunc updateR = [this](const RobotEKF::MatrixZ1 &) {
        RobotEKF::MatrixZZ R = RobotEKF::MatrixZZ::Zero();

        const double sx   = params_.r_x;
        const double sy   = params_.r_y;
        const double sz   = params_.r_z;
        const double syaw = params_.r_yaw;

        R(0,0) = sx   * sx;   // x 方差 (mm^2)
        R(1,1) = sy   * sy;   // y
        R(2,2) = sz   * sz;   // z
        R(3,3) = syaw * syaw; // yaw 方差 (rad^2)
        return R;
    };




    // 初始协方差
    RobotEKF::MatrixXX P0 = RobotEKF::MatrixXX::Identity() * params_.p0;

    ekf_ = std::make_unique<RobotEKF>(
        Predict(dt_, model_),   // 过程模型
        Measure{},              // 量测模型
        updateQ,
        updateR,
        P0);

    x_.setZero();
    ekf_->setState(x_);
}

void Tracker::setDt(double dt)
{
    dt_ = dt;
    ekf_->setPredictFunc(Predict(dt_, model_));
}

void Tracker::setMotionModel(MotionModel m)
{
    model_ = m;
    ekf_->setPredictFunc(Predict(dt_, model_));
}

void Tracker::resetFromArmor(const Measurement &z, double init_r_mm)
{
    const double xa  = z(0);
    const double ya  = z(1);
    const double za  = z(2);
    const double yaw = z(3);

    const double r  = init_r_mm;

    const double c = std::cos(yaw);
    const double s = std::sin(yaw);

    const double xc = xa - OFFSET_SIGN * r * c;
    const double yc = ya + OFFSET_SIGN * r * s;
    const double zc = za;

    x_.setZero();
    x_(0) = xc;
    x_(2) = yc;
    x_(4) = zc;
    x_(6) = yaw;
    x_(8) = r;                

    ekf_->setState(x_);
    ekf_->setP(RobotEKF::MatrixXX::Identity() * params_.p0);

    another_r_ = r;
    last_yaw_  = yaw;

    state_flag_ = TrackState::DETECTING;
    detect_cnt_ = 0;
    lost_cnt_   = 0;
}


Tracker::State Tracker::predict()
{
    if (!ekf_)
        return x_;

    x_ = ekf_->predict();

    if (state_flag_ == TrackState::DETECTING ||
        state_flag_ == TrackState::TRACKING)
    {
        if (++detect_cnt_ > tracking_thres_)
            state_flag_ = TrackState::TRACKING;
    }
    else if (state_flag_ == TrackState::TEMP_LOST)
    {
        if (++lost_cnt_ > lost_thres_)
            state_flag_ = TrackState::LOST;
    }

    return x_;
}

Tracker::State Tracker::update(const Measurement &z)
{
    if (!ekf_)
        return x_;

    // 直接用测量：只保证 yaw 本身在 [-pi, pi]，不要根据 pred_yaw 去改它
    Measurement z_proc = z;
    double meas_yaw = z(3);
    z_proc(3) = std::atan2(std::sin(meas_yaw), std::cos(meas_yaw));

    // 使用处理过的测量更新 EKF
    x_ = ekf_->update(z_proc);

    double &yaw  = x_(6);
    double &vyaw = x_(7);
    double &r    = x_(8);
    //double &dz   = x_(9);

    // 1) yaw wrap 到 [-pi, pi]
    yaw = std::atan2(std::sin(yaw), std::cos(yaw));

    // 2) 限制角速度（按实测陀螺速度调）
    const double max_vyaw = 6.0;   // 或 8.0，看录像
    vyaw = std::clamp(vyaw, -max_vyaw, max_vyaw);

    // 3) 限制半径 / 高度偏置
    r  = std::clamp(r, 150.0, 380.0);   // mm，按车几何调;
    //dz = std::clamp(dz, -80.0, 80.0);   // mm，两层高度差预留空间

    ekf_->setState(x_);

    // 有量测就认为还没彻底丢
    lost_cnt_ = 0;
    if (state_flag_ == TrackState::LOST)
        state_flag_ = TrackState::DETECTING;

    return x_;
}


Eigen::Matrix<double, 4, 1> Tracker::predictAhead(double t_ahead) const
{
    Eigen::Matrix<double, 4, 1> out;
    out.setZero();

    if (!ekf_)
        return out;

    const auto &x = x_;

    double xc   = x(0) + x(1) * t_ahead;
    double yc   = x(2) + x(3) * t_ahead;
    double zc   = x(4) + x(5) * t_ahead;
    double yaw  = x(6) + x(7) * t_ahead;

    // wrap 到 [-pi, pi]
    yaw = std::atan2(std::sin(yaw), std::cos(yaw));

    out << xc, yc, zc, yaw;
    return out;
}

Eigen::Vector3d Tracker::armorPositionFromState(const State& x, int offset_sign)
{
    const double xc  = x(0), yc = x(2), zc = x(4);
    const double yaw = x(6), r  = x(8);// dz = x(9);

    const double c = std::cos(yaw);
    const double s = std::sin(yaw);

    const double xa = xc + offset_sign * r * s;
    const double ya = yc - offset_sign * r * c;
    const double za = zc;//+ dz;
    return {xa, ya, za};
}

void Tracker::handleArmorJump(double measured_yaw,
                              const Eigen::Vector3d &measured_pos)
{
    // 1) 计算角度差 (保持原样)
    const double pred_yaw = x_(6);
    double dyaw = measured_yaw - pred_yaw;
    dyaw = std::atan2(std::sin(dyaw), std::cos(dyaw));

    bool is_switching = false;

    // 2) 处理 Yaw 跳变
    if (std::abs(dyaw) > max_match_yaw_diff_) {
        is_switching = true;

        if (armors_num_ == ArmorsNum::NORMAL_4) {
            d_za_ = (x_(4) + x_(9)) - measured_pos.z();
            
            // 【修正 3】安全交换半径
            if (another_r_ > 100.0) std::swap(x_(8), another_r_);
            else another_r_ = x_(8);

            ///d_zc_ = (std::abs(d_zc_) < 1e-6) ? (-d_za_) : 0.0;
            //x_(9) = d_zc_;
        }

        x_(6) = measured_yaw;
        // 【修正 2】不要清零角速度！
        // x_(7) = 0.0;  <-- 删除这行
        ekf_->setState(x_);
    }

    // 3) 几何一致性检查
    const Eigen::Vector3d infer_pos = armorPositionFromState(x_, OFFSET_SIGN);
    
    // 如果是切换状态，或者距离误差实在太大
    if (is_switching || (infer_pos - measured_pos).norm() > max_match_distance_) {
        
        // 利用观测位置 + EKF的半径 强行拉回中心
        const double xa  = measured_pos.x();
        const double ya  = measured_pos.y();
        const double za  = measured_pos.z();
        const double yaw = x_(6);
        const double r   = x_(8); // 信任滤波器收敛的半径
        //const double dz  = x_(9);
        const double c   = std::cos(yaw);
        const double s   = std::sin(yaw);

        // 反解中心
        const double xc = xa - OFFSET_SIGN * r * s;
        const double yc = ya + OFFSET_SIGN * r * c;
        const double zc = za;// - dz;

        // 【修正 1】只修正位置，保留速度
        x_(0) = xc; 
        // x_(1) = 0.0; <-- 删除
        x_(2) = yc; 
        // x_(3) = 0.0; <-- 删除
        x_(4) = zc; 
        // x_(5) = 0.0; <-- 删除
        
        // x_(7) = 0.0; <-- 删除
        
        ekf_->setState(x_);
    }
    
    last_yaw_ = measured_yaw;
}




} // namespace armor_ekf