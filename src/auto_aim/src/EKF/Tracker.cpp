// #include "EKF/Tracker.h"
// #include <algorithm>
// #include <cmath>

// namespace armor_ekf
// {

// // 辅助函数：计算最短角度差 (如果没有引入 external 库，可以手动加在 Tracker.cpp 顶部)
// static double shortest_angular_distance(double from, double to) {
//     double angle = to - from;
//     while (angle <= -M_PI) angle += 2 * M_PI;
//     while (angle > M_PI) angle -= 2 * M_PI;
//     return angle;
// }

// Tracker::Tracker(double dt, const EKFParams &params)
//     : params_(params),
//       dt_(dt),
//       model_(MotionModel::CONSTANT_VEL_ROT),
//       state_flag_(TrackState::LOST)
// {
//     // Q 更新函数
//     RobotEKF::UpdateQFunc updateQ = [this]() {
//         RobotEKF::MatrixXX Q = RobotEKF::MatrixXX::Zero();
//         const double t = dt_;// 这里的 dt_ 是秒

//         auto put_block = [&](int p, int v, double s2) {
//             // 使用 t^3 和 t^2，而不是 t^4
//             Q(p, p) += 0.3333 * t * t * t * s2;  // 位置方差 ~ 1/3 * dt^3 * sigma^2
//             Q(p, v) += 0.5    * t * t * s2;      // 协方差   ~ 1/2 * dt^2 * sigma^2
//             Q(v, p) += 0.5    * t * t * s2;
//             Q(v, v) +=          t * s2;          // 速度方差 ~ dt * sigma^2
//         };

//         put_block(0, 1, params_.s2qx);   
//         put_block(2, 3, params_.s2qy);   
//         put_block(4, 5, params_.s2qz);   
//         put_block(6, 7, params_.s2qyaw); 

//         // r 和 dz 的动态通常较慢，用一阶积分即可
//         Q(8, 8) = t * params_.s2qr;
//         Q(9, 9) = t * params_.s2qdz;

//         return Q;
//     };

//     // R 更新函数：使用固定测量噪声（params_ 里保存的是“标准差”）
//     RobotEKF::UpdateRFunc updateR = [this](const RobotEKF::MatrixZ1 &) {
//         RobotEKF::MatrixZZ R = RobotEKF::MatrixZZ::Zero();

//         const double sx   = params_.r_x;
//         const double sy   = params_.r_y;
//         const double sz   = params_.r_z;
//         const double syaw = params_.r_yaw;

//         R(0,0) = sx   * sx;   // x 方差 (mm^2)
//         R(1,1) = sy   * sy;   // y
//         R(2,2) = sz   * sz;   // z
//         R(3,3) = syaw * syaw; // yaw 方差 (rad^2)
//         return R;
//     };




//     // 初始协方差
//     RobotEKF::MatrixXX P0 = RobotEKF::MatrixXX::Identity() * params_.p0;

//     ekf_ = std::make_unique<RobotEKF>(
//         Predict(dt_, model_),   // 过程模型
//         Measure{},              // 量测模型
//         updateQ,
//         updateR,
//         P0);

//     x_.setZero();
//     ekf_->setState(x_);
// }

// void Tracker::setDt(double dt)
// {
//     dt_ = dt;
//     ekf_->setPredictFunc(Predict(dt_, model_));
// }

// void Tracker::setMotionModel(MotionModel m)
// {
//     model_ = m;
//     ekf_->setPredictFunc(Predict(dt_, model_));
// }

// void Tracker::resetFromArmor(const Measurement &z,
//                              double init_r_mm,
//                              double init_dz_mm)
// {
//     // z: [xa, ya, za, yaw]
//     const double xa  = z(0);
//     const double ya  = z(1);
//     const double za  = z(2);
//     const double yaw = z(3);

//     const double r  = init_r_mm;
//     const double dz = init_dz_mm;

//     const double c = std::cos(yaw);
//     const double s = std::sin(yaw);

//     const double xc = xa - OFFSET_SIGN * r * s;
//     const double yc = ya + OFFSET_SIGN * r * c;
//     const double zc = za - dz;

//     x_.setZero();
//     x_(0) = xc;
//     x_(2) = yc;
//     x_(4) = zc;
//     x_(6) = yaw;
//     x_(8) = r;
//     x_(9) = 0.0;

//     ekf_->setState(x_);
//     ekf_->setP(RobotEKF::MatrixXX::Identity() * params_.p0);

//     another_r_ = r;
//     d_za_      = 0.0;
//     d_zc_      = dz;
//     last_yaw_  = yaw;

//     state_flag_ = TrackState::DETECTING;
//     detect_cnt_ = 0;
//     lost_cnt_   = 0;
// }


// Tracker::State Tracker::predict()
// {
//     if (!ekf_)
//         return x_;

//     x_ = ekf_->predict();

//     if (state_flag_ == TrackState::DETECTING ||
//         state_flag_ == TrackState::TRACKING)
//     {
//         if (++detect_cnt_ > tracking_thres_)
//             state_flag_ = TrackState::TRACKING;
//     }
//     else if (state_flag_ == TrackState::TEMP_LOST)
//     {
//         if (++lost_cnt_ > lost_thres_)
//             state_flag_ = TrackState::LOST;
//     }

//     return x_;
// }



// Tracker::State Tracker::update(const Measurement &z)
// {
//     if (!ekf_)
//         return x_;

//     Measurement z_proc = z;
//     double meas_yaw = z(3);
//     double pred_yaw = x_(6);
    
//     // 1. 解缠绕：计算最短角度差
//     double yaw_diff = shortest_angular_distance(pred_yaw, meas_yaw);

//     // 2. 设定阈值 (单位必须统一为 mm)
//     const double MAX_YAW_ERROR = 1.5;
//     const double MAX_POS_ERROR = 3600.0;

//     // 3. 计算位置偏差
//     Eigen::Vector3d pred_pos = armorPositionFromState(x_, OFFSET_SIGN);
//     double dist_diff = (z_proc.head<3>() - pred_pos).norm();


//     // 检查是否未初始化（当前位置几乎为0）
//     bool is_uninitialized = (state_flag_ == TrackState::LOST);

//     // 4. 异常值拒绝
//     bool is_outlier = false;
    
//     // 如果已经初始化，进行严格的波门检查
//     if (!is_uninitialized) { 
//         if (std::abs(yaw_diff) > MAX_YAW_ERROR || dist_diff > MAX_POS_ERROR) {
//             is_outlier = true;
//             std::cout << "Outlier! YawDiff: " << yaw_diff << " DistDiff: " << dist_diff << std::endl;
//         }
//     }
   
//     if (is_outlier) {
//         lost_cnt_++;
//         if (lost_cnt_ > lost_thres_) state_flag_ = TrackState::LOST;
//         return x_; // 拒绝更新，直接返回预测值
//     }

//     // 5. 准备更新数据
//     z_proc(3) = pred_yaw + yaw_diff; // 将测量 Yaw 拉到预测 Yaw 的周期内

//     // 6. EKF 更新
//     x_ = ekf_->update(z_proc);

//     // 7. 后处理限制
//     double &yaw  = x_(6);
//     double &vyaw = x_(7);
//     double &r    = x_(8);
//     double &dz   = x_(9);

//     yaw = std::atan2(std::sin(yaw), std::cos(yaw)); // 归一化 yaw
    
//     // 限制角速度，避免发散
//     const double max_vyaw = 12.0; 
//     vyaw = std::clamp(vyaw, -max_vyaw, max_vyaw);

//     // 限制半径
//     r  = std::clamp(r, 150.0, 400.0);   
//     dz = std::clamp(dz, -100.0, 100.0); 

//     ekf_->setState(x_);

//     lost_cnt_ = 0;
//     if (state_flag_ == TrackState::LOST)
//         state_flag_ = TrackState::DETECTING;

//     return x_;
// }


// Eigen::Matrix<double, 4, 1> Tracker::predictAhead(double t_ahead) const
// {
//     Eigen::Matrix<double, 4, 1> out;
//     out.setZero();

//     if (!ekf_)
//         return out;

//     const auto &x = x_;

//     double xc   = x(0) + x(1) * t_ahead;
//     double yc   = x(2) + x(3) * t_ahead;
//     double zc   = x(4) + x(5) * t_ahead;
//     double yaw  = x(6) + x(7) * t_ahead;

//     // wrap 到 [-pi, pi]
//     yaw = std::atan2(std::sin(yaw), std::cos(yaw));

//     out << xc, yc, zc, yaw;
//     return out;
// }

// Eigen::Vector3d Tracker::armorPositionFromState(const State& x, int offset_sign)
// {
//     const double xc  = x(0), yc = x(2), zc = x(4);
//     const double yaw = x(6), r  = x(8), dz = x(9);

//     const double c = std::cos(yaw);
//     const double s = std::sin(yaw);

//     const double xa = xc + offset_sign * r * s;
//     const double ya = yc - offset_sign * r * c;
//     const double za = zc + dz;
//     return {xa, ya, za};
// }
// /*
// void Tracker::handleArmorJump(double measured_yaw,
//                               const Eigen::Vector3d &measured_pos)
// {
//     // 1) 计算角度差 (保持原样)
//     const double pred_yaw = x_(6);
//     double dyaw = measured_yaw - pred_yaw;
//     dyaw = std::atan2(std::sin(dyaw), std::cos(dyaw));

//     bool is_switching = false;

//     // 2) 处理 Yaw 跳变
//     if (std::abs(dyaw) > max_match_yaw_diff_) {
//         is_switching = true;

//         if (armors_num_ == ArmorsNum::NORMAL_4) {
//             d_za_ = (x_(4) + x_(9)) - measured_pos.z();
            
//             if (another_r_ > 100.0) std::swap(x_(8), another_r_);
//             else another_r_ = x_(8);

//             //d_zc_ = (std::abs(d_zc_) < 1e-6) ? (-d_za_) : 0.0;
//             x_(9) = (std::abs(x_(9)) < 1e-3) ? -d_za_ : 0.0;
//         }

//         x_(6) = measured_yaw;
        
//         ekf_->setState(x_);
//     }

//     // 3) 几何一致性检查
//     const Eigen::Vector3d infer_pos = armorPositionFromState(x_, OFFSET_SIGN);
    
//     // 如果是切换状态，或者距离误差实在太大
//     if (is_switching || (infer_pos - measured_pos).norm() > max_match_distance_) {
        
//         // 利用观测位置 + EKF的半径 强行拉回中心
//         const double xa  = measured_pos.x();
//         const double ya  = measured_pos.y();
//         const double za  = measured_pos.z();
//         const double yaw = x_(6);
//         const double r   = x_(8);
//         const double dz  = x_(9);
//         const double c   = std::cos(yaw);
//         const double s   = std::sin(yaw);

//         // 反解中心
//         const double xc = xa - OFFSET_SIGN * r * s;
//         const double yc = ya + OFFSET_SIGN * r * c;
//         const double zc = za - dz;

//         // 当位置偏差过大时，必须清零速度，否则 EKF 会带着巨大的错误速度继续发散
//         x_(0) = xc; 
//         x_(1) = 0.0;
//         x_(2) = yc; 
//         x_(3) = 0.0;
//         x_(4) = zc; 
//         x_(5) = 0.0;
        
//         // 角速度通常可以保留，对陀螺的信任程度
//         // x_(7) = 0.0;
//         ekf_->setState(x_);
//     }
    
//     last_yaw_ = measured_yaw;
// }
// */
// void Tracker::handleArmorJump(double measured_yaw,
//                               const Eigen::Vector3d &measured_pos)
// {
//     // 1) 当前预测 yaw
//     const double pred_yaw = x_(6);
//     double dyaw = measured_yaw - pred_yaw;
//     dyaw = std::atan2(std::sin(dyaw), std::cos(dyaw));

//     bool is_switching = false;

//     // 2) 判断是否跳装甲
//     if (std::abs(dyaw) > max_match_yaw_diff_) {
//         is_switching = true;

//         // 4 装甲时，只做半径切换，不再动 dz
//         if (armors_num_ == ArmorsNum::NORMAL_4) {
//             if (another_r_ > 100.0) {
//                 std::swap(x_(8), another_r_);
//             } else {
//                 another_r_ = x_(8);
//             }
//         }

//         // 直接把 yaw 拉到观测
//         x_(6) = measured_yaw;
//         ekf_->setState(x_);
//     }

//     // 3) 用当前状态推一下一块装甲的位置
//     const Eigen::Vector3d infer_pos = armorPositionFromState(x_, OFFSET_SIGN);

//     // 4) 如果刚跳装甲，或者距离偏差太大，就用观测 + (r, dz) 重新反解中心
//     if (is_switching || (infer_pos - measured_pos).norm() > max_match_distance_) {
//         const double xa  = measured_pos.x();
//         const double ya  = measured_pos.y();
//         const double za  = measured_pos.z();
//         const double yaw = x_(6);
//         const double r   = x_(8);
//         const double dz  = x_(9);

//         const double c = std::cos(yaw);
//         const double s = std::sin(yaw);

//         const double xc = xa - OFFSET_SIGN * r * s;
//         const double yc = ya + OFFSET_SIGN * r * c;
//         const double zc = za - dz;

//         // 位置重置 + 速度清零
//         x_(0) = xc; x_(1) = 0.0;
//         x_(2) = yc; x_(3) = 0.0;
//         x_(4) = zc; x_(5) = 0.0;

//         ekf_->setState(x_);
//     }

//     last_yaw_ = measured_yaw;
// }




// } // namespace armor_ekf






#include "EKF/Tracker.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace armor_ekf
{

// 辅助函数：计算最短角度差
static double shortest_angular_distance(double from, double to)
{
    double angle = to - from;
    while (angle <= -M_PI) angle += 2.0 * M_PI;
    while (angle >  M_PI)  angle -= 2.0 * M_PI;
    return angle;
}

Tracker::Tracker(double dt, const EKFParams &params)
    : params_(params),
      dt_(dt),
      model_(MotionModel::CONSTANT_VEL_ROT),
      state_flag_(TrackState::LOST)
{
    // -------- 位置 EKF 的 Q 更新函数 --------
    PosEKF::UpdateQFunc posUpdateQ = [this]() {
        PosEKF::MatrixXX Q = PosEKF::MatrixXX::Zero();
        const double t = dt_; // 秒

        auto put_block = [&](int p, int v, double s2) {
            // 常速度离散化
            Q(p, p) += 0.3333 * t * t * t * s2;  // 位置方差 ~ 1/3 * dt^3 * sigma^2
            Q(p, v) += 0.5    * t * t     * s2;  // 协方差   ~ 1/2 * dt^2 * sigma^2
            Q(v, p) += 0.5    * t * t     * s2;
            Q(v, v) +=          t         * s2;  // 速度方差 ~ dt * sigma^2
        };

        put_block(0, 1, params_.s2qx); // xc, vxc
        put_block(2, 3, params_.s2qy); // yc, vyc
        put_block(4, 5, params_.s2qz); // zc, vzc

        return Q;
    };

    // -------- 位置 EKF 的 R 更新函数 --------
    PosEKF::UpdateRFunc posUpdateR = [this](const PosEKF::MatrixZ1 &) {
        PosEKF::MatrixZZ R = PosEKF::MatrixZZ::Zero();
        R(0,0) = params_.r_x * params_.r_x;
        R(1,1) = params_.r_y * params_.r_y;
        R(2,2) = params_.r_z * params_.r_z;
        return R;
    };

    // -------- 几何 EKF 的 Q 更新函数 --------
    GeoEKF::UpdateQFunc geoUpdateQ = [this]() {
        GeoEKF::MatrixXX Q = GeoEKF::MatrixXX::Zero();
        const double t = dt_;

        auto put_block = [&](int p, int v, double s2) {
            Q(p, p) += 0.3333 * t * t * t * s2;
            Q(p, v) += 0.5    * t * t     * s2;
            Q(v, p) += 0.5    * t * t     * s2;
            Q(v, v) +=          t         * s2;
        };

        // yaw, vyaw
        put_block(0, 1, params_.s2qyaw);

        // r, dz — 近似常值，一阶积分噪声
        Q(2,2) = t * 1e-6;
        Q(3,3) = t * 1e-6;

        return Q;
    };

    // -------- 几何 EKF 的 R 更新函数 --------
    GeoEKF::UpdateRFunc geoUpdateR = [this](const GeoEKF::MatrixZ1 &) {
        GeoEKF::MatrixZZ R = GeoEKF::MatrixZZ::Zero();
        const double syaw = params_.r_yaw;
        const double sr   = 0.5 * (params_.r_x + params_.r_y);
        const double sdz  = params_.r_z;

        R(0,0) = syaw * syaw; // yaw
        R(1,1) = sr   * sr;   // r
        R(2,2) = sdz  * sdz;  // dz

        return R;
    };

    // 初始协方差
    PosEKF::MatrixXX P0_pos = PosEKF::MatrixXX::Identity() * params_.p0;
    GeoEKF::MatrixXX P0_geo = GeoEKF::MatrixXX::Identity() * params_.p0;

    pos_ekf_ = std::make_unique<PosEKF>(
        PosPredict(dt_, model_),
        PosMeasure{},
        posUpdateQ,
        posUpdateR,
        P0_pos);

    geo_ekf_ = std::make_unique<GeoEKF>(
        GeoPredict(dt_),
        GeoMeasure{},
        geoUpdateQ,
        geoUpdateR,
        P0_geo);

    x_pos_.setZero();
    x_geo_.setZero();
    syncFullFromSub();

    pos_ekf_->setState(x_pos_);
    geo_ekf_->setState(x_geo_);
}

void Tracker::setDt(double dt)
{
    dt_ = dt;
    if (pos_ekf_) {
        pos_ekf_->setPredictFunc(PosPredict(dt_, model_));
    }
    if (geo_ekf_) {
        geo_ekf_->setPredictFunc(GeoPredict(dt_));
    }
}

void Tracker::setMotionModel(MotionModel m)
{
    model_ = m;
    if (pos_ekf_) {
        pos_ekf_->setPredictFunc(PosPredict(dt_, model_));
    }
}

void Tracker::syncFullFromSub()
{
    x_full_.setZero();
    // 位置
    x_full_(0) = x_pos_(0);
    x_full_(1) = x_pos_(1);
    x_full_(2) = x_pos_(2);
    x_full_(3) = x_pos_(3);
    x_full_(4) = x_pos_(4);
    x_full_(5) = x_pos_(5);
    // 几何
    x_full_(6) = x_geo_(0);
    x_full_(7) = x_geo_(1);
    x_full_(8) = x_geo_(2);
    x_full_(9) = x_geo_(3);
}

void Tracker::resetFromArmor(const Measurement &z,
                             double init_r_mm,
                             double init_dz_mm)
{
    const double xa  = z(0);
    const double ya  = z(1);
    const double za  = z(2);
    const double yaw = z(3);

    const double r  = init_r_mm;
    const double dz = init_dz_mm;

    const double c = std::cos(yaw);
    const double s = std::sin(yaw);

    const double xc = xa - OFFSET_SIGN * r * s;
    const double yc = ya + OFFSET_SIGN * r * c;
    const double zc = za - dz;

    // 初始化位置 EKF
    x_pos_.setZero();
    x_pos_(0) = xc;
    x_pos_(2) = yc;
    x_pos_(4) = zc;

    // 初始化几何 EKF
    x_geo_.setZero();
    x_geo_(0) = yaw;
    x_geo_(2) = r;
    x_geo_(3) = dz;

    PosEKF::MatrixXX P0_pos = PosEKF::MatrixXX::Identity() * params_.p0;
    GeoEKF::MatrixXX P0_geo = GeoEKF::MatrixXX::Identity() * params_.p0;

    pos_ekf_->setState(x_pos_);
    geo_ekf_->setState(x_geo_);
    pos_ekf_->setP(P0_pos);
    geo_ekf_->setP(P0_geo);

    syncFullFromSub();

    another_r_ = r;
    d_za_      = 0.0;
    d_zc_      = dz;
    last_yaw_  = yaw;

    state_flag_ = TrackState::DETECTING;
    detect_cnt_ = 0;
    lost_cnt_   = 0;
}

Tracker::State Tracker::predict()
{
    if (!pos_ekf_ || !geo_ekf_)
        return x_full_;

    x_pos_ = pos_ekf_->predict();
    x_geo_ = geo_ekf_->predict();
    syncFullFromSub();

    if (state_flag_ == TrackState::DETECTING ||
        state_flag_ == TrackState::TRACKING)
    {
        if (++detect_cnt_ > tracking_thres_) {
            state_flag_ = TrackState::TRACKING;
        }
    }
    else if (state_flag_ == TrackState::TEMP_LOST)
    {
        if (++lost_cnt_ > lost_thres_) {
            state_flag_ = TrackState::LOST;
        }
    }

    return x_full_;
}

Tracker::State Tracker::update(const Measurement &z)
{
    if (!pos_ekf_ || !geo_ekf_)
        return x_full_;

    const double xa       = z(0);
    const double ya       = z(1);
    const double za       = z(2);
    const double meas_yaw = z(3);

    // 1. 波门：用预测装甲位置和 yaw
    Eigen::Vector3d meas_pos(xa, ya, za);
    Eigen::Vector3d pred_pos = armorPositionFromState(x_full_, OFFSET_SIGN);

    double yaw_pred  = x_geo_(0);
    double yaw_diff  = shortest_angular_distance(yaw_pred, meas_yaw);
    double dist_diff = (meas_pos - pred_pos).norm();

    const double MAX_YAW_ERROR = 1.5;
    const double MAX_POS_ERROR = 500.0;

    bool is_uninitialized = (state_flag_ == TrackState::LOST);
    bool is_outlier       = false;

    if (!is_uninitialized) {
        if (std::abs(yaw_diff) > MAX_YAW_ERROR || dist_diff > MAX_POS_ERROR) {
            is_outlier = true;
            std::cout << "EKF Outlier! YawDiff: " << yaw_diff
                      << " DistDiff: " << dist_diff << std::endl;
        }
    }

    if (is_outlier) {
        ++lost_cnt_;
        if (lost_cnt_ > lost_thres_) {
            state_flag_ = TrackState::LOST;
        }
        return x_full_;
    }

    // 把 yaw 测量拉到预测 yaw 的同一周期
    double yaw_for_update = yaw_pred + yaw_diff;

    // 2. 更新位置 EKF：用当前几何状态，把装甲测量反解成中心测量
    const double r_use  = x_geo_(2);
    const double dz_use = x_geo_(3);

    const double c = std::cos(yaw_for_update);
    const double s = std::sin(yaw_for_update);

    const double xc_meas = xa - OFFSET_SIGN * r_use * s;
    const double yc_meas = ya + OFFSET_SIGN * r_use * c;
    const double zc_meas = za - dz_use;

    PosEKF::MatrixZ1 z_pos;
    z_pos << xc_meas, yc_meas, zc_meas;
    x_pos_ = pos_ekf_->update(z_pos);

    // 3. 更新几何 EKF：从“更新后的中心 + 装甲位置”反解 r / dz
    const double xc = x_pos_(0);
    const double yc = x_pos_(2);
    const double zc = x_pos_(4);

    const double dx = xa - xc;
    const double dy = ya - yc;
    const double r_meas  = std::sqrt(dx * dx + dy * dy);
    const double dz_meas = za - zc;

    GeoEKF::MatrixZ1 z_geo;
    z_geo << yaw_for_update, r_meas, dz_meas;
    x_geo_ = geo_ekf_->update(z_geo);

    // 4. 几何后处理
    double &yaw  = x_geo_(0);
    double &vyaw = x_geo_(1);
    double &r    = x_geo_(2);
    double &dz   = x_geo_(3);

    yaw = std::atan2(std::sin(yaw), std::cos(yaw)); // 归一化 yaw

    // 限制角速度，避免发散
    const double max_vyaw = 15.0;
    if (vyaw >  max_vyaw) vyaw =  max_vyaw;
    if (vyaw < -max_vyaw) vyaw = -max_vyaw;

    // 限制半径 / 高度偏置
    if (r  < 200.0) r  = 200.0;
    if (r  > 400.0) r  = 400.0;
    if (dz < -50.0) dz = -50.0;
    if (dz >  50.0) dz =  50.0;

    pos_ekf_->setState(x_pos_);
    geo_ekf_->setState(x_geo_);

    syncFullFromSub();

    lost_cnt_ = 0;
    if (state_flag_ == TrackState::LOST) {
        state_flag_ = TrackState::DETECTING;
    }

    return x_full_;
}

Eigen::Matrix<double, 4, 1> Tracker::predictAhead(double t_ahead) const
{
    Eigen::Matrix<double, 4, 1> out;
    out.setZero();

    if (!pos_ekf_ || !geo_ekf_)
        return out;

    // 线性外推中心和平移
    double xc  = x_pos_(0) + x_pos_(1) * t_ahead;
    double yc  = x_pos_(2) + x_pos_(3) * t_ahead;
    double zc  = x_pos_(4) + x_pos_(5) * t_ahead;
    double yaw = x_geo_(0) + x_geo_(1) * t_ahead;

    yaw = std::atan2(std::sin(yaw), std::cos(yaw));

    out << xc, yc, zc, yaw;
    return out;
}

Eigen::Vector3d Tracker::armorPositionFromState(const State& x, int offset_sign)
{
    const double xc  = x(0);
    const double yc  = x(2);
    const double zc  = x(4);
    const double yaw = x(6);
    const double r   = x(8);
    const double dz  = x(9);

    const double c = std::cos(yaw);
    const double s = std::sin(yaw);

    const double xa = xc + offset_sign * r * s;
    const double ya = yc - offset_sign * r * c;
    const double za = zc + dz;

    return Eigen::Vector3d(xa, ya, za);
}

void Tracker::handleArmorJump(double measured_yaw,
                              const Eigen::Vector3d &measured_pos)
{
    if (!pos_ekf_ || !geo_ekf_)
        return;

    // 1) 计算角度差
    const double pred_yaw = x_geo_(0);
    double dyaw = measured_yaw - pred_yaw;
    dyaw = std::atan2(std::sin(dyaw), std::cos(dyaw));

    bool is_switching = false;

    // 2) 如果 yaw 跳变太大，认为是切换装甲
    if (std::abs(dyaw) > max_match_yaw_diff_) {
        is_switching = true;

        // 简单处理：四装甲时在两个半径之间切换
        if (armors_num_ == ArmorsNum::NORMAL_4) {
            if (another_r_ > 10.0) {
                std::swap(x_geo_(2), another_r_);
            } else {
                another_r_ = x_geo_(2);
            }
        }

        x_geo_(0) = measured_yaw;
        geo_ekf_->setState(x_geo_);
    }

    syncFullFromSub();
    const Eigen::Vector3d infer_pos = armorPositionFromState(x_full_, OFFSET_SIGN);

    // 3) 几何一致性检查：如果强烈不一致，就直接用观测位置 + 当前几何重置中心
    if (is_switching || (infer_pos - measured_pos).norm() > max_match_distance_) {
        const double xa  = measured_pos.x();
        const double ya  = measured_pos.y();
        const double za  = measured_pos.z();
        const double yaw = x_geo_(0);
        const double r   = x_geo_(2);
        const double dz  = x_geo_(3);

        const double c = std::cos(yaw);
        const double s = std::sin(yaw);

        const double xc = xa - OFFSET_SIGN * r * s;
        const double yc = ya + OFFSET_SIGN * r * c;
        const double zc = za - dz;

        x_pos_(0) = xc; x_pos_(1) = 0.0;
        x_pos_(2) = yc; x_pos_(3) = 0.0;
        x_pos_(4) = zc; x_pos_(5) = 0.0;

        pos_ekf_->setState(x_pos_);
        syncFullFromSub();
    }

    last_yaw_ = measured_yaw;
}

Eigen::Vector3d Tracker::predictBestArmorPosition(double t_ahead, double switch_thres_rad) const
{
    // 1. 获取基础预测 (中心位置 + 当前跟踪ID的yaw)
    Eigen::Matrix<double, 4, 1> pred = predictAhead(t_ahead);
    double xc = pred(0);
    double yc = pred(1);
    double zc = pred(2);
    double yaw_main = pred(3); // 这是当前跟踪的那块板的预测Yaw

    double r  = x_geo_(2);
    double dz = x_geo_(3);

    // 2. 遍历所有装甲板，计算“视角代价”
    int armor_n = static_cast<int>(armors_num_);
    int best_idx = 0;
    double min_score = 1e9;

    // 理想视角 Yaw：相机在原点，目标在(xc, yc)，板应该正对相机
    // 也就是板的法线方向应该是 (0,0) - (xc, yc) 的方向
    double view_yaw = std::atan2(yc, xc) + M_PI;

    for (int i = 0; i < armor_n; ++i) {
        // 计算第 i 块板在 t_ahead 时刻的 yaw
        // i=0 对应当前正在跟踪的那块，i=1 是下一块...
        double yaw_i = yaw_main + i * (2.0 * M_PI / armor_n);
        
        // 计算与理想视角的偏差 (绝对值越小越好)
        double diff = std::abs(shortest_angular_distance(yaw_i, view_yaw));
        
        double score = diff;
        
        // 【核心逻辑】
        // i=0 是当前板。
        // 如果 switch_thres_rad > 0，score 变小，更难被替换（死咬上一块）。
        // 如果 switch_thres_rad < 0，score 变大，更容易被替换（提前预瞄下一块）。
        if (i == 0) {
            score -= switch_thres_rad;
        }

        if (score < min_score) {
            min_score = score;
            best_idx = i;
        }
    }

    // 3. 选中最佳装甲板，计算其三维坐标
    double best_yaw = yaw_main + best_idx * (2.0 * M_PI / armor_n);

    // 根据 motion_model 的几何定义计算位置
    double xa = xc + OFFSET_SIGN * r * std::sin(best_yaw);
    double ya = yc - OFFSET_SIGN * r * std::cos(best_yaw);
    double za = zc + dz;

    return Eigen::Vector3d(xa, ya, za);
}

} // namespace armor_ekf
