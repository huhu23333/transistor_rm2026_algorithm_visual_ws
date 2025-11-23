// #include "armor_detector/Tracker.h"
// #include "rclcpp/rclcpp.hpp"

// Tracker::Tracker(double dt, const EKFParams& params) : dt_(dt), state(LOST) {
//     RobotEKF::UpdateQFunc update_Q = [this, params]() {
//         Eigen::Matrix<double, N_x, N_x> Q = Eigen::Matrix<double, N_x, N_x>::Zero();
//         double t = this->dt_;
//         // 从传入的参数结构体中获取过程噪声
//         double s2qx = params.s2qx, s2qy = params.s2qy, s2qz = params.s2qz, 
//                s2qyaw = params.s2qyaw, s2qr = params.s2qr;// s2qd_yc = params.s2qd_yc;

//         // X方向 (索引 0, 1)
//         double q_x_x=pow(t,4)/4*s2qx, q_x_vx=pow(t,3)/2*s2qx, q_vx_vx=pow(t,2)*s2qx;
//         Q(0,0)=q_x_x; Q(0,1)=q_x_vx; Q(1,0)=q_x_vx; Q(1,1)=q_vx_vx;

//         // Y方向 (索引 2, 3)
//         double q_y_y=pow(t,4)/4*s2qy, q_y_vy=pow(t,3)/2*s2qy, q_vy_vy=pow(t,2)*s2qy;
//         Q(2,2)=q_y_y; Q(2,3)=q_y_vy; Q(3,2)=q_y_vy; Q(3,3)=q_vy_vy;

//         // Z方向 (索引 4, 5)
//         double q_z_z=pow(t,4)/4*s2qz, q_z_vz=pow(t,3)/2*s2qz, q_vz_vz=pow(t,2)*s2qz;
//         Q(4,4)=q_z_z; Q(4,5)=q_z_vz; Q(5,4)=q_z_vz; Q(5,5)=q_vz_vz;

//         // Yaw (索引 6, 7)
//         double q_yaw_yaw=pow(t,4)/4*s2qyaw, q_yaw_vyaw=pow(t,3)/2*s2qyaw, q_vyaw_vyaw=pow(t,2)*s2qyaw;
//         Q(6,6)=q_yaw_yaw; Q(6,7)=q_yaw_vyaw; Q(7,6)=q_yaw_vyaw; Q(7,7)=q_vyaw_vyaw;

//         // r (索引 8)
//         Q(8,8) = pow(t,2)*s2qr;

//         // d_yc (索引 9)
//         Q(9,9) = pow(t,2)*0.01; // 设为一个较小的常数，假设d_yc变化不大

//         return Q;
//     };

//     RobotEKF::UpdateRFunc update_R = [params](const Measurement& z) {
//         RobotEKF::MatrixZZ R;
//         // 从传入的参数结构体中获取测量噪声
//         double r_x_coeff = params.r_x;
//         double r_y_val = params.r_y;
//         double r_z_coeff = params.r_z;
//         double r_yaw_val = params.r_yaw;

//         // 动态噪声模型：x和z的测量不确定性随距离增大
//         R << r_x_coeff * std::abs(z[0]), 0, 0, 0,
//              0, r_y_val, 0, 0,
//              0, 0, r_z_coeff * std::abs(z[2]), 0,
//              0, 0, 0, r_yaw_val;
//         return R;
//     };

//     // 从传入的参数结构体中获取初始协方差
//     RobotEKF::MatrixXX P0 = RobotEKF::MatrixXX::Identity();
//     P0 = P0 * params.p0;
//     //P0(9, 9) = 0.000000001;

//     ekf_ = std::make_unique<RobotEKF>(Predict(dt_), Measure(), update_Q, update_R, P0);
// }

// void Tracker::reset(const Measurement& z) {
//     state = DETECTING;
//     detect_count_ = 0; lost_count_ = 0;
//     State x0 = State::Zero();
    
//     double xa = z(0), ya = z(1), za = z(2), yaw = z(3);
//     double r = 180.0;
    
//     double xc = xa + r * cos(yaw);
//     double zc = za + r * sin(yaw);
//     double yc = ya;
    
//     x0(0) = xc;
//     x0(2) = yc;
//     x0(4) = zc;
//     x0(6) = yaw;
//     x0(8) = r;

//     ekf_->setState(x0);
//     RCLCPP_DEBUG(rclcpp::get_logger("armor_detect_node"), "Tracker RESET!");
// }

// Tracker::State Tracker::predict() {
//     auto state_vec = ekf_->predict();
//     if (state == TRACKING) {
//         lost_count_++;
//         if (lost_count_ > lost_thres_) state = LOST;
//         else state = TEMP_LOST;
//     } else if (state == DETECTING) {
//         lost_count_++;
//         if (lost_count_ > lost_thres_ / 2) state = LOST;
//     }
//     return state_vec;
// }

// Tracker::State Tracker::update(const Measurement& z) {
//     auto state_vec = ekf_->update(z);
    
//     if (state_vec(8) < 120.0) state_vec(8) = 120.0;
//     else if (state_vec(8) > 400.0) state_vec(8) = 400.0;
//     ekf_->setState(state_vec);

//     lost_count_ = 0;
//     if (state == DETECTING) {
//         detect_count_++;
//         if (detect_count_ > tracking_thres_) {
//             state = TRACKING;
//             RCLCPP_DEBUG(rclcpp::get_logger("armor_detect_node"), "Tracker stable: TRACKING");
//         }
//     } else if (state == TEMP_LOST) state = TRACKING;
//     return state_vec;
// }

// Tracker::State Tracker::getTargetState() const { return ekf_->getState(); }

// Eigen::Vector3d Tracker::getArmorPosition() const {
//     State x = getTargetState();
//     double xc=x(0), yc=x(2), zc=x(4), yaw=x(6), r=x(8);
//     return {xc - r * cos(yaw), yc, zc - r * sin(yaw)};
// }

// Tracker::State Tracker::predictAhead(double t_ahead) const {
//     if (t_ahead < 1e-3) return getTargetState();
//     State x_k = getTargetState();
//     Predict pred(t_ahead);
//     State x_final;
//     pred(x_k.data(), x_final.data());
//     return x_final;
// }




#include "EKF/Tracker.h"
#include "rclcpp/rclcpp.hpp"

Tracker::Tracker(double dt, const EKFParams& params) : dt_(dt), state(LOST) {
    RobotEKF::UpdateQFunc update_Q = [this, params]() {
        Eigen::Matrix<double, N_x, N_x> Q = Eigen::Matrix<double, N_x, N_x>::Zero();
        double t = this->dt_;
        
        // 计算Q矩阵元素的系数 (基于离散维纳过程加速模型 DWPA)
        // 噪声源是 Jerk (加加速度)
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t2 * t2;
        double t5 = t2 * t3;

        // Q block matrix coefficients
        // cov(x, x) = dt^5 / 20
        // cov(x, v) = dt^4 / 8
        // cov(x, a) = dt^3 / 6
        // cov(v, v) = dt^3 / 3
        // cov(v, a) = dt^2 / 2
        // cov(a, a) = dt
        
        double q_x_x   = t5 / 20.0, q_x_v   = t4 / 8.0,  q_x_a   = t3 / 6.0;
        double q_v_v   = t3 / 3.0,  q_v_a   = t2 / 2.0;
        double q_a_a   = t;

        // X轴 (Indices 0, 1, 2)
        double sx = params.s2q_ax; // Variance of Jerk X
        Q(0,0) = q_x_x * sx; Q(0,1) = q_x_v * sx; Q(0,2) = q_x_a * sx;
        Q(1,0) = q_x_v * sx; Q(1,1) = q_v_v * sx; Q(1,2) = q_v_a * sx;
        Q(2,0) = q_x_a * sx; Q(2,1) = q_v_a * sx; Q(2,2) = q_a_a * sx;

        // Y轴 (Indices 3, 4, 5)
        double sy = params.s2q_ay;
        Q(3,3) = q_x_x * sy; Q(3,4) = q_x_v * sy; Q(3,5) = q_x_a * sy;
        Q(4,3) = q_x_v * sy; Q(4,4) = q_v_v * sy; Q(4,5) = q_v_a * sy;
        Q(5,3) = q_x_a * sy; Q(5,4) = q_v_a * sy; Q(5,5) = q_a_a * sy;

        // Z轴 (Indices 6, 7, 8)
        double sz = params.s2q_az;
        Q(6,6) = q_x_x * sz; Q(6,7) = q_x_v * sz; Q(6,8) = q_x_a * sz;
        Q(7,6) = q_x_v * sz; Q(7,7) = q_v_v * sz; Q(7,8) = q_v_a * sz;
        Q(8,6) = q_x_a * sz; Q(8,7) = q_v_a * sz; Q(8,8) = q_a_a * sz;

        return Q;
    };

    RobotEKF::UpdateRFunc update_R = [params](const Measurement& z) {
        RobotEKF::MatrixZZ R = RobotEKF::MatrixZZ::Identity();
        // 简单测量噪声模型
        R(0,0) = params.r_x;
        R(1,1) = params.r_y;
        R(2,2) = params.r_z;
        return R;
    };

    // 初始协方差
    RobotEKF::MatrixXX P0 = RobotEKF::MatrixXX::Identity() * params.p0;

    ekf_ = std::make_unique<RobotEKF>(Predict(dt_), Measure(), update_Q, update_R, P0);
}

void Tracker::reset(const Measurement& z) {
    state = DETECTING;
    detect_count_ = 0; lost_count_ = 0;
    
    State x0 = State::Zero();
    // 初始化位置
    x0(Idx::X) = z(0);
    x0(Idx::Y) = z(1);
    x0(Idx::Z) = z(2);
    // 速度和加速度默认初始化为0
    
    ekf_->setState(x0);
    RCLCPP_INFO(rclcpp::get_logger("armor_detect_node"), "Tracker RESET with 9D CA Model!");
}

Tracker::State Tracker::predict() {
    auto state_vec = ekf_->predict();
    if (state == TRACKING) {
        lost_count_++;
        if (lost_count_ > lost_thres_) state = LOST;
        else state = TEMP_LOST;
    } else if (state == DETECTING) {
        lost_count_++;
        if (lost_count_ > lost_thres_ / 2) state = LOST;
    }
    return state_vec;
}

Tracker::State Tracker::update(const Measurement& z) {
    auto state_vec = ekf_->update(z);
    ekf_->setState(state_vec);

    lost_count_ = 0;
    if (state == DETECTING) {
        detect_count_++;
        if (detect_count_ > tracking_thres_) {
            state = TRACKING;
        }
    } else if (state == TEMP_LOST) state = TRACKING;
    return state_vec;
}

Tracker::State Tracker::getTargetState() const { return ekf_->getState(); }

Eigen::Vector3d Tracker::getArmorPosition() const {
    State x = getTargetState();
    return {x(Idx::X), x(Idx::Y), x(Idx::Z)};
}

Tracker::State Tracker::predictAhead(double t_ahead) const {
    if (t_ahead < 1e-4) return getTargetState();
    State x_k = getTargetState();
    Predict pred(t_ahead);
    State x_final;
    pred(x_k.data(), x_final.data());
    return x_final;
}

// // 新增函数的实现
// void Tracker::guideState(const Measurement& z) {
//     // === 状态引导逻辑 ===
//     RCLCPP_WARN(rclcpp::get_logger("armor_detect_node"), "Potential armor switch detected! Guiding the state.");

//     // 1. 获取当前状态
//     State current_state = this->getTargetState();

//     // 2. 保留并约束历史速度
//     Eigen::Vector3d old_velocity(current_state(1), current_state(3), current_state(5));
//     double old_speed = old_velocity.norm();
    
//     constexpr double MAX_REASONABLE_SPEED = 1000.0; // 3 m/s，可调
//     if (old_speed > MAX_REASONABLE_SPEED) {
//         old_velocity = old_velocity.normalized() * MAX_REASONABLE_SPEED;
//     }

//     // 3. 构建新的“引导”状态
//     State guided_state;
//     guided_state(0) = z(0);
//     guided_state(1) = old_velocity.x();
//     guided_state(2) = z(1);
//     guided_state(3) = old_velocity.y();
//     guided_state(4) = z(2);
//     guided_state(5) = old_velocity.z();

//     // 4. 手动设置EKF内部状态
//     ekf_->setState(guided_state);
    
//     // 5. 将追踪器状态拉回到TRACKING
//     lost_count_ = 0;
//     detect_count_ = 0; // 可以顺便重置，以防万一
//     state = TRACKING; // 恢复状态
//     RCLCPP_DEBUG(rclcpp::get_logger("armor_detect_node"), "Tracker state guided back to TRACKING.");
// }



// #include "EKF/Tracker.h"
// #include "rclcpp/rclcpp.hpp"

// Tracker::Tracker(double dt, const EKFParams& params) : dt_(dt), state(LOST) {
//     RobotEKF::UpdateQFunc update_Q = [this, params]() {
//         Eigen::Matrix<double, N_x, N_x> Q = Eigen::Matrix<double, N_x, N_x>::Zero();
//         double t = this->dt_;
//         // 从传入的参数结构体中获取过程噪声
//         double s2qx = params.s2qx, s2qy = params.s2qy, s2qz = params.s2qz, 
//                s2qyaw = params.s2qyaw, s2qr = params.s2qr;

//         // X方向 (索引 0, 1)
//         double q_x_x=pow(t,4)/4*s2qx, q_x_vx=pow(t,3)/2*s2qx, q_vx_vx=pow(t,2)*s2qx;
//         Q(0,0)=q_x_x; Q(0,1)=q_x_vx; Q(1,0)=q_x_vx; Q(1,1)=q_vx_vx;

//         // Y方向 (索引 2, 3)
//         double q_y_y=pow(t,4)/4*s2qy, q_y_vy=pow(t,3)/2*s2qy, q_vy_vy=pow(t,2)*s2qy;
//         Q(2,2)=q_y_y; Q(2,3)=q_y_vy; Q(3,2)=q_y_vy; Q(3,3)=q_vy_vy;

//         // Z方向 (索引 4, 5)
//         double q_z_z=pow(t,4)/4*s2qz, q_z_vz=pow(t,3)/2*s2qz, q_vz_vz=pow(t,2)*s2qz;
//         Q(4,4)=q_z_z; Q(4,5)=q_z_vz; Q(5,4)=q_z_vz; Q(5,5)=q_vz_vz;

//         // Yaw (索引 6, 7)
//         double q_yaw_yaw=pow(t,4)/4*s2qyaw, q_yaw_vyaw=pow(t,3)/2*s2qyaw, q_vyaw_vyaw=pow(t,2)*s2qyaw;
//         Q(6,6)=q_yaw_yaw; Q(6,7)=q_yaw_vyaw; Q(7,6)=q_yaw_vyaw; Q(7,7)=q_vyaw_vyaw;

//         // r (索引 8)
//         Q(8,8) = pow(t,2)*s2qr;

//         return Q;
//     };

//     RobotEKF::UpdateRFunc update_R = [params](const Measurement& z) {
//         RobotEKF::MatrixZZ R;
//         // 从传入的参数结构体中获取测量噪声
//         double r_x = params.r_x;
//         double r_y = params.r_y;
//         double r_z = params.r_z;
//         double r_yaw = params.r_yaw;

//         // 动态噪声模型：x和z的测量不确定性随距离增大
//         double dist = sqrt(z[0]*z[0] + z[2]*z[2]);
//         R << r_x * dist, 0, 0, 0,
//              0, r_y * dist, 0, 0,
//              0, 0, r_z * dist, 0,
//              0, 0, 0, r_yaw;
//         return R;
//     };

//     // 从传入的参数结构体中获取初始协方差
//     RobotEKF::MatrixXX P0 = RobotEKF::MatrixXX::Identity() * params.p0;

//     ekf_ = std::make_unique<RobotEKF>(Predict(dt_), Measure(), update_Q, update_R, P0);
// }

// void Tracker::reset(const Measurement& z) {
//     state = DETECTING;
//     detect_count_ = 0; lost_count_ = 0;
    
//     State x0 = State::Zero();
    
//     double xa = z(0), ya = z(1), za = z(2), yaw = z(3);
//     // 使用一个合理的初始半径来估计中心位置，例如200mm
//     double r_init = 200.0; 
    
//     // 根据Measure函数的反函数来计算中心初始位置
//     double xc = xa + r_init * sin(yaw);
//     double yc = ya - r_init * cos(yaw); // 简化模型，高度相同
//     double zc = za ;
    
//     x0(0) = xc;
//     x0(2) = yc;
//     x0(4) = zc;
//     x0(6) = yaw;
//     x0(8) = r_init; // 设置初始半径

//     ekf_->setState(x0);
//     RCLCPP_DEBUG(rclcpp::get_logger("armor_detect_node"), "Tracker RESET with 9D model!");
// }

// Tracker::State Tracker::predict() {
//     auto state_vec = ekf_->predict();
//     if (state == TRACKING) {
//         lost_count_++;
//         if (lost_count_ > lost_thres_) state = LOST;
//         else state = TEMP_LOST;
//     } else if (state == DETECTING) {
//         lost_count_++;
//         if (lost_count_ > lost_thres_ / 2) state = LOST;
//     }
//     return state_vec;
// }

// Tracker::State Tracker::update(const Measurement& z) {
//     auto state_vec = ekf_->update(z);
    
//     // 对半径增加约束，防止发散
//     if (state_vec(8) < 120.0) state_vec(8) = 120.0; // 最小半径
//     else if (state_vec(8) > 400.0) state_vec(8) = 400.0; // 最大半径
//     ekf_->setState(state_vec);

//     lost_count_ = 0;
//     if (state == DETECTING) {
//         detect_count_++;
//         if (detect_count_ > tracking_thres_) {
//             state = TRACKING;
//             RCLCPP_DEBUG(rclcpp::get_logger("armor_detect_node"), "Tracker stable: TRACKING");
//         }
//     } else if (state == TEMP_LOST) state = TRACKING;
//     return state_vec;
// }

// Tracker::State Tracker::getTargetState() const { return ekf_->getState(); }

// Eigen::Vector3d Tracker::getArmorPosition() const {
//     State x = getTargetState();
//     double xc=x(0), yc=x(2), zc=x(4), yaw=x(6), r=x(8);
//     // 根据Measure函数计算装甲板位置
//     return {xc - r * sin(yaw), yc + r * cos(yaw), zc};
// }

// Tracker::State Tracker::predictAhead(double t_ahead) const {
//     if (t_ahead < 1e-3) return getTargetState();
//     State x_k = getTargetState();
//     Predict pred(t_ahead);
//     State x_final;
//     pred(x_k.data(), x_final.data());
//     return x_final;
// }