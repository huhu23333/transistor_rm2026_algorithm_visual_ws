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




// #include "EKF/Tracker.h"
// #include "rclcpp/rclcpp.hpp"

// Tracker::Tracker(double dt, const EKFParams& params) : dt_(dt), state(LOST) {
//     RobotEKF::UpdateQFunc update_Q = [this, params]() {
//         Eigen::Matrix<double, N_x, N_x> Q = Eigen::Matrix<double, N_x, N_x>::Zero();
//         double t = this->dt_;
//         double t2 = t * t;
//         double t3_2 = pow(t, 3) / 2.0;
//         double t4_4 = pow(t, 4) / 4.0;
        
//         // 从传入的参数结构体中获取过程噪声标准差
//         double s2qx = params.s2qx, s2qy = params.s2qy, s2qz = params.s2qz;

//         // 匀速模型的过程噪声矩阵
//         // Q = G * diag([s2qx, s2qy, s2qz]) * G^T, where G = [t^2/2, t, 0, 0, ...]^T
//         Q(0,0)=t4_4*s2qx; Q(0,1)=t3_2*s2qx;
//         Q(1,0)=t3_2*s2qx; Q(1,1)=t2*s2qx;

//         Q(2,2)=t4_4*s2qy; Q(2,3)=t3_2*s2qy;
//         Q(3,2)=t3_2*s2qy; Q(3,3)=t2*s2qy;

//         Q(4,4)=t4_4*s2qz; Q(4,5)=t3_2*s2qz;
//         Q(5,4)=t3_2*s2qz; Q(5,5)=t2*s2qz;

//         return Q;
//     };

//     RobotEKF::UpdateRFunc update_R = [params](const Measurement& z) {
//         RobotEKF::MatrixZZ R;
//         // 测量噪声仍然可以和距离相关
//         double r_x = params.r_x;
//         double r_y = params.r_y;
//         double r_z = params.r_z;
        
//         // 示例：让x,y的噪声也和z相关，可以根据实际情况调整
//         double dist_z = std::max(1.0, std::abs(z[2]));
//         R << r_x * dist_z, 0, 0,
//              0, r_y * dist_z, 0,
//              0, 0, r_z * dist_z;
//         return R;
//     };

//     RobotEKF::MatrixXX P0 = RobotEKF::MatrixXX::Identity() * params.p0;

//     ekf_ = std::make_unique<RobotEKF>(Predict(dt_), Measure(), update_Q, update_R, P0);
// }

// void Tracker::reset(const Measurement& z) {
//     state = DETECTING;
//     detect_count_ = 0; lost_count_ = 0;
    
//     State x0 = State::Zero();
//     // 测量值z现在是 [xa, ya, za]
//     x0(0) = z(0);
//     x0(2) = z(1);
//     x0(4) = z(2);
//     // 速度初始化为0
    
//     ekf_->setState(x0);
//     RCLCPP_DEBUG(rclcpp::get_logger("armor_detect_node"), "Tracker RESET with 6D model!");
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
//     return {x(0), x(2), x(4)};
// }

// Tracker::State Tracker::predictAhead(double t_ahead) const {
//     if (t_ahead < 1e-3) return getTargetState();
//     State x_k = getTargetState();
//     Predict pred(t_ahead);
//     State x_final;
//     pred(x_k.data(), x_final.data());
//     return x_final;
// }

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



#include "EKF/Tracker.h"
#include "rclcpp/rclcpp.hpp"

Tracker::Tracker(double dt, const EKFParams& params) : dt_(dt), state(LOST) {
    RobotEKF::UpdateQFunc update_Q = [this, params]() {
        Eigen::Matrix<double, N_x, N_x> Q = Eigen::Matrix<double, N_x, N_x>::Zero();
        double t = this->dt_;
        // 从传入的参数结构体中获取过程噪声
        double s2qx = params.s2qx, s2qy = params.s2qy, s2qz = params.s2qz, 
               s2qyaw = params.s2qyaw, s2qr = params.s2qr;

        // X方向 (索引 0, 1)
        double q_x_x=pow(t,4)/4*s2qx, q_x_vx=pow(t,3)/2*s2qx, q_vx_vx=pow(t,2)*s2qx;
        Q(0,0)=q_x_x; Q(0,1)=q_x_vx; Q(1,0)=q_x_vx; Q(1,1)=q_vx_vx;

        // Y方向 (索引 2, 3)
        double q_y_y=pow(t,4)/4*s2qy, q_y_vy=pow(t,3)/2*s2qy, q_vy_vy=pow(t,2)*s2qy;
        Q(2,2)=q_y_y; Q(2,3)=q_y_vy; Q(3,2)=q_y_vy; Q(3,3)=q_vy_vy;

        // Z方向 (索引 4, 5)
        double q_z_z=pow(t,4)/4*s2qz, q_z_vz=pow(t,3)/2*s2qz, q_vz_vz=pow(t,2)*s2qz;
        Q(4,4)=q_z_z; Q(4,5)=q_z_vz; Q(5,4)=q_z_vz; Q(5,5)=q_vz_vz;

        // Yaw (索引 6, 7)
        double q_yaw_yaw=pow(t,4)/4*s2qyaw, q_yaw_vyaw=pow(t,3)/2*s2qyaw, q_vyaw_vyaw=pow(t,2)*s2qyaw;
        Q(6,6)=q_yaw_yaw; Q(6,7)=q_yaw_vyaw; Q(7,6)=q_yaw_vyaw; Q(7,7)=q_vyaw_vyaw;

        // r (索引 8)
        Q(8,8) = pow(t,2)*s2qr;

        return Q;
    };

    RobotEKF::UpdateRFunc update_R = [params](const Measurement& z) {
        RobotEKF::MatrixZZ R;
        // 从传入的参数结构体中获取测量噪声
        double r_x = params.r_x;
        double r_y = params.r_y;
        double r_z = params.r_z;
        double r_yaw = params.r_yaw;

        // 动态噪声模型：x和z的测量不确定性随距离增大
        double dist = sqrt(z[0]*z[0] + z[2]*z[2]);
        R << r_x * dist, 0, 0, 0,
             0, r_y * dist, 0, 0,
             0, 0, r_z * dist, 0,
             0, 0, 0, r_yaw;
        return R;
    };

    // 从传入的参数结构体中获取初始协方差
    RobotEKF::MatrixXX P0 = RobotEKF::MatrixXX::Identity() * params.p0;

    ekf_ = std::make_unique<RobotEKF>(Predict(dt_), Measure(), update_Q, update_R, P0);
}

void Tracker::reset(const Measurement& z) {
    state = DETECTING;
    detect_count_ = 0; lost_count_ = 0;
    
    State x0 = State::Zero();
    
    double xa = z(0), ya = z(1), za = z(2), yaw = z(3);
    // 使用一个合理的初始半径来估计中心位置，例如200mm
    double r_init = 200.0; 
    
    // 根据Measure函数的反函数来计算中心初始位置
    double xc = xa + r_init * sin(yaw);
    double yc = ya - r_init * cos(yaw); // 简化模型，高度相同
    double zc = za ;
    
    x0(0) = xc;
    x0(2) = yc;
    x0(4) = zc;
    x0(6) = yaw;
    x0(8) = r_init; // 设置初始半径

    ekf_->setState(x0);
    RCLCPP_DEBUG(rclcpp::get_logger("armor_detect_node"), "Tracker RESET with 9D model!");
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
    
    // 对半径增加约束，防止发散
    if (state_vec(8) < 120.0) state_vec(8) = 120.0; // 最小半径
    else if (state_vec(8) > 400.0) state_vec(8) = 400.0; // 最大半径
    ekf_->setState(state_vec);

    lost_count_ = 0;
    if (state == DETECTING) {
        detect_count_++;
        if (detect_count_ > tracking_thres_) {
            state = TRACKING;
            RCLCPP_DEBUG(rclcpp::get_logger("armor_detect_node"), "Tracker stable: TRACKING");
        }
    } else if (state == TEMP_LOST) state = TRACKING;
    return state_vec;
}

Tracker::State Tracker::getTargetState() const { return ekf_->getState(); }

Eigen::Vector3d Tracker::getArmorPosition() const {
    State x = getTargetState();
    double xc=x(0), yc=x(2), zc=x(4), yaw=x(6), r=x(8);
    // 根据Measure函数计算装甲板位置
    return {xc - r * sin(yaw), yc + r * cos(yaw), zc};
}

Tracker::State Tracker::predictAhead(double t_ahead) const {
    if (t_ahead < 1e-3) return getTargetState();
    State x_k = getTargetState();
    Predict pred(t_ahead);
    State x_final;
    pred(x_k.data(), x_final.data());
    return x_final;
}