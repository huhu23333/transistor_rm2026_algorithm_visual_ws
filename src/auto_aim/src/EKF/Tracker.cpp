#include "EKF/Tracker.h"
#include "rclcpp/rclcpp.hpp"

Tracker::Tracker(double dt, const EKFParams& params) : dt_(dt), state(LOST)
{
    RobotEKF::UpdateQFunc update_Q = [this, params]()
    {
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

    RobotEKF::UpdateRFunc update_R = [params](const Measurement& z)
    {
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
    r_init_ = 200.0; 
    
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