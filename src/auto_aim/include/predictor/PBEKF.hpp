#include <vector>
#include <cmath>
#include <array>
#include <iostream>

// Eigen库用于矩阵运算，需要安装Eigen3
#include <Eigen/Dense>
#include <Eigen/LU>

struct PBEKF_ObservedData {
    double dt;   // 时间间隔
    double x;   // x坐标
    double y;   // y坐标  
    double z;   // z坐标
    double yaw; // 偏航角
};

class PBEKF_EKFTracker {
private:
    // 状态维度
    static constexpr int STATE_DIM = 9;
    static constexpr int OBS_DIM = 4;
    
    // 状态向量: [xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r]
    Eigen::VectorXd state_;
    
    // 协方差矩阵
    Eigen::MatrixXd P_;
    
    // 过程噪声协方差矩阵
    Eigen::MatrixXd Q_;
    
    // 观测噪声协方差矩阵  
    Eigen::MatrixXd R_;
    
    // 时间相关参数
    double dt_ = 0.033; // 默认时间间隔
    
    // OUTPOST_3模式参数
    int tracked_armors_num_ = 3; // 前哨站有3个装甲板
    std::string tracked_id_ = "outpost";
    double another_r_ = 250.0; // 另一个半径
    double dz_ = 0.0; // 高度差
    
    // 存储所有装甲板的角度
    std::vector<double> all_armors_yaw_;
    
    // 对象属性
    double center_x_, center_vx_, center_y_, center_vy_, center_z_, center_vz_;
    double yaw_, vyaw_, r_;

    double last_yaw = 0.0;

public:
    PBEKF_EKFTracker(const PBEKF_ObservedData& init_observed_data) 
        : state_(STATE_DIM), P_(STATE_DIM, STATE_DIM), Q_(STATE_DIM, STATE_DIM), 
          R_(OBS_DIM, OBS_DIM), all_armors_yaw_(tracked_armors_num_) {
        
        // 初始化状态
        center_x_ = init_observed_data.x - 250.0 * std::sin(init_observed_data.yaw);
        center_y_ = init_observed_data.y + 250.0 * std::cos(init_observed_data.yaw);
        center_z_ = init_observed_data.z;
        center_vx_ = 0.0;
        center_vy_ = 0.0;
        center_vz_ = 0.0;
        yaw_ = init_observed_data.yaw;
        vyaw_ = 0.0;
        r_ = 250.0;
        
        state_ << center_x_, center_vx_, center_y_, center_vy_, 
                  center_z_, center_vz_, yaw_, vyaw_, r_;
        
        // 误差协方差矩阵
        P_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 10.0;
        
        // 过程噪声协方差矩阵
        Q_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.1;
        
        // 观测噪声协方差矩阵
        R_ = Eigen::MatrixXd::Zero(OBS_DIM, OBS_DIM);
        R_.diagonal() << 100.0, 100.0, 100.0, 0.1;
        
        // 初始化所有装甲板的角度
        for (int i = 0; i < tracked_armors_num_; ++i) {
            all_armors_yaw_[i] = init_observed_data.yaw;
        }
        
        std::cout << "EKF Tracker initialized with OUTPOST_3 mode, " 
                  << tracked_armors_num_ << " armors" << std::endl;
    }

private:
    /**
     * 状态转移函数（过程模型）
     */
    Eigen::VectorXd processModel(const Eigen::VectorXd& state, double dt) {
        Eigen::VectorXd x_new = state;
        // 匀速模型
        x_new(0) += state(1) * dt; // xc += v_xc * dt
        x_new(2) += state(3) * dt; // yc += v_yc * dt
        x_new(4) += state(5) * dt; // za += v_za * dt
        x_new(6) += state(7) * dt; // yaw += v_yaw * dt
        return x_new;
    }

    /**
     * 观测模型
     */
    Eigen::VectorXd observationModel(const Eigen::VectorXd& state) {
        double xc = state(0);
        double yc = state(2);
        double za = state(4);
        double yaw = state(6);
        double r = state(8);
        
        Eigen::VectorXd z_obs(OBS_DIM);
        z_obs << xc + r * std::sin(yaw),  // x = xc + r*sin(yaw)
                 yc - r * std::cos(yaw),  // y = yc - r*cos(yaw)
                 za,                      // z = za
                 yaw;                     // yaw = yaw
        return z_obs;
    }

    /**
     * 状态转移函数的雅可比矩阵
     */
    Eigen::MatrixXd jacobianF(const Eigen::VectorXd& state, double dt) {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
        F(0, 1) = dt; // ∂f0/∂v_xc = dt
        F(2, 3) = dt; // ∂f2/∂v_yc = dt
        F(4, 5) = dt; // ∂f4/∂v_za = dt
        F(6, 7) = dt; // ∂f6/∂v_yaw = dt
        return F;
    }

    /**
     * 观测函数的雅可比矩阵
     */
    Eigen::MatrixXd jacobianH(const Eigen::VectorXd& state) {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(OBS_DIM, STATE_DIM);
        
        double xc = state(0);
        double yc = state(2);
        double za = state(4);
        double yaw = state(6);
        double r = state(8);
        
        // 根据修改后的观测模型计算雅可比矩阵
        // z0 = xc + r * sin(yaw)
        H(0, 0) = 1.0;                    // ∂z0/∂xc = 1
        H(0, 6) = r * std::cos(yaw);      // ∂z0/∂yaw = r*cos(yaw)
        H(0, 8) = std::sin(yaw);          // ∂z0/∂r = sin(yaw)
        
        // z1 = yc - r * cos(yaw)
        H(1, 2) = 1.0;                    // ∂z1/∂yc = 1
        H(1, 6) = r * std::sin(yaw);      // ∂z1/∂yaw = r*sin(yaw)
        H(1, 8) = -std::cos(yaw);         // ∂z1/∂r = -cos(yaw)
        
        // z2 = za
        H(2, 4) = 1.0;                    // ∂z2/∂za = 1
        
        // z3 = yaw
        H(3, 6) = 1.0;                    // ∂z3/∂yaw = 1
        
        return H;
    }

    /**
     * 更新过程噪声协方差矩阵
     */
    void updateQ(double dt) {
        Q_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.1;
        
        // 给位置和速度添加更多噪声
        double s2qxyz = 20.0;  // 位置噪声
        double s2qyaw = 1.0; // 偏航角噪声
        double s2qr = 10.0;   // 半径噪声
        
        // 位置和速度的噪声（基于时间）
        double q_x_x = std::pow(dt, 4) / 4.0 * s2qxyz;
        double q_x_vx = std::pow(dt, 3) / 2.0 * s2qxyz;
        double q_vx_vx = std::pow(dt, 2) * s2qxyz;
        
        double q_y_y = std::pow(dt, 4) / 4.0 * s2qyaw;
        double q_y_vy = std::pow(dt, 3) / 2.0 * s2qyaw;
        double q_vy_vy = std::pow(dt, 2) * s2qyaw;
        
        double q_r = std::pow(dt, 4) / 4.0 * s2qr;
        
        // 设置噪声矩阵
        Q_(0, 0) = q_x_x;      // xc
        Q_(0, 1) = q_x_vx;     // xc-v_xc
        Q_(1, 0) = q_x_vx;     // v_xc-xc
        Q_(1, 1) = q_vx_vx;    // v_xc
        
        Q_(2, 2) = q_x_x;      // yc
        Q_(2, 3) = q_x_vx;     // yc-v_yc
        Q_(3, 2) = q_x_vx;     // v_yc-yc
        Q_(3, 3) = q_vx_vx;    // v_yc
        
        Q_(4, 4) = q_x_x;      // za
        Q_(4, 5) = q_x_vx;     // za-v_za
        Q_(5, 4) = q_x_vx;     // v_za-za
        Q_(5, 5) = q_vx_vx;    // v_za
        
        Q_(6, 6) = q_y_y;      // yaw
        Q_(6, 7) = q_y_vy;     // yaw-v_yaw
        Q_(7, 6) = q_y_vy;     // v_yaw-yaw
        Q_(7, 7) = q_vy_vy;    // v_yaw
        
        Q_(8, 8) = q_r;        // r
    }

    /**
     * 更新观测噪声协方差矩阵
     */
    void updateR(const Eigen::VectorXd& z) {
        R_ = Eigen::MatrixXd::Zero(OBS_DIM, OBS_DIM);
        R_.diagonal() << 10.0, 10.0, 10.0, 0.1;
    }

    /**
     * 更新对象属性
     */
    void updateAttributes() {
        center_x_ = state_(0);
        center_vx_ = state_(1);
        center_y_ = state_(2);
        center_vy_ = state_(3);
        center_z_ = state_(4);
        center_vz_ = state_(5);
        yaw_ = state_(6);
        vyaw_ = state_(7);
        r_ = state_(8);
    }

    /**
     * 更新所有装甲板的角度（OUTPOST_3模式）
     */
    void updateArmorsYaw() {
        double base_yaw = yaw_;
        double armor_interval = 2.0 * M_PI / tracked_armors_num_;
        for (int i = 0; i < tracked_armors_num_; ++i) {
            all_armors_yaw_[i] = base_yaw + i * armor_interval;
        }
    }

    /**
     * 处理装甲板跳变（OUTPOST_3模式）
     */
    bool handleArmorJump(const PBEKF_ObservedData& observed_data) {
        double dt = observed_data.dt;
        
        double measured_yaw = observed_data.yaw;
        
        // 计算角度差异
        double yaw_diff = std::abs(measured_yaw - last_yaw);
        
        // 如果角度差异超过阈值，可能是装甲板跳变
        double jump_threshold = M_PI / 3.0; // 60度阈值
        
        if (yaw_diff > jump_threshold) {
            // 在OUTPOST_3模式下，直接更新偏航角状态
            state_(6) = measured_yaw - vyaw_ * dt;
            std::cout << "Armor jump detected! Updating yaw from " 
                      << last_yaw << " to " << measured_yaw << std::endl;
            
            // 更新属性
            updateAttributes();
            return true;
        }
        
        return false;
    }

public:
    /**
     * 预测步骤
     */
    std::array<double, 4> predict(double dt = -1.0) {
        if (dt < 0) {
            dt = dt_;
        }
        
        // 更新过程噪声
        updateQ(dt);
        
        // 计算雅可比矩阵
        Eigen::MatrixXd F = jacobianF(state_, dt);
        
        // 状态预测
        state_ = processModel(state_, dt);
        
        // 协方差预测
        P_ = F * P_ * F.transpose() + Q_;
        
        // 更新对象属性
        updateAttributes();
        
        // 计算所有装甲板的角度（OUTPOST_3模式）
        updateArmorsYaw();
        
        // 返回预测的观测值
        Eigen::VectorXd pred_obs = observationModel(state_);
        return {pred_obs(0), pred_obs(1), pred_obs(2), pred_obs(3)};
    }

    /**
     * 更新步骤
     */
    void update(const PBEKF_ObservedData& observed_data) {
        // 处理装甲板跳变（OUTPOST_3模式）
        handleArmorJump(observed_data);
        last_yaw = observed_data.yaw;
        
        // 计算时间间隔
        double dt = observed_data.dt;
        
        // 预测步骤
        predict(dt);
        
        // 观测值
        Eigen::VectorXd z(OBS_DIM);
        z << observed_data.x, observed_data.y, observed_data.z, observed_data.yaw;
        
        // 更新观测噪声
        updateR(z);
        
        // 计算雅可比矩阵
        Eigen::MatrixXd H = jacobianH(state_);
        
        // 计算卡尔曼增益
        Eigen::MatrixXd S = H * P_ * H.transpose() + R_;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
        
        // 观测预测
        Eigen::VectorXd z_pred = observationModel(state_);
        
        // 状态更新
        Eigen::VectorXd innovation = z - z_pred;
        state_ = state_ + K * innovation;
        
        // 协方差更新
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
        P_ = (I - K * H) * P_;

        // 处理异常属性
        if (std::abs(state_(7)) > 5.0) {
            state_(7) = 0.0;
        }
        
        // 更新对象属性
        updateAttributes();
        
        // 更新所有装甲板的角度
        updateArmorsYaw();
    }

    /**
     * 获取用于可视化的状态信息
     */
    std::vector<double> getStateForVisualization() const {
        std::vector<double> result;
        result.push_back(center_x_);
        result.push_back(center_vx_);
        result.push_back(center_y_);
        result.push_back(center_vy_);
        result.push_back(center_z_);
        result.push_back(center_vz_);
        result.push_back(vyaw_);
        result.push_back(r_);
        result.insert(result.end(), all_armors_yaw_.begin(), all_armors_yaw_.end());
        return result;
    }

    // Getter方法
    double getCenterX() const { return center_x_; }
    double getCenterY() const { return center_y_; }
    double getCenterZ() const { return center_z_; }
    double getYaw() const { return yaw_; }
    double getRadius() const { return r_; }
    const std::vector<double>& getAllArmorsYaw() const { return all_armors_yaw_; }
};