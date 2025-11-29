// RotationMotionModel.h
#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include "utils/DataProcessFuncs.h"
#include "3d_processing/RestFrame.h"
#include "2d_armor_detector/Armor.h"

struct ObservedData {
    double x;
    double y;
    double z;
    double yaw;
    double t;
    
    ObservedData(double x_val, double y_val, double z_val, double yaw_val, double t_val)
        : x(x_val), y(y_val), z(z_val), yaw(yaw_val), t(t_val) {}
};

struct SimpleArmor {
    double x;
    double y;
    double z;
    double yaw;
};

struct PredictResult {
    double center_x;
    double center_y;
    double center_z;
    double r;
    double yaw;
    int rotation_direction;
    std::vector<SimpleArmor> armors;
};

struct RotationMotionState {
    double center_vx;
    double center_vy;
    double center_vz;
    double vyaw;
    double r;
    double center_x;
    double center_y;
    double center_z;
};

// 平移和半径跟踪的EKF
class TranslationRadiusEKF {
private:
    static constexpr int STATE_DIM = 7;  // [xc, yc, zc, vx, vy, vz, r]
    static constexpr int OBS_DIM = 3;    // [xa, ya, za]
    
    Eigen::VectorXd state_;              // [xc, yc, zc, vx, vy, vz, r]
    Eigen::MatrixXd P_;                  // 协方差矩阵
    Eigen::MatrixXd Q_;                  // 过程噪声
    Eigen::MatrixXd R_;                  // 观测噪声
    
    bool initialized_ = false;
    double last_time_ = 0.0;
    
public:
    TranslationRadiusEKF() 
        : state_(STATE_DIM), P_(STATE_DIM, STATE_DIM), 
          Q_(STATE_DIM, STATE_DIM), R_(OBS_DIM, OBS_DIM) {
        
        // 初始化状态
        state_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 250.0;
        
        // 初始化协方差
        P_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 10.0;
        
        // 初始化过程噪声
        Q_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.1;
        
        // 初始化观测噪声
        R_ = Eigen::MatrixXd::Identity(OBS_DIM, OBS_DIM) * 1.0;
    }
    
    void initialize(double xc, double yc, double zc, double r, double current_time) {
        state_(0) = xc;
        state_(1) = yc;
        state_(2) = zc;
        state_(6) = r;
        last_time_ = current_time;
        initialized_ = true;
    }
    
    bool isInitialized() const { return initialized_; }
    
    /**
     * 状态转移函数 - 匀速直线运动 + 恒定半径
     */
    Eigen::VectorXd processModel(const Eigen::VectorXd& state, double dt) {
        Eigen::VectorXd new_state = state;
        new_state(0) += state(3) * dt;  // xc += vx * dt
        new_state(1) += state(4) * dt;  // yc += vy * dt  
        new_state(2) += state(5) * dt;  // zc += vz * dt
        // 速度保持不变，半径保持不变
        return new_state;
    }
    
    /**
     * 观测模型 - 根据旋转中心、半径和装甲板yaw计算装甲板位置
     */
    Eigen::Vector3d observationModel(const Eigen::VectorXd& state, double armor_yaw) {
        Eigen::Vector3d z_obs;
        double xc = state(0), yc = state(1), zc = state(2), r = state(6);
        
        z_obs(0) = xc + r * std::sin(armor_yaw);  // xa = xc + r * sin(yaw)
        z_obs(1) = yc - r * std::cos(armor_yaw);  // ya = yc - r * cos(yaw)
        z_obs(2) = zc;                            // za = zc
        
        return z_obs;
    }
    
    /**
     * 状态转移雅可比矩阵
     */
    Eigen::MatrixXd jacobianF(double dt) {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
        F(0, 3) = dt;  // ∂xc/∂vx
        F(1, 4) = dt;  // ∂yc/∂vy
        F(2, 5) = dt;  // ∂zc/∂vz
        return F;
    }
    
    /**
     * 观测雅可比矩阵
     */
    Eigen::Matrix<double, 3, 7> jacobianH(const Eigen::VectorXd& state, double armor_yaw) {
        Eigen::Matrix<double, 3, 7> H = Eigen::Matrix<double, 3, 7>::Zero();
        double r = state(6);
        
        // ∂xa/∂xc = 1, ∂xa/∂r = sin(yaw)
        H(0, 0) = 1.0;
        H(0, 6) = std::sin(armor_yaw);
        
        // ∂ya/∂yc = 1, ∂ya/∂r = -cos(yaw)
        H(1, 1) = 1.0;
        H(1, 6) = -std::cos(armor_yaw);
        
        // ∂za/∂zc = 1
        H(2, 2) = 1.0;
        
        return H;
    }
    
    /**
     * 更新过程噪声
     */
    void updateQ(double dt) {
        // 过程噪声参数
        double s2q_pos = 100.0;    // 位置过程噪声
        double s2q_vel = 1.0;   // 速度过程噪声
        double s2q_r = 0.1;    // 半径过程噪声
        
        Q_.setZero();
        Q_(0, 0) = std::pow(dt, 4)/4 * s2q_pos;  // xc
        Q_(1, 1) = std::pow(dt, 4)/4 * s2q_pos;  // yc
        Q_(2, 2) = std::pow(dt, 4)/4 * s2q_pos;  // zc
        Q_(3, 3) = std::pow(dt, 2) * s2q_vel;    // vx
        Q_(4, 4) = std::pow(dt, 2) * s2q_vel;    // vy
        Q_(5, 5) = std::pow(dt, 2) * s2q_vel;    // vz
        Q_(6, 6) = dt * s2q_r;                   // r
    }
    
    /**
     * 预测步骤
     */
    void predict(double current_time) {
        if (!initialized_) return;
        
        double dt = current_time - last_time_;
        if (dt <= 0) return;
        
        // 更新过程噪声
        updateQ(dt);
        
        // 计算雅可比矩阵
        Eigen::MatrixXd F = jacobianF(dt);
        
        // 状态预测
        state_ = processModel(state_, dt);
        
        // 协方差预测
        P_ = F * P_ * F.transpose() + Q_;
        
        last_time_ = current_time;
    }
    
    /**
     * 更新步骤 - 使用装甲板观测数据
     */
    void update(double armor_x, double armor_y, double armor_z, double armor_yaw, double current_time) {
        if (!initialized_) {
            // 使用第一个观测初始化状态
            double init_r = 250.0;  // 默认半径
            double init_xc = armor_x - init_r * std::sin(armor_yaw);
            double init_yc = armor_y + init_r * std::cos(armor_yaw);
            initialize(init_xc, init_yc, armor_z, init_r, current_time);
            return;
        }
        
        // 预测步骤
        predict(current_time);
        
        // 观测值
        Eigen::Vector3d z;
        z << armor_x, armor_y, armor_z;
        
        // 计算雅可比矩阵
        Eigen::Matrix<double, 3, 7> H = jacobianH(state_, armor_yaw);
        
        // 计算卡尔曼增益
        Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
        Eigen::Matrix<double, 7, 3> K = P_ * H.transpose() * S.inverse();
        
        // 观测预测
        Eigen::Vector3d z_pred = observationModel(state_, armor_yaw);
        
        // 状态更新
        Eigen::Vector3d innovation = z - z_pred;
        state_ = state_ + K * innovation;
        
        // 协方差更新
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
        P_ = (I - K * H) * P_;
        
        // 限制半径范围
        state_(6) = std::max(100.0, std::min(600.0, state_(6)));
    }
    
    // Getter方法
    double getCenterX() const { return state_(0); }
    double getCenterY() const { return state_(1); }
    double getCenterZ() const { return state_(2); }
    double getVelocityX() const { return state_(3); }
    double getVelocityY() const { return state_(4); }
    double getVelocityZ() const { return state_(5); }
    double getRadius() const { return state_(6); }
    const Eigen::VectorXd& getState() const { return state_; }
};

// 用于角度和角速度跟踪的EKF
class AngleEKF {
private:
    static constexpr int STATE_DIM = 2;  // [yaw, vyaw]
    static constexpr int OBS_DIM = 3;    // [yaw, xa, ya]
    
    Eigen::Vector2d state_;              // [yaw, vyaw]
    Eigen::Matrix2d P_;                  // 协方差矩阵
    Eigen::Matrix2d Q_;                  // 过程噪声
    Eigen::Matrix3d R_;                  // 观测噪声
    
    double last_yaw_ = 0.0;
    bool initialized_ = false;
    
public:
    AngleEKF() : state_(2), P_(2, 2), Q_(2, 2), R_(3, 3) {
        // 初始化状态
        state_ << 0.0, 0.0;
        
        // 初始化协方差
        P_ = Eigen::Matrix2d::Identity() * 10.0;
        
        // 初始化过程噪声
        Q_ = Eigen::Matrix2d::Identity() * 0.1;
        
        // 初始化观测噪声
        R_ = Eigen::Matrix3d::Identity() * 1.0;
    }
    
    void initialize(double init_yaw) {
        state_(0) = init_yaw;
        state_(1) = 0.0;
        last_yaw_ = init_yaw;
        initialized_ = true;
    }
    
    bool isInitialized() const { return initialized_; }
    
    /**
     * 角度跟踪的状态转移函数
     */
    Eigen::Vector2d processModel(const Eigen::Vector2d& state, double dt) {
        Eigen::Vector2d new_state;
        new_state(0) = state(0) + state(1) * dt;  // yaw += vyaw * dt
        new_state(1) = state(1);                  // vyaw 保持不变
        return new_state;
    }
    
    /**
     * 观测模型 - 使用xc, yc, r计算装甲板位置
     */
    Eigen::Vector3d observationModel(const Eigen::Vector2d& state, double xc, double yc, double r) {
        Eigen::Vector3d z_obs;
        z_obs(0) = state(0);                                  // 观测yaw
        z_obs(1) = xc + r * std::sin(state(0));               // 观测xa
        z_obs(2) = yc - r * std::cos(state(0));               // 观测ya
        return z_obs;
    }
    
    /**
     * 状态转移雅可比矩阵
     */
    Eigen::Matrix2d jacobianF(double dt) {
        Eigen::Matrix2d F;
        F << 1.0, dt,
             0.0, 1.0;
        return F;
    }
    
    /**
     * 观测雅可比矩阵
     */
    Eigen::Matrix<double, 3, 2> jacobianH(const Eigen::Vector2d& state, double xc, double yc, double r) {
        Eigen::Matrix<double, 3, 2> H;
        double yaw = state(0);
        
        // ∂z0/∂yaw = 1, ∂z0/∂vyaw = 0
        H(0, 0) = 1.0;
        H(0, 1) = 0.0;
        
        // ∂z1/∂yaw = r * cos(yaw), ∂z1/∂vyaw = 0
        H(1, 0) = r * std::cos(yaw);
        H(1, 1) = 0.0;
        
        // ∂z2/∂yaw = r * sin(yaw), ∂z2/∂vyaw = 0
        H(2, 0) = r * std::sin(yaw);
        H(2, 1) = 0.0;
        
        return H;
    }
    
    /**
     * 更新过程噪声
     */
    void updateQ(double dt) {
        double s2q_yaw = 0.1;    // yaw过程噪声
        double s2q_vyaw = 0.01;   // 角速度过程噪声
        
        Q_(0, 0) = std::pow(dt, 4) / 4.0 * s2q_yaw;
        Q_(0, 1) = std::pow(dt, 3) / 2.0 * s2q_yaw;
        Q_(1, 0) = std::pow(dt, 3) / 2.0 * s2q_yaw;
        Q_(1, 1) = std::pow(dt, 2) * s2q_vyaw;
    }
    
    /**
     * 处理角度跳变
     */
    bool handleYawJump(double measured_yaw, double dt) {
        if (!initialized_) return false;
        
        double yaw_diff = measured_yaw - last_yaw_;
        
        // 处理角度环绕
        if (yaw_diff > M_PI) yaw_diff -= 2.0 * M_PI;
        if (yaw_diff < -M_PI) yaw_diff += 2.0 * M_PI;
        
        // 如果角度差异超过阈值，可能是装甲板跳变
        double jump_threshold = M_PI / 3.0; // 60度阈值
        
        if (std::abs(yaw_diff) > jump_threshold) {
            // 直接更新偏航角状态，保持角速度不变
            state_(0) = measured_yaw - state_(1) * dt;
            std::cout << "Yaw jump detected! Updating yaw from " 
                      << last_yaw_ << " to " << measured_yaw << std::endl;
            return true;
        }
        
        return false;
    }
    
    /**
     * 预测步骤
     */
    void predict(double dt) {
        if (!initialized_) return;
        
        // 更新过程噪声
        updateQ(dt);
        
        // 计算雅可比矩阵
        Eigen::Matrix2d F = jacobianF(dt);
        
        // 状态预测
        state_ = processModel(state_, dt);
        
        // 协方差预测
        P_ = F * P_ * F.transpose() + Q_;
        
        // 处理角度环绕
        if (state_(0) > M_PI) state_(0) -= 2.0 * M_PI;
        if (state_(0) < -M_PI) state_(0) += 2.0 * M_PI;
    }
    
    /**
     * 更新步骤
     */
    void update(double measured_yaw, double measured_xa, double measured_ya, 
                double xc, double yc, double r, double dt) {
        if (!initialized_) {
            initialize(measured_yaw);
            return;
        }
        
        // 处理角度跳变
        handleYawJump(measured_yaw, dt);
        
        // 预测步骤
        predict(dt);
        
        // 观测值
        Eigen::Vector3d z;
        z << measured_yaw, measured_xa, measured_ya;
        
        // 计算雅可比矩阵
        Eigen::Matrix<double, 3, 2> H = jacobianH(state_, xc, yc, r);
        
        // 计算卡尔曼增益
        Eigen::Matrix<double, 3, 3> S = H * P_ * H.transpose() + R_;
        Eigen::Matrix<double, 2, 3> K = P_ * H.transpose() * S.inverse();
        
        // 观测预测
        Eigen::Vector3d z_pred = observationModel(state_, xc, yc, r);
        
        // 状态更新
        Eigen::Vector3d innovation = z - z_pred;
        state_ = state_ + K * innovation;
        
        // 协方差更新
        Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
        P_ = (I - K * H) * P_;
        
        // 处理角度环绕
        if (state_(0) > M_PI) state_(0) -= 2.0 * M_PI;
        if (state_(0) < -M_PI) state_(0) += 2.0 * M_PI;
        
        // 限制角速度
        if (std::abs(state_(1)) > 5.0) {
            state_(1) = 0.0;
        }
        
        last_yaw_ = state_(0);
    }
    
    // Getter方法
    double getYaw() const { return state_(0); }
    double getVyaw() const { return state_(1); }
    const Eigen::Vector2d& getState() const { return state_; }
};

class RotationMotionModel {
private:
    // 移除滑动窗口
    // std::vector<ObservedData> observedDataHistory;
    
    // 使用两个EKF分别处理平移半径和角度
    std::unique_ptr<TranslationRadiusEKF> trans_radius_ekf_;
    std::unique_ptr<AngleEKF> angle_ekf_;
    
    double last_update_time_;
    int max_history;
    double jump_period_frames = 1.0;
    double rotation_period;
    double current_phase;
    int n_armors;
    int rotation_direction;
    double jump_rad;
    double delta_phase;

    std::shared_ptr<RestFrame> rest_frame_;
    bool is_outpost;

public:
    RotationMotionModel(ObservedData& initObservedData, std::shared_ptr<RestFrame> rest_frame_, bool is_outpost);
    void update(ObservedData& observedData);
    PredictResult predict(double predictTime);
    double getJumpPeriod();
    void emptyUpdate(double update_time);
    RotationMotionState getState();
    double getTheoreticYaw(double armor_x, double armor_y);
    double getTheoreticYawFacingArmor(double armor_x, double armor_y);
    double getCamToCenterYaw();
};