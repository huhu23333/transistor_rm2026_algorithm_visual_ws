// RotationMotionModel.cpp
#include "predictor/RotationMotionModel.h"

using namespace Eigen;

RotationMotionModel::RotationMotionModel(ObservedData& initObservedData, std::shared_ptr<RestFrame> rest_frame_, bool is_outpost)
    : rest_frame_(rest_frame_), is_outpost(is_outpost), last_observed_data(initObservedData) {
    observedDataHistory.push_back(initObservedData);
    
    // 初始化EKF用于角度跟踪
    angle_ekf_ = std::make_unique<AngleEKF>();
    last_update_time_ = initObservedData.t;
    
    center_vx = 0.0;
    center_vy = 0.0;
    center_vz = 0.0;
    
    if (is_outpost) {
        n_armors = 3;
        r = 276.5;
    } else {
        n_armors = 4;  // 4装甲板模式
        r = 250.0;
    }
    
    r_prev_ = r;  // 初始化历史半径值
    regularization_weight_ = 10.0;  // 正则化权重，可调整
    
    // 初始化中心位置
    center_x = initObservedData.x - r * sin(initObservedData.yaw);
    center_y = initObservedData.y + r * cos(initObservedData.yaw);
    center_z = initObservedData.z;  // 使用观测数据的z值初始化
    
    max_history = 90;
    rotation_direction = 1;
    jump_rad = M_PI * 2.0 / n_armors;
    
    // 初始化指数衰减最小二乘
    resetExponentialLS();
}

void RotationMotionModel::resetExponentialLS() {
    // 初始化7x7协方差矩阵
    P_center_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 1000.0;
    
    // 初始状态估计 [center_x, center_y, center_z, center_vx, center_vy, center_vz, r]
    x_center_ = Eigen::VectorXd::Zero(STATE_DIM);
    x_center_(0) = center_x;  // center_x
    x_center_(1) = center_y;  // center_y
    x_center_(2) = center_z;  // center_z (新增)
    x_center_(3) = 0.0;       // center_vx
    x_center_(4) = 0.0;       // center_vy
    x_center_(5) = 0.0;       // center_vz (新增)
    x_center_(6) = r;         // r
    
    // 遗忘因子，值越小遗忘越快
    lambda_ = 0.95;
}

void RotationMotionModel::updateExponentialLS(double armor_x, double armor_y, double armor_z, double armor_yaw, double t, double weight) {
    // 构建三个测量方程
    
    // 测量1: 装甲板到中心的向量与装甲板朝向垂直 (xy平面)
    // offAxisVector · armorToCenterVector = 0
    double cosYaw = cos(armor_yaw);
    double sinYaw = sin(armor_yaw);
    double offAxisX = cosYaw;   // 垂直于装甲板朝向的单位向量x分量
    double offAxisY = sinYaw;   // 垂直于装甲板朝向的单位向量y分量
    
    // 测量1的值：装甲板在垂直方向上的投影应为0
    double z1 = offAxisX * armor_x + offAxisY * armor_y;
    
    // 测量1的测量矩阵 (7维)
    Eigen::RowVectorXd H1(STATE_DIM);
    H1 << offAxisX, offAxisY, 0.0, offAxisX * t, offAxisY * t, 0.0, 0.0;
    
    // 测量2: 装甲板到中心的向量在装甲板法向上的投影等于半径r (xy平面)
    // axisVector · armorToCenterVector = r
    double axisX = -sinYaw;     // 装甲板法向单位向量x分量
    double axisY = cosYaw;      // 装甲板法向单位向量y分量
    
    // 测量2的值：装甲板在法向上的投影应为r
    double z2 = axisX * armor_x + axisY * armor_y;
    
    // 测量2的测量矩阵 (7维)
    Eigen::RowVectorXd H2(STATE_DIM);
    H2 << axisX, axisY, 0.0, axisX * t, axisY * t, 0.0, -1.0;
    
    // 测量3: z轴测量 - 装甲板z坐标与中心z坐标的关系
    // armor_z = center_z + center_vz * t (假设装甲板在z方向没有相对运动)
    double z3 = armor_z;
    
    // 测量3的测量矩阵 (7维)
    Eigen::RowVectorXd H3(STATE_DIM);
    H3 << 0.0, 0.0, 1.0, 0.0, 0.0, t, 0.0;
    
    // 正则化测量: 保持r接近上一步的值
    double z4 = r_prev_;  // 使用上一步的r值作为正则化目标
    Eigen::RowVectorXd H4 = Eigen::RowVectorXd::Zero(STATE_DIM);
    H4(6) = 1.0;  // 只对r进行正则化
    
    // 更新测量1
    double weight1 = weight;
    double S1 = H1 * P_center_ * H1.transpose() + 1.0 / weight1;
    Eigen::VectorXd K1 = P_center_ * H1.transpose() / S1;
    double innovation1 = z1 - H1 * x_center_;
    x_center_ = x_center_ + K1 * innovation1;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    P_center_ = (I - K1 * H1) * P_center_ / lambda_;
    
    // 更新测量2
    double weight2 = weight;
    double S2 = H2 * P_center_ * H2.transpose() + 1.0 / weight2;
    Eigen::VectorXd K2 = P_center_ * H2.transpose() / S2;
    double innovation2 = z2 - H2 * x_center_;
    x_center_ = x_center_ + K2 * innovation2;
    P_center_ = (I - K2 * H2) * P_center_ / lambda_;
    
    // 更新测量3（z轴测量）
    double weight3 = weight;  // z轴测量权重
    double S3 = H3 * P_center_ * H3.transpose() + 1.0 / weight3;
    Eigen::VectorXd K3 = P_center_ * H3.transpose() / S3;
    double innovation3 = z3 - H3 * x_center_;
    x_center_ = x_center_ + K3 * innovation3;
    P_center_ = (I - K3 * H3) * P_center_ / lambda_;
    
    // 更新正则化测量
    double weight4 = regularization_weight_;  // 正则化权重
    double S4 = H4 * P_center_ * H4.transpose() + 1.0 / weight4;
    Eigen::VectorXd K4 = P_center_ * H4.transpose() / S4;
    double innovation4 = z4 - H4 * x_center_;
    x_center_ = x_center_ + K4 * innovation4;
    P_center_ = (I - K4 * H4) * P_center_ / lambda_;
    
    // 限制半径在合理范围内
    if (x_center_(6) < 100.0) x_center_(6) = 100.0;
    if (x_center_(6) > 600.0) x_center_(6) = 600.0;
}

void RotationMotionModel::updateCenterResult(double current_time) {
    center_x = x_center_(0) + x_center_(3) * current_time;
    center_y = x_center_(1) + x_center_(4) * current_time;
    center_z = x_center_(2) + x_center_(5) * current_time;
    center_vx = x_center_(3);
    center_vy = x_center_(4);
    center_vz = x_center_(5);
    // 更新半径值
    r = x_center_(6);
    r_prev_ = r;  // 保存当前r值用于下一次正则化
}

void RotationMotionModel::update(ObservedData& observedData) {
    observedDataHistory.push_back(observedData);
    if (observedDataHistory.size() > max_history) {
        observedDataHistory = std::vector<ObservedData>(
            observedDataHistory.end() - max_history, observedDataHistory.end());
    }
    last_observed_data = observedData;
    
    // 不再使用单独的线性回归计算z方向状态，统一使用指数衰减最小二乘
    
    // 使用指数衰减最小二乘更新所有状态（包括z轴）
    resetExponentialLS();
    double current_time = observedData.t;
    for (size_t i = 0; i < observedDataHistory.size(); ++i) {
        const auto& data = observedDataHistory[i];
        double time_offset = data.t - current_time;  // 相对于当前时间的时间偏移
        
        // 计算权重：时间越近权重越大
        double time_weight = std::exp(-std::abs(time_offset) * 0.1);
        
        updateExponentialLS(data.x, data.y, data.z, data.yaw, time_offset, time_weight);
    }
    
    // 获取当前中心状态
    updateCenterResult(0.0);  // 当前时刻
    
    // 使用EKF更新角度和角速度，传入xc, yc, r
    double dt = observedData.t - last_update_time_;
    if (dt > 0) {
        angle_ekf_->update(observedData.yaw, observedData.x, observedData.y, 
                          center_x, center_y, r, dt);
        last_update_time_ = observedData.t;
    }
    
    // 使用EKF状态拟合旋转参数
    fitRotationParameters();
}

void RotationMotionModel::emptyUpdate(double update_time) {
    PredictResult pred_data_to_update = predict(update_time - last_update_time_);
    ObservedData update_data({
        pred_data_to_update.armors[0].x,
        pred_data_to_update.armors[0].y,
        pred_data_to_update.armors[0].z,
        pred_data_to_update.armors[0].yaw,
        update_time
    });
    update(update_data);
}

void RotationMotionModel::fitRotationParameters() {
    // 从EKF获取角度和角速度
    double ekf_yaw = angle_ekf_->getYaw();
    double ekf_vyaw = angle_ekf_->getVyaw();
}

PredictResult RotationMotionModel::predict(double predictTime) {
    PredictResult result;
    
    // 使用原始方法预测平移状态
    result.center_x = center_x + predictTime * center_vx;
    result.center_y = center_y + predictTime * center_vy;
    result.center_z = center_z + predictTime * center_vz;
    result.r = r;
    
    // 使用EKF预测角度
    if (angle_ekf_->isInitialized()) {
        double ekf_yaw = angle_ekf_->getYaw();
        double ekf_vyaw = angle_ekf_->getVyaw();
        result.yaw = ekf_yaw + ekf_vyaw * predictTime;
        
        // 处理角度环绕
        if (result.yaw > M_PI) result.yaw -= 2.0 * M_PI;
        if (result.yaw < -M_PI) result.yaw += 2.0 * M_PI;
    } else {
        result.yaw = last_observed_data.yaw;
    }
    
    result.rotation_direction = rotation_direction;
    
    // 生成装甲板预测
    for (int i = 0; i < n_armors; i++) {
        double armor_yaw = result.yaw - i * rotation_direction * jump_rad;
        result.armors.push_back(SimpleArmor({
            result.center_x + r * std::sin(armor_yaw),
            result.center_y - r * std::cos(armor_yaw),
            result.center_z,
            armor_yaw
        }));
    }

    return result;
}

RotationMotionState RotationMotionModel::getState() {
    RotationMotionState state;
    state.center_vx = center_vx;
    state.center_vy = center_vy;
    state.center_vz = center_vz;
    state.r = r;
    state.center_x = center_x;
    state.center_y = center_y;
    state.center_z = center_z;
    
    if (angle_ekf_->isInitialized()) {
        state.vyaw = angle_ekf_->getVyaw();  // 使用EKF的角速度
    } else {
        state.vyaw = 0.0;
    }
    
    return state;
}

double RotationMotionModel::getCamToCenterYaw() {
    std::vector<float> cam_center = rest_frame_ -> getCamPosition();
    std::vector<double> cam_to_rotation_center_vector = {center_x - cam_center[0], center_y - cam_center[1]};
    double cam_to_rotation_center_yaw = std::atan2(-cam_to_rotation_center_vector[0], cam_to_rotation_center_vector[1]);
    return cam_to_rotation_center_yaw;
}

double RotationMotionModel::getTheoreticYaw(double armor_x, double armor_y) {
    double theoreticYawFacingArmor = getTheoreticYawFacingArmor(armor_x, armor_y);
    std::vector<float> cam_center = rest_frame_ -> getCamPosition();
    std::vector<double> cam_to_rotation_center_vector = {center_x - cam_center[0], center_y - cam_center[1]};
    double cam_to_rotation_center_yaw = getCamToCenterYaw();
    return cam_to_rotation_center_yaw + theoreticYawFacingArmor;
}

double RotationMotionModel::getTheoreticYawFacingArmor(double armor_x, double armor_y) {
    std::vector<float> cam_center = rest_frame_ -> getCamPosition();
    std::vector<double> cam_to_rotation_center_vector = {center_x - cam_center[0], center_y - cam_center[1]};
    double cam_to_rotation_center_vector_len = std::sqrt(cam_to_rotation_center_vector[0] * cam_to_rotation_center_vector[0] + cam_to_rotation_center_vector[1] * cam_to_rotation_center_vector[1]);
    std::vector<double> right_unit_v = {cam_to_rotation_center_vector[1] / cam_to_rotation_center_vector_len, - cam_to_rotation_center_vector[0] / cam_to_rotation_center_vector_len};
    std::vector<double> center_armor_v = {armor_x - center_x, armor_y - center_y};
    double right_shift = right_unit_v[0] * center_armor_v[0] + right_unit_v[1] * center_armor_v[1];
    if (right_shift > r) {
        return M_PI / 2.0;
    }
    if (right_shift < -r) {
        return - M_PI / 2.0;
    }
    return std::asin(right_shift / r);
}
