// RotationMotionModel.cpp
#include "predictor/RotationMotionModel.h"

using namespace Eigen;

RotationMotionModel::RotationMotionModel(ObservedData& initObservedData, std::shared_ptr<RestFrame> rest_frame_, bool is_outpost)
    : rest_frame_(rest_frame_), is_outpost(is_outpost) {
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
        delta_phase = 25.0 * M_PI / 180.0;
    } else {
        n_armors = 4;  // 4装甲板模式
        r = 250.0;
        delta_phase = 25.0 * M_PI / 180.0;
    }
    
    r_prev_ = r;  // 初始化历史半径值
    regularization_weight_ = 10.0;  // 正则化权重，可调整
    
    // 初始化中心位置
    center_x = initObservedData.x - r * sin(initObservedData.yaw);
    center_y = initObservedData.y + r * cos(initObservedData.yaw);
    center_z = initObservedData.z;
    
    max_history = 90;
    rotation_period = 0.0;
    current_phase = 0.0;
    rotation_direction = 1;
    jump_rad = M_PI * 2.0 / n_armors;
    
    // 初始化指数衰减最小二乘
    initializeExponentialLS();
}

void RotationMotionModel::initializeExponentialLS() {
    // 初始化5x5协方差矩阵
    P_center_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 1000.0;
    
    // 初始状态估计 [center_x, center_y, center_vx, center_vy, r]
    x_center_ = Eigen::VectorXd::Zero(STATE_DIM);
    x_center_(0) = center_x;  // center_x
    x_center_(1) = center_y;  // center_y
    x_center_(2) = 0.0;       // center_vx
    x_center_(3) = 0.0;       // center_vy
    x_center_(4) = r;         // r
    
    // 遗忘因子，值越小遗忘越快
    lambda_ = 0.95;
    center_initialized_ = true;
}

void RotationMotionModel::updateExponentialLS(double armor_x, double armor_y, double armor_yaw, double t, double weight) {
    if (!center_initialized_) {
        initializeExponentialLS();
    }
    
    // 构建两个测量方程
    
    // 测量1: 装甲板到中心的向量与装甲板朝向垂直
    // offAxisVector · armorToCenterVector = 0
    double cosYaw = cos(armor_yaw);
    double sinYaw = sin(armor_yaw);
    double offAxisX = cosYaw;   // 垂直于装甲板朝向的单位向量x分量
    double offAxisY = sinYaw;   // 垂直于装甲板朝向的单位向量y分量
    
    // 测量1的值：装甲板在垂直方向上的投影应为0
    double z1 = offAxisX * armor_x + offAxisY * armor_y;
    
    // 测量1的测量矩阵
    Eigen::RowVectorXd H1(STATE_DIM);
    H1 << offAxisX, offAxisY, offAxisX * t, offAxisY * t, 0.0;
    
    // 测量2: 装甲板到中心的向量在装甲板法向上的投影等于半径r
    // axisVector · armorToCenterVector = r
    double axisX = -sinYaw;     // 装甲板法向单位向量x分量
    double axisY = cosYaw;      // 装甲板法向单位向量y分量
    
    // 测量2的值：装甲板在法向上的投影应为r
    double z2 = axisX * armor_x + axisY * armor_y;
    
    // 测量2的测量矩阵
    Eigen::RowVectorXd H2(STATE_DIM);
    H2 << axisX, axisY, axisX * t, axisY * t, -1.0;
    
    // 正则化测量: 保持r接近上一步的值
    double z3 = r_prev_;  // 使用上一步的r值作为正则化目标
    Eigen::RowVectorXd H3 = Eigen::RowVectorXd::Zero(STATE_DIM);
    H3(4) = 1.0;  // 只对r进行正则化
    
    // 更新测量1
    double S1 = H1 * P_center_ * H1.transpose() + 1.0 / weight;
    Eigen::VectorXd K1 = P_center_ * H1.transpose() / S1;
    double innovation1 = z1 - H1 * x_center_;
    x_center_ = x_center_ + K1 * innovation1;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    P_center_ = (I - K1 * H1) * P_center_ / lambda_;
    
    // 更新测量2（赋予较大权重，因为这是半径的主要约束）
    double weight2 = weight * 1.0;  // 给第二个测量更大的权重
    double S2 = H2 * P_center_ * H2.transpose() + 1.0 / weight2;
    Eigen::VectorXd K2 = P_center_ * H2.transpose() / S2;
    double innovation2 = z2 - H2 * x_center_;
    x_center_ = x_center_ + K2 * innovation2;
    P_center_ = (I - K2 * H2) * P_center_ / lambda_;
    
    // 更新正则化测量
    double weight3 = regularization_weight_;  // 正则化权重
    double S3 = H3 * P_center_ * H3.transpose() + 1.0 / weight3;
    Eigen::VectorXd K3 = P_center_ * H3.transpose() / S3;
    double innovation3 = z3 - H3 * x_center_;
    x_center_ = x_center_ + K3 * innovation3;
    P_center_ = (I - K3 * H3) * P_center_ / lambda_;
    
    // 限制半径在合理范围内
    if (x_center_(4) < 100.0) x_center_(4) = 100.0;
    if (x_center_(4) > 600.0) x_center_(4) = 600.0;
}

fitCenterXYResult RotationMotionModel::getCenterResult(double current_time) {
    fitCenterXYResult result;
    
    if (center_initialized_) {
        // 使用状态向量计算当前时刻的中心位置
        result.center_x = x_center_(0) + x_center_(2) * current_time;
        result.center_y = x_center_(1) + x_center_(3) * current_time;
        result.center_vx = x_center_(2);
        result.center_vy = x_center_(3);
        
        // 更新半径值
        r = x_center_(4);
        r_prev_ = r;  // 保存当前r值用于下一次正则化
    } else {
        // 回退到简单估计
        result.center_x = center_x;
        result.center_y = center_y;
        result.center_vx = center_vx;
        result.center_vy = center_vy;
    }
    
    return result;
}

void RotationMotionModel::update(ObservedData& observedData) {
    observedDataHistory.push_back(observedData);
    if (observedDataHistory.size() > max_history) {
        observedDataHistory = std::vector<ObservedData>(
            observedDataHistory.end() - max_history, observedDataHistory.end());
    }
    
    auto params = getParams();
    std::vector<double> tData = params[0];
    std::vector<double> xData = params[1];
    std::vector<double> yData = params[2];
    std::vector<double> zData = params[3];
    std::vector<double> yawData = params[4];
    
    // 使用原始方法更新z方向状态
    LinearRegressionResult zResult = linearRegression(tData, zData);
    center_z = zResult.a;
    center_vz = zResult.b;
    
    // 使用指数衰减最小二乘更新xy平面状态和半径
    double current_time = observedData.t;
    for (size_t i = 0; i < observedDataHistory.size(); ++i) {
        const auto& data = observedDataHistory[i];
        double time_offset = data.t - current_time;  // 相对于当前时间的时间偏移
        
        // 计算权重：时间越近权重越大
        double time_weight = std::exp(-std::abs(time_offset) * 0.1);
        
        updateExponentialLS(data.x, data.y, data.yaw, time_offset, time_weight);
    }
    
    // 获取当前中心状态
    fitCenterXYResult centerResult = getCenterResult(0.0);  // 当前时刻
    center_x = centerResult.center_x;
    center_y = centerResult.center_y;
    center_vx = centerResult.center_vx;
    center_vy = centerResult.center_vy;
    
    // 移除原有的 calculateR() 调用，因为r已经在updateExponentialLS中更新
    
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
    if (observedDataHistory.empty()) return;
    
    ObservedData last_observed_data = observedDataHistory.back();
    PredictResult pred_data_to_update = predict(update_time - last_observed_data.t);
    ObservedData update_data({
        pred_data_to_update.armors[0].x,
        pred_data_to_update.armors[0].y,
        pred_data_to_update.armors[0].z,
        pred_data_to_update.armors[0].yaw,
        update_time
    });
    update(update_data);
}

std::vector<std::vector<double>> RotationMotionModel::getParams() {
    std::vector<double> tData, xData, yData, zData, yawData;
    double lastT = observedDataHistory.back().t;
    
    for (const auto& data : observedDataHistory) {
        tData.push_back(data.t - lastT);
        xData.push_back(data.x);
        yData.push_back(data.y);
        zData.push_back(data.z);
        yawData.push_back(data.yaw);
    }
    
    return {tData, xData, yData, zData, yawData};
}

void RotationMotionModel::fitRotationParameters() {
    if (observedDataHistory.size() < 10) return;
    
    // 从EKF获取角度和角速度
    double ekf_yaw = angle_ekf_->getYaw();
    double ekf_vyaw = angle_ekf_->getVyaw();
    
    // 使用EKF的角速度计算旋转参数
    rotation_direction = (ekf_vyaw >= 0) ? 1 : -1;
    
    if (std::abs(ekf_vyaw) > 1e-5) {
        rotation_period = 2.0 * M_PI / std::abs(ekf_vyaw);
    } else {
        rotation_period = 0.0;
    }
    
    // 基于旋转周期计算跳变周期
    if (rotation_period > 0) {
        double avg_interval = 0.0;
        for (size_t i = 1; i < observedDataHistory.size(); i++) {
            avg_interval += (observedDataHistory[i].t - observedDataHistory[i-1].t);
        }
        avg_interval /= (observedDataHistory.size() - 1);
        
        jump_period_frames = rotation_period / n_armors / avg_interval;
    }
    
    // 更新当前相位
    current_phase = ekf_yaw;
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
        result.yaw = observedDataHistory.back().yaw;
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

int RotationMotionModel::getRotationDirection(const std::vector<double>& yawData) {
    if (yawData.size() < 2) return 1;
    
    double d_yaw_integrate = 0.0;
    for (int first_idx = 0; first_idx < yawData.size() - 1; first_idx += 1) {
        d_yaw_integrate += 0.1 * (yawData[first_idx + 1] - yawData[first_idx]) / 
                           ((yawData[first_idx + 1] - yawData[first_idx]) * (yawData[first_idx + 1] - yawData[first_idx]) + 0.01);
    }
    
    return (d_yaw_integrate > 0) ? 1 : -1;
}

double RotationMotionModel::getJumpPeriod() {
    return jump_period_frames;
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