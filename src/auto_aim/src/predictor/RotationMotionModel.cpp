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
    // 初始化协方差矩阵为较大的值，表示初始不确定性
    P_center_ = Eigen::Matrix4d::Identity() * 1000.0;
    
    // 初始状态估计
    x_center_ = Eigen::Vector4d::Zero();
    x_center_(0) = center_x;  // center_x
    x_center_(1) = center_y;  // center_y
    x_center_(2) = 0.0;       // center_vx
    x_center_(3) = 0.0;       // center_vy
    
    // 遗忘因子，值越小遗忘越快
    lambda_ = 0.95;
    center_initialized_ = true;
}

void RotationMotionModel::updateExponentialLS(double armor_x, double armor_y, double armor_yaw, double t, double weight) {
    if (!center_initialized_) {
        initializeExponentialLS();
    }
    
    // 构建测量向量和矩阵
    double cosYaw = cos(armor_yaw);
    double sinYaw = sin(armor_yaw);
    double offAxisX = cosYaw;   // 垂直于装甲板朝向的单位向量x分量
    double offAxisY = sinYaw;   // 垂直于装甲板朝向的单位向量y分量
    
    // 测量值：装甲板在垂直方向上的投影
    double z = offAxisX * armor_x + offAxisY * armor_y;
    
    // 测量矩阵
    Eigen::RowVector4d H;
    H << offAxisX, offAxisY, offAxisX * t, offAxisY * t;
    
    // 计算卡尔曼增益
    double S = H * P_center_ * H.transpose() + 1.0 / weight;  // 测量噪声方差倒数作为权重
    Eigen::Vector4d K = P_center_ * H.transpose() / S;
    
    // 计算新息（测量残差）
    double innovation = z - H * x_center_;
    
    // 更新状态估计
    x_center_ = x_center_ + K * innovation;
    
    // 更新协方差矩阵（带指数衰减）
    Eigen::Matrix4d I = Eigen::Matrix4d::Identity();
    P_center_ = (I - K * H) * P_center_ / lambda_;
}

fitCenterXYResult RotationMotionModel::getCenterResult(double current_time) {
    fitCenterXYResult result;
    
    if (center_initialized_) {
        // 使用状态向量计算当前时刻的中心位置
        result.center_x = x_center_(0) + x_center_(2) * current_time;
        result.center_y = x_center_(1) + x_center_(3) * current_time;
        result.center_vx = x_center_(2);
        result.center_vy = x_center_(3);
    } else {
        // 回退到简单估计
        result.center_x = center_x;
        result.center_y = center_y;
        result.center_vx = center_vx;
        result.center_vy = center_vy;
    }
    
    return result;
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

// 修改update方法
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
    
    // 使用指数衰减最小二乘更新xy平面状态
    // 对最近的观测数据应用指数衰减最小二乘更新
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
    
    if (!is_outpost) {
        calculateR();
    }
    
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

void RotationMotionModel::calculateR() {
    auto params = getParams();
    std::vector<double> tData = params[0];
    std::vector<double> xData = params[1];
    std::vector<double> yData = params[2];
    
    double sum = 0.0;
    int n = tData.size();
    for (int i = 0; i < n; i++) {
        double dx = xData[i] - (center_x + center_vx * tData[i]);
        double dy = yData[i] - (center_y + center_vy * tData[i]);
        sum += sqrt(dx * dx + dy * dy);
    }
    
    r = sum / n;
    r = std::max(std::min(r, 600.0), 100.0);
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

// fitCenterXYResult RotationMotionModel::fitCenterXY(const std::vector<double>& armorYaw,
//                                                   const std::vector<double>& armorX,
//                                                   const std::vector<double>& armorY,
//                                                   const std::vector<double>& dataT,
//                                                   double lastR) {
//     int n = armorYaw.size();
    
//     // 构建轴向量
//     std::vector<double> axisX(n), axisY(n);
//     std::vector<double> offAxisX(n), offAxisY(n);
    
//     for (int i = 0; i < n; i++) {
//         double cosYaw = cos(armorYaw[i]);
//         double sinYaw = sin(armorYaw[i]);
//         axisX[i] = -sinYaw;
//         axisY[i] = cosYaw;
//         offAxisX[i] = axisY[i];
//         offAxisY[i] = -axisX[i];
//     }
    
//     // 构建设计矩阵
//     MatrixXd A(n, 4);
//     VectorXd b(n);
    
//     for (int i = 0; i < n; i++) {
//         A(i, 0) = offAxisX[i];
//         A(i, 1) = offAxisY[i];
//         A(i, 2) = offAxisX[i] * dataT[i];
//         A(i, 3) = offAxisY[i] * dataT[i];
//         b(i) = offAxisX[i] * armorX[i] + offAxisY[i] * armorY[i];
//     }
    
//     // 使用SVD求解
//     JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
//     VectorXd x = svd.solve(b);
    
//     fitCenterXYResult result;
//     result.center_x = x(0);
//     result.center_y = x(1);
//     result.center_vx = x(2);
//     result.center_vy = x(3);
    
//     return result;
// }

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