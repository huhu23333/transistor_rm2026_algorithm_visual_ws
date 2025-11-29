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
    
    // 使用原始方法更新平移状态
    LinearRegressionResult zResult = linearRegression(tData, zData);
    center_z = zResult.a;
    center_vz = zResult.b;
    
    fitCenterXYResult centerResult = fitCenterXY(yawData, xData, yData, tData, r);
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

fitCenterXYResult RotationMotionModel::fitCenterXY(const std::vector<double>& armorYaw,
                                                  const std::vector<double>& armorX,
                                                  const std::vector<double>& armorY,
                                                  const std::vector<double>& dataT,
                                                  double lastR) {
    int n = armorYaw.size();
    
    // 构建轴向量
    std::vector<double> axisX(n), axisY(n);
    std::vector<double> offAxisX(n), offAxisY(n);
    
    for (int i = 0; i < n; i++) {
        double cosYaw = cos(armorYaw[i]);
        double sinYaw = sin(armorYaw[i]);
        axisX[i] = -sinYaw;
        axisY[i] = cosYaw;
        offAxisX[i] = axisY[i];
        offAxisY[i] = -axisX[i];
    }
    
    // 构建设计矩阵
    MatrixXd A(n, 4);
    VectorXd b(n);
    
    for (int i = 0; i < n; i++) {
        A(i, 0) = offAxisX[i];
        A(i, 1) = offAxisY[i];
        A(i, 2) = offAxisX[i] * dataT[i];
        A(i, 3) = offAxisY[i] * dataT[i];
        b(i) = offAxisX[i] * armorX[i] + offAxisY[i] * armorY[i];
    }
    
    // 使用SVD求解
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    VectorXd x = svd.solve(b);
    
    fitCenterXYResult result;
    result.center_x = x(0);
    result.center_y = x(1);
    result.center_vx = x(2);
    result.center_vy = x(3);
    
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