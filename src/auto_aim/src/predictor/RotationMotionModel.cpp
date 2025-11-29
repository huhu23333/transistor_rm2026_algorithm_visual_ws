// RotationMotionModel.cpp
#include "predictor/RotationMotionModel.h"

using namespace Eigen;

RotationMotionModel::RotationMotionModel(ObservedData& initObservedData, 
                                       std::shared_ptr<RestFrame> rest_frame_, 
                                       bool is_outpost)
    : rest_frame_(rest_frame_), is_outpost(is_outpost) {
    
    // 初始化EKF
    trans_radius_ekf_ = std::make_unique<TranslationRadiusEKF>();
    angle_ekf_ = std::make_unique<AngleEKF>();
    last_update_time_ = initObservedData.t;
    
    // 配置参数
    if (is_outpost) {
        n_armors = 3;
        delta_phase = 25.0 * M_PI / 180.0;
    } else {
        n_armors = 4;
        delta_phase = 25.0 * M_PI / 180.0;
    }
    
    max_history = 90;
    rotation_period = 0.0;
    current_phase = 0.0;
    rotation_direction = 1;
    jump_rad = M_PI * 2.0 / n_armors;
    
    // 使用初始观测数据初始化EKF
    trans_radius_ekf_->update(initObservedData.x, initObservedData.y, 
                             initObservedData.z, initObservedData.yaw, 
                             initObservedData.t);
    angle_ekf_->initialize(initObservedData.yaw);
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

void RotationMotionModel::update(ObservedData& observedData) {
    // 第一步：使用观测数据更新平移和半径EKF
    // 注意：这里直接使用观测的yaw，即使有噪声
    trans_radius_ekf_->update(observedData.x, observedData.y, observedData.z, 
                             observedData.yaw, observedData.t);
    
    // 第二步：使用更新后的中心位置和半径，以及观测数据更新角度EKF
    if (trans_radius_ekf_->isInitialized()) {
        double xc = trans_radius_ekf_->getCenterX();
        double yc = trans_radius_ekf_->getCenterY();
        double r = trans_radius_ekf_->getRadius();
        double dt = observedData.t - last_update_time_;
        if (dt > 0) {
            angle_ekf_->update(observedData.yaw, observedData.x, observedData.y, 
                               xc, yc, r, dt);
        }
    }
    
    last_update_time_ = observedData.t;
    
    // 拟合旋转参数
    if (angle_ekf_->isInitialized()) {
        double ekf_vyaw = angle_ekf_->getVyaw();
        rotation_direction = (ekf_vyaw >= 0) ? 1 : -1;
        
        if (std::abs(ekf_vyaw) > 1e-5) {
            rotation_period = 2.0 * M_PI / std::abs(ekf_vyaw);
        } else {
            rotation_period = 0.0;
        }
        
        // 更新当前相位
        current_phase = angle_ekf_->getYaw();
    }
}

PredictResult RotationMotionModel::predict(double predictTime) {
    PredictResult result;
    
    // 预测平移和半径状态
    if (trans_radius_ekf_->isInitialized()) {
        double current_time = last_update_time_;
        trans_radius_ekf_->predict(current_time + predictTime);
        
        result.center_x = trans_radius_ekf_->getCenterX();
        result.center_y = trans_radius_ekf_->getCenterY();
        result.center_z = trans_radius_ekf_->getCenterZ();
        result.r = trans_radius_ekf_->getRadius();
    } else {
        // 回退到默认值
        result.center_x = 0.0;
        result.center_y = 0.0;
        result.center_z = 0.0;
        result.r = 250.0;
    }
    
    // 预测角度
    if (angle_ekf_->isInitialized()) {
        double dt = predictTime;
        angle_ekf_->predict(dt);
        result.yaw = angle_ekf_->getYaw();
        
        // 处理角度环绕
        if (result.yaw > M_PI) result.yaw -= 2.0 * M_PI;
        if (result.yaw < -M_PI) result.yaw += 2.0 * M_PI;
    } else {
        result.yaw = 0.0;
    }
    
    result.rotation_direction = rotation_direction;
    
    // 生成装甲板预测
    for (int i = 0; i < n_armors; i++) {
        double armor_yaw = result.yaw - i * rotation_direction * jump_rad;
        result.armors.push_back(SimpleArmor({
            result.center_x + result.r * std::sin(armor_yaw),
            result.center_y - result.r * std::cos(armor_yaw),
            result.center_z,
            armor_yaw
        }));
    }

    return result;
}

// 其他方法保持不变...
double RotationMotionModel::getJumpPeriod() {
    return jump_period_frames;
}

RotationMotionState RotationMotionModel::getState() {
    RotationMotionState state;
    
    if (trans_radius_ekf_->isInitialized()) {
        state.center_x = trans_radius_ekf_->getCenterX();
        state.center_y = trans_radius_ekf_->getCenterY();
        state.center_z = trans_radius_ekf_->getCenterZ();
        state.center_vx = trans_radius_ekf_->getVelocityX();
        state.center_vy = trans_radius_ekf_->getVelocityY();
        state.center_vz = trans_radius_ekf_->getVelocityZ();
        state.r = trans_radius_ekf_->getRadius();
    }
    
    if (angle_ekf_->isInitialized()) {
        state.vyaw = angle_ekf_->getVyaw();
    } else {
        state.vyaw = 0.0;
    }
    
    return state;
}

double RotationMotionModel::getCamToCenterYaw() {
    std::vector<float> cam_center = rest_frame_->getCamPosition();
    double center_x = trans_radius_ekf_->getCenterX();
    double center_y = trans_radius_ekf_->getCenterY();
    
    std::vector<double> cam_to_rotation_center_vector = {
        center_x - cam_center[0], 
        center_y - cam_center[1]
    };
    double cam_to_rotation_center_yaw = std::atan2(
        -cam_to_rotation_center_vector[0], 
        cam_to_rotation_center_vector[1]
    );
    return cam_to_rotation_center_yaw;
}

double RotationMotionModel::getTheoreticYaw(double armor_x, double armor_y) {
    double theoreticYawFacingArmor = getTheoreticYawFacingArmor(armor_x, armor_y);
    double cam_to_rotation_center_yaw = getCamToCenterYaw();
    return cam_to_rotation_center_yaw + theoreticYawFacingArmor;
}

double RotationMotionModel::getTheoreticYawFacingArmor(double armor_x, double armor_y) {
    std::vector<float> cam_center = rest_frame_->getCamPosition();
    double center_x = trans_radius_ekf_->getCenterX();
    double center_y = trans_radius_ekf_->getCenterY();
    double r = trans_radius_ekf_->getRadius();
    
    std::vector<double> cam_to_rotation_center_vector = {
        center_x - cam_center[0], 
        center_y - cam_center[1]
    };
    double cam_to_rotation_center_vector_len = std::sqrt(
        cam_to_rotation_center_vector[0] * cam_to_rotation_center_vector[0] + 
        cam_to_rotation_center_vector[1] * cam_to_rotation_center_vector[1]
    );
    
    std::vector<double> right_unit_v = {
        cam_to_rotation_center_vector[1] / cam_to_rotation_center_vector_len, 
        -cam_to_rotation_center_vector[0] / cam_to_rotation_center_vector_len
    };
    
    std::vector<double> center_armor_v = {
        armor_x - center_x, 
        armor_y - center_y
    };
    
    double right_shift = right_unit_v[0] * center_armor_v[0] + right_unit_v[1] * center_armor_v[1];
    
    if (right_shift > r) {
        return M_PI / 2.0;
    }
    if (right_shift < -r) {
        return -M_PI / 2.0;
    }
    return std::asin(right_shift / r);
}

