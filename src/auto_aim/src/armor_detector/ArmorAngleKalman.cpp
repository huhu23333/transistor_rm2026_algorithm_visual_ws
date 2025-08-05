// ArmorAngleKalman.cpp 修改实现
#include "armor_detector/ArmorAngleKalman.h"

ArmorAngleKalman::ArmorAngleKalman(float dt) : dt_(dt) {
    // 状态变量: [x, y, z, vx, vy, vz]
    kalman_filter_ = cv::KalmanFilter(6, 3, 0);
    
    // 正确状态转移矩阵 (包含时间参数)
    kalman_filter_.transitionMatrix = (cv::Mat_<float>(6,6) <<
        1, 0, 0, dt, 0, 0,
        0, 1, 0, 0, dt, 0,
        0, 0, 1, 0, 0, dt,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);

    // 观测矩阵 [I|0]
    kalman_filter_.measurementMatrix = cv::Mat::zeros(3,6,CV_32F);
    kalman_filter_.measurementMatrix.at<float>(0,0) = 1;
    kalman_filter_.measurementMatrix.at<float>(1,1) = 1;
    kalman_filter_.measurementMatrix.at<float>(2,2) = 1;

    // // 过程噪声协方差 (Q)
    // cv::setIdentity(kalman_filter_.processNoiseCov, cv::Scalar::all(1e+21));
    // // 观测噪声协方差 (R)
    // cv::setIdentity(kalman_filter_.measurementNoiseCov, cv::Scalar::all(1e-30));
    // // 初始后验误差协方差
    // cv::setIdentity(kalman_filter_.errorCovPost, cv::Scalar::all(0.1));

    // 过程噪声协方差 (Q)
    cv::setIdentity(kalman_filter_.processNoiseCov, cv::Scalar::all(1e+21));
    // 观测噪声协方差 (R)
    cv::setIdentity(kalman_filter_.measurementNoiseCov, cv::Scalar::all(1e-20));
    // 初始后验误差协方差
    cv::setIdentity(kalman_filter_.errorCovPost, cv::Scalar::all(0.1));
}

bool ArmorAngleKalman::isInitialized() const {
    // 检查状态向量是否非零（或根据需求自定义逻辑）
    return cv::norm(kalman_filter_.statePost) > 1e-6;
}

void ArmorAngleKalman::reconfigure(float dt, float process_noise, float measure_noise) {
    dt_ = dt;
    // 更新转移矩阵时间参数
    kalman_filter_.transitionMatrix.at<float>(0,3) = dt;
    kalman_filter_.transitionMatrix.at<float>(1,4) = dt;
    kalman_filter_.transitionMatrix.at<float>(2,5) = dt;
    // 调整噪声参数
    kalman_filter_.processNoiseCov *= process_noise;
    kalman_filter_.measurementNoiseCov *= measure_noise;
}

void ArmorAngleKalman::updateKalmanFilter(const cv::Point3f& measured_point) {
    cv::Mat measurement = (cv::Mat_<float>(3,1) << 
        measured_point.x, measured_point.y, measured_point.z);
    kalman_filter_.correct(measurement);
}

void ArmorAngleKalman::reset(const cv::Point3f& initial_pos) {
    // 重置状态向量 [x, y, z, vx, vy, vz]
    kalman_filter_.statePost = (cv::Mat_<float>(6,1) << 
        initial_pos.x, 
        initial_pos.y, 
        initial_pos.z, 
        0.0f,  // 初始速度设为0
        0.0f, 
        0.0f
    );
    
    // 重置协方差矩阵
    const float INITIAL_POS_VAR = 100.0f;  // 位置初始方差 (mm^2)
    const float INITIAL_VEL_VAR = 1.0f;    // 速度初始方差 (mm/s)^2
    cv::setIdentity(kalman_filter_.errorCovPost, cv::Scalar::all(INITIAL_POS_VAR));
    kalman_filter_.errorCovPost.at<float>(3,3) = INITIAL_VEL_VAR;
    kalman_filter_.errorCovPost.at<float>(4,4) = INITIAL_VEL_VAR;
    kalman_filter_.errorCovPost.at<float>(5,5) = INITIAL_VEL_VAR;
}




cv::Point3f ArmorAngleKalman::predictKalmanFilter(float predict_time) {
    if(predict_time <= 0) {
        cv::Mat pred = kalman_filter_.predict();
        return cv::Point3f(pred.at<float>(0), pred.at<float>(1), pred.at<float>(2));
    }
    
    // 保存原始参数
    cv::Mat orig_trans = kalman_filter_.transitionMatrix.clone();
    const int steps = std::ceil(predict_time / dt_);
    
    // 调整转移矩阵进行多步预测
    kalman_filter_.transitionMatrix.at<float>(0,3) = dt_ * steps;
    kalman_filter_.transitionMatrix.at<float>(1,4) = dt_ * steps; 
    kalman_filter_.transitionMatrix.at<float>(2,5) = dt_ * steps;
    
    cv::Mat pred = kalman_filter_.predict();
    // 恢复原始参数
    kalman_filter_.transitionMatrix = orig_trans;
    
    return cv::Point3f(pred.at<float>(0), pred.at<float>(1), pred.at<float>(2));
}