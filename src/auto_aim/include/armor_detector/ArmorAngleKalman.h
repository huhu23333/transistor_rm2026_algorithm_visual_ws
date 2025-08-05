#ifndef ARMOR_ANGLE_KALMAN_H
#define ARMOR_ANGLE_KALMAN_H

#include <opencv2/opencv.hpp>

class ArmorAngleKalman {
public:
    bool isInitialized() const;
    explicit ArmorAngleKalman(float dt = 0.1f); // 默认100ms时间间隔
    void reconfigure(float dt, float process_noise, float measure_noise);
    void updateKalmanFilter(const cv::Point3f& measured_point);
    cv::Point3f predictKalmanFilter(float predict_time = 0.0f);
    void reset(const cv::Point3f& initial_pos = cv::Point3f(0,0,0));

private:
    cv::KalmanFilter kalman_filter_;
    float dt_; // 时间间隔
    float last_predict_time_ = 0.0f;
};

#endif // ARMOR_ANGLE_KALMAN_H
