// PredictionTrans.h
#ifndef PREDICTION_TRANS_H
#define PREDICTION_TRANS_H

#include <opencv2/opencv.hpp>
#include "armor_detector/Armor.h"
#include <vector>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <memory>

class Trans2DPredTo3DClass {
public:
    Trans2DPredTo3DClass(std::shared_ptr<YAML::Node> config_file_ptr);
    ~Trans2DPredTo3DClass();
    cv::Point3f trans2DPredTo3D(ArmorResult& best_result, cv::Point3f aim_position, std::vector<ArmorResult>& fourierPredictions,
                                float total_delay, float current_fps);

private:
    
    float yaw_rad_to_x_pixel_ratio;
    float pitch_rad_to_y_pixel_ratio;
    std::shared_ptr<YAML::Node> config_file_ptr;
};

#endif // PREDICTION_TRANS_H