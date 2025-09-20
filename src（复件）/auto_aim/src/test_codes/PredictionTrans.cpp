// PredictionTrans.cpp
#include "test_codes/PredictionTrans.h"

Trans2DPredTo3DClass::Trans2DPredTo3DClass(std::shared_ptr<YAML::Node> config_file_ptr) 
: config_file_ptr(config_file_ptr) {

    yaw_rad_to_x_pixel_ratio = (*config_file_ptr)["yaw_rad_to_x_pixel_ratio"].as<float>(); 
    pitch_rad_to_y_pixel_ratio = (*config_file_ptr)["pitch_rad_to_y_pixel_ratio"].as<float>(); 
    
}

Trans2DPredTo3DClass::~Trans2DPredTo3DClass() {}

cv::Point3f Trans2DPredTo3DClass::trans2DPredTo3D(ArmorResult& best_result, cv::Point3f aim_position,
                                      std::vector<ArmorResult>& fourierPredictions,
                                      float total_delay, float current_fps) {
    int number = best_result.number;
    ArmorResult fourierPrediction = fourierPredictions[number];
    int max_predict_index = fourierPrediction.predictions.size() - 1;
    if (max_predict_index < 0) {
        return aim_position;
    }
    int predict_index = static_cast<int>(total_delay * current_fps) + 1;
    predict_index = std::min(predict_index, max_predict_index);
    cv::Point2f pred_2D_delta = fourierPrediction.predictions[predict_index] - fourierPrediction.predictions[0];
    float delta_yaw = pred_2D_delta.x / yaw_rad_to_x_pixel_ratio;
    float delta_pitch = pred_2D_delta.y / pitch_rad_to_y_pixel_ratio;
    cv::Point3f result_position(aim_position.x - delta_yaw * aim_position.z, aim_position.y - delta_pitch * aim_position.z, aim_position.z);
    return result_position;
}