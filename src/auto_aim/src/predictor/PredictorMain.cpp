#include "predictor/PredictorMain.h"

void PredictorMain::update_serial_info(float bullet_velocity, float last_pitch_rad_delayed, float last_yaw_rad_delayed, float total_yaw_rad_delayed) {
    for (std::shared_ptr<AllPredictor>& all_predictor : all_predictors_) {
        if (all_predictor) {
            all_predictor -> update_serial_info(bullet_velocity, last_pitch_rad_delayed, last_yaw_rad_delayed, total_yaw_rad_delayed);
        }
    }
}

PredictorResult PredictorMain::step(std::vector<ArmorResult>& classifyResults, cv::Mat& frame, PredictorType::PredictorType predictor_type, ArmorType::ArmorType priority_armor, bool auto_aim_switch) {

    if (last_auto_aim_switch == false && auto_aim_switch == true) {
        for (size_t all_predictors_index = 0; all_predictors_index < classify_classes; all_predictors_index++) {
            all_predictors_[all_predictors_index] -> reset_integration();
        }
    }
    last_auto_aim_switch = auto_aim_switch;

    std::vector<std::vector<ArmorResult>> classified_classifyResults(classify_classes);
    for (ArmorResult& classify_result : classifyResults) {
        classified_classifyResults[classify_result.number].push_back(classify_result);
    }
    // todo 前哨站/基地特殊处理
    std::vector<PredictorResult> classified_predictor_results;
    for (size_t all_predictors_index = 0; all_predictors_index < classify_classes; all_predictors_index++) {
        if (classified_classifyResults[all_predictors_index].size() != 0) {
            all_predictors_[all_predictors_index] -> is_reset = false;
        }
        if (all_predictors_[all_predictors_index] -> is_reset == false) {
            classified_predictor_results.push_back(
                all_predictors_[all_predictors_index] -> step(classified_classifyResults[all_predictors_index], frame, predictor_type)
            );
            // RCLCPP_INFO(node->get_logger(), "%ld updating", all_predictors_index);
        }
    }

    if (priority_armor != ArmorType::AutoSwitch) {
        for (PredictorResult predictor_result : classified_predictor_results) {
            if (predictor_result.armor_type == priority_armor && !predictor_result.reset) {
                return predictor_result;
            }
        }
    }
    
    if (!classified_predictor_results.empty()) {
        auto it = std::min_element(
            classified_predictor_results.begin(), classified_predictor_results.end(),
            [](const PredictorResult& a, const PredictorResult& b) {
                return a.pixel_horizontal_center_distance < b.pixel_horizontal_center_distance;
            }
        );
        if (it != classified_predictor_results.end()) {
            auto nearest_result = *it;
            return nearest_result;
        }
    }
    return PredictorResult();
}