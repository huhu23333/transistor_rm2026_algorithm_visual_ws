#include "predictor/PredictorMain.h"

void PredictorMain::update_serial_info(float bullet_velocity, float last_pitch_rad_delayed, float last_yaw_rad_delayed, float total_yaw_rad_delayed) {
    last_pitch_rad_delayed_ = last_pitch_rad_delayed;
    last_yaw_rad_delayed_ = last_yaw_rad_delayed;
    total_yaw_rad_delayed_ = total_yaw_rad_delayed;
    for (std::shared_ptr<AllPredictor>& all_predictor : all_predictors_) {
        if (all_predictor) {
            all_predictor -> update_serial_info(bullet_velocity, last_pitch_rad_delayed, last_yaw_rad_delayed, total_yaw_rad_delayed);
        }
    }
}

PredictorResult PredictorMain::step(std::vector<ArmorResult>& classifyResults, cv::Mat& frame, PredictorType::PredictorType predictor_type, ArmorType::ArmorType priority_armor, bool auto_aim_switch) {

    PredictorResult chosen_result;

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

    if (priority_armor == ArmorType::Middle) {
        if (!classified_predictor_results.empty()) {
            auto it = std::min_element(
                classified_predictor_results.begin(), classified_predictor_results.end(),
                [](const PredictorResult& a, const PredictorResult& b) {
                    return a.pixel_horizontal_center_distance < b.pixel_horizontal_center_distance;
                }
            );
            if (it != classified_predictor_results.end()) {
                auto middle_result = *it;
                chosen_result = middle_result;
            }
        }
    } else if (priority_armor == ArmorType::Nearest) {
        if (!classified_predictor_results.empty()) {
            auto it = std::min_element(
                classified_predictor_results.begin(), classified_predictor_results.end(),
                [](const PredictorResult& a, const PredictorResult& b) {
                    return a.latest_armor_distance < b.latest_armor_distance;
                }
            );
            if (it != classified_predictor_results.end()) {
                auto nearest_result = *it;
                chosen_result = nearest_result;
            }
        }
    } else {
        for (PredictorResult predictor_result : classified_predictor_results) {
            if (predictor_result.armor_type == priority_armor && !predictor_result.reset) {
                chosen_result = predictor_result;
            }
        }
    }

    if (auto_aim_switch) { // 仅在电控自瞄开关打开时进行积分
        pitch_integration += chosen_result.command_delta_pitch * 0.02;
        yaw_integration += chosen_result.command_delta_yaw * 0.02;
        if (pitch_integration > 60.0 * M_PI / 180.0) {
            pitch_integration = 60.0 * M_PI / 180.0;
        }
        if (pitch_integration < -60.0 * M_PI / 180.0) {
            pitch_integration = -60.0 * M_PI / 180.0;
        }

        if (yaw_integration > 60.0 * M_PI / 180.0) {
            yaw_integration = 60.0 * M_PI / 180.0;
        }
        if (yaw_integration < -60.0 * M_PI / 180.0) {
            yaw_integration = -60.0 * M_PI / 180.0;
        }
    }
    chosen_result.command_pitch = last_pitch_rad_delayed_ + chosen_result.command_delta_pitch * 1.0 + pitch_integration; // PI控制
    chosen_result.command_yaw = last_yaw_rad_delayed_ + chosen_result.command_delta_yaw * 1.0 + yaw_integration; // 缓解yaw轴输入数据掉线问题（并不能()）
    return chosen_result;
}
