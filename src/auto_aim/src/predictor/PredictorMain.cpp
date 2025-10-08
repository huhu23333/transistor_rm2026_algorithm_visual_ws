#include "predictor/PredictorMain.h"

void PredictorMain::update_serial_info(float bullet_velocity, float last_pitch_rad_delayed, float last_yaw_rad_delayed, float total_yaw_rad_delayed) {
    for (std::shared_ptr<AllPredictor>& all_predictor : all_predictors_) {
        if (all_predictor) {
            all_predictor -> update_serial_info(bullet_velocity, last_pitch_rad_delayed, last_yaw_rad_delayed, total_yaw_rad_delayed);
        }
    }
}

PredictorResult PredictorMain::step(std::vector<ArmorResult>& classifyResults, cv::Mat& frame) {
    std::vector<std::vector<ArmorResult>> classified_classifyResults(classify_classes);
    for (ArmorResult& classify_result : classifyResults) {
        classified_classifyResults[classify_result.number].push_back(classify_result);
    }
    // todo 前哨站/基地特殊处理
    std::vector<PredictorResult> classified_predictor_results;
    for (size_t all_predictors_index = 0; all_predictors_index < classify_classes; all_predictors_index++) {
        if (classified_classifyResults[all_predictors_index].size() != 0) {
            /* if (!all_predictors_[all_predictors_index]) {
                all_predictors_[all_predictors_index] = std::make_shared<AllPredictor>(
                                                        config_file_ptr, node, node_start_time, armor_solver_,
                                                        ballistic_solver_, rest_frame_, fps_counter);
            } */
            all_predictors_[all_predictors_index] -> is_reset = false;
        }
        if (all_predictors_[all_predictors_index] -> is_reset == false) {
            classified_predictor_results.push_back(
                all_predictors_[all_predictors_index] -> step(classified_classifyResults[all_predictors_index], frame)
            );
            // RCLCPP_INFO(node->get_logger(), "%ld updating", all_predictors_index);
        }
        // RCLCPP_INFO(node->get_logger(), "%ld is_reset %d", all_predictors_index, predictors_is_reset[all_predictors_index]);
    }

    // todo 结果选择
    for (PredictorResult predictor_result : classified_predictor_results) {
        if (predictor_result.reset == false) {
            return predictor_result;
        }
    }
    return PredictorResult();
}