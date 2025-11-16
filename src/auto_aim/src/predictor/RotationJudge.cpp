#include "predictor/RotationJudge.h"

void RotationJudge::addPoint(bool is_seen, const cv::Point3f& armor_position, const cv::Point3f& linear_predict_position) {
    cv::Point3f use_armor_position = armor_position;
    if (!is_seen) {
        if (armor_position_infos.size() > 0) {
            use_armor_position = armor_position_infos.back().armor_position;
        } else {
            use_armor_position = cv::Point3f(0.0, 0.0, 0.0);
        }
    } else {
        seen_frame_count += 1;
    }
    armor_position_infos.push_back(armorPositionInfo(is_seen, armor_position, linear_predict_position));
    if (armor_position_infos.size() > check_frames) {
        seen_frame_count -= static_cast<int>(armor_position_infos.front().is_seen);
        armor_position_infos.pop_front();
    }

    float right_shift = get_right_shift(use_armor_position, linear_predict_position);
    // right_shift_data_smoother -> addPoint(right_shift);
    // float smoothed_right_shift = right_shift_data_smoother -> getExponentialValue();
    
    right_shifts.push_back(right_shift);
    if (right_shifts.size() > check_frames) {
        right_shifts.pop_front();
    }
    periodic_data_fitter -> addPoint(right_shift);
    periodic_data_fitter -> autoFindPeriod();
    // periodic_data_fitter -> setPeriod(24);
    RCLCPP_DEBUG(node->get_logger(), "RotationJudge: right_shift: %f", right_shift);
    // RCLCPP_DEBUG(node->get_logger(), "RotationJudge: smoothed_right_shift: %f", smoothed_right_shift);
#ifdef SHOW_WINDOWS
    oscilloscope_ -> update();
    oscilloscope_ -> addDataPoint(periodic_data_fitter -> smooth(0) / 300.0);
    oscilloscope_ -> putText("period rs:"+std::to_string(periodic_data_fitter->getPeriod()), cv::Point2f(240, 20), cv::Scalar(0, 255, 0), 0.7);
    oscilloscope_ -> show();
#endif
}

void RotationJudge::clearHistory() {
    armor_position_infos.clear();
    periodic_data_fitter -> clearHistory();
    seen_frame_count = 0;
    right_shifts.clear();
    // right_shift_data_smoother -> clearHistory();
}

float RotationJudge::get_right_shift(cv::Point3f armor_position, cv::Point3f linear_predict_position) {
    std::vector<float> cam_position = rest_frame_ -> getCamPosition();
    std::vector<double> cam_to_linear_predict_vector = {linear_predict_position.x - cam_position[0], linear_predict_position.y - cam_position[1]};
    double cam_to_linear_predict_vector_len = std::sqrt(cam_to_linear_predict_vector[0] * cam_to_linear_predict_vector[0] + cam_to_linear_predict_vector[1] * cam_to_linear_predict_vector[1]);
    std::vector<double> right_unit_v = {cam_to_linear_predict_vector[1] / cam_to_linear_predict_vector_len, - cam_to_linear_predict_vector[0] / cam_to_linear_predict_vector_len};
    std::vector<double> cam_armor_v = {armor_position.x - cam_position[0], armor_position.y - cam_position[1]};
    float right_shift = right_unit_v[0] * cam_armor_v[0] + right_unit_v[1] * cam_armor_v[1];
    return right_shift;
}

bool RotationJudge::is_rotation(float p3d_period, float rmm_period) {
    if (armor_position_infos.size() < min_rotation_frames || seen_frame_count < min_seen_frames) {
        RCLCPP_DEBUG(node->get_logger(), "RotationJudge: false 1");
        return false;
    }
    float right_shift_period = periodic_data_fitter -> getPeriod();
    float max_period = std::max({p3d_period, rmm_period, right_shift_period});
    float min_period = std::min({p3d_period, rmm_period, right_shift_period});
    RCLCPP_DEBUG(node->get_logger(), "RotationJudge: right_shift_period: %f", right_shift_period);
    if (max_period - min_period > max_period_divergence) {
        RCLCPP_DEBUG(node->get_logger(), "RotationJudge: false 2");
        return false;
    }
    float right_shifts_variance = variance(right_shifts);
    RCLCPP_DEBUG(node->get_logger(), "RotationJudge: variance: %f", right_shifts_variance);
    if (right_shifts_variance < min_right_shift_variance) {
        RCLCPP_DEBUG(node->get_logger(), "RotationJudge: false 3");
        return false;
    }
    float right_shifts_fit_mse = periodic_data_fitter -> getFitMse();
    float R_squared = 1.0 - right_shifts_fit_mse / right_shifts_variance;
    RCLCPP_DEBUG(node->get_logger(), "RotationJudge: R_squared: %f", R_squared);
    if (R_squared < min_right_shift_fit_R_squared) {
        RCLCPP_DEBUG(node->get_logger(), "RotationJudge: false 4");
        return false;
    }
    RCLCPP_DEBUG(node->get_logger(), "RotationJudge: right_shift_period: %f", right_shift_period);
    if (right_shift_period > max_period) {
        RCLCPP_DEBUG(node->get_logger(), "RotationJudge: false 5");
        return false;
    }
    RCLCPP_DEBUG(node->get_logger(), "RotationJudge: true");
    return true;
}