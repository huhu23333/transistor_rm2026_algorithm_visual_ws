#include "predictor/PredictorSwitcher.h"

namespace PredictorType {
    std::vector<std::string> PredictorTypeStrings = {
        "None",
        "EKF",
        "FP",
        "RMM",
        "AutoSwitch(should not be used)"
    };
}

void PredictorSwitcher::clearHistory() {
    predictors_results.clear();
    real_points.clear();
    P3D_periods.clear();
    RMM_periods.clear();
    rotation_judge -> clearHistory();
}


PredictorType::PredictorType PredictorSwitcher::step(bool is_seen, cv::Point3f real_point, 
    cv::Point3f None_result, cv::Point3f EKF_result, cv::Point3f P3D_result, cv::Point3f RMM_result, 
    float P3D_period, float RMM_period, cv::Point3f linear_predict_position) {
    //return PredictorType::RotationMotionModel;
    return PredictorType::EKF;

    predictors_results.push_back(PredictorsResult(None_result, EKF_result, P3D_result, RMM_result));
    if (predictors_results.size() > check_frames * 2) {
        predictors_results.pop_front();
    }
    real_points.push_back(RealPoint(is_seen, real_point));
    if (real_points.size() > check_frames) {
        real_points.pop_front();
    }
    P3D_periods.push_back(P3D_period);
    RMM_periods.push_back(RMM_period);
    if (P3D_periods.size() > period_check_frames) {
        P3D_periods.pop_front();
        RMM_periods.pop_front();
    }

    rotation_judge -> addPoint(is_seen, real_point, linear_predict_position);
    bool is_rotation = rotation_judge -> is_rotation(P3D_period, RMM_period);
    
    if (static_cast<int>(predictors_results.size()) - check_frames < min_check_frames) {
        return PredictorType::None;
    }
    int seen_real_point_count = 0;
    for (int seen_real_point_index = 0; seen_real_point_index < real_points.size(); seen_real_point_index++) {
        if (real_points[seen_real_point_index].is_seen)
        {
            seen_real_point_count++;
        }
    }
    if (seen_real_point_count < min_check_frames) {
        return PredictorType::None;
    }

    std::vector<cv::Point3f> seen_real, seen_None, seen_EKF, seen_P3D, seen_RMM;
    for (int index_from_now_to_back = 0; index_from_now_to_back < predictors_results.size() - check_frames; index_from_now_to_back++) {
        int real_index = real_points.size() - 1 - index_from_now_to_back;
        int pred_index = predictors_results.size() - check_frames - 1 - index_from_now_to_back;
        if (real_points[real_index].is_seen) {
            seen_real.push_back(real_points[real_index].pos);
            seen_None.push_back(predictors_results[pred_index].None_result);
            seen_EKF.push_back(predictors_results[pred_index].EKF_result);
            seen_P3D.push_back(predictors_results[pred_index].P3D_result);
            seen_RMM.push_back(predictors_results[pred_index].RMM_result);
        }
    }
    float real_variance = variancePoint3f(seen_real);
    float None_mse = meanSquaredErrorPoint3f(seen_None, seen_real);
    float EKF_mse = meanSquaredErrorPoint3f(seen_None, seen_EKF);
    float P3D_mse = meanSquaredErrorPoint3f(seen_None, seen_P3D);
    float RMM_mse = meanSquaredErrorPoint3f(seen_None, seen_RMM);
    float EKF_variance = variancePoint3f(seen_EKF);

    RCLCPP_INFO(node->get_logger(), "real_variance: %f, None_mse: %f, EKF_mse: %f, P3D_mse: %f, RMM_mse: %f, EKF_variance: %f, ", 
        real_variance, None_mse, EKF_mse, P3D_mse, RMM_mse, EKF_variance);

    if ((real_variance < 2500.0 && None_mse < 2500.0) || 
        (None_mse < EKF_mse && None_mse < P3D_mse && None_mse < RMM_mse)) {
        RCLCPP_INFO(node->get_logger(), "None 1");
        return PredictorType::None;
    }

    float P3D_period_variance = variance(P3D_periods);
    float RMM_period_variance = variance(RMM_periods);
    float mean_period = (std::accumulate(P3D_periods.begin(), P3D_periods.end(), 0.0) + 
                         std::accumulate(RMM_periods.begin(), RMM_periods.end(), 0.0)) / 
                         static_cast<float>(P3D_periods.size() + RMM_periods.size());

    is_rotation = true;
    if (is_rotation &&
        ((P3D_period_variance < 10.0) || (RMM_period_variance < 10.0)) && 
        (mean_period > 2) && (mean_period < 40.0)
        ) {
        if (mean_period < 10.0 || (RMM_mse > 200000.0)) {
            return PredictorType::FirePredictor;
        } else {
            return PredictorType::RotationMotionModel;
        }
    }

    if (EKF_variance < real_variance && EKF_mse < None_mse) {
        return PredictorType::EKF;
    }

    RCLCPP_INFO(node->get_logger(), "None 2");
    return PredictorType::None;
}