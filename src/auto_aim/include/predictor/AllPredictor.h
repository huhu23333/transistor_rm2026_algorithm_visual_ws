#pragma once
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <memory>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>

#include <3d_processing/BallisticSolver.h>
#include "3d_processing/ArmorSolver.h"
#include "2d_armor_detector/Armor.h"
#include "utils/FrameRateCounter.h"
#include "3d_processing/RestFrame.h"
#include "visualizer/DataVisualizer.h"
#include "utils/SimpleDataFilter.h"
#include "predictor/RotationMotionModel.h"
#include "predictor/PredictorSwitcher.h"
#include "EKF/Tracker.h"
#include "macro/AutoAimMacro.h"

struct PredictorResult {
    bool reset = true;
    float command_pitch = 0.0;
    float command_yaw = 0.0;
    float command_delta_pitch = 0.0;
    float command_delta_yaw = 0.0;
    bool fire_flag = false;
    PredictorType::PredictorType predictor_type = PredictorType::None;
    ArmorType::ArmorType armor_type = ArmorType::Hero;
    float pixel_horizontal_center_distance = 1e10;
    float latest_armor_distance = 1e10;
};

struct RMM_fire_result_t {
    bool aim_center;
    bool fire;
};

class AllPredictor {
public:
    AllPredictor(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node, 
        std::chrono::time_point<std::chrono::steady_clock> node_start_time, 
        std::shared_ptr<ArmorSolver> armor_solver_,
        std::shared_ptr<BallisticSolver> ballistic_solver_,
        std::shared_ptr<RestFrame> rest_frame_, std::shared_ptr<FrameRateCounter> fps_counter,
        ArmorType::ArmorType armor_class
    ) : config_file_ptr(config_file_ptr), node(node), node_start_time(node_start_time), 
    armor_solver_(armor_solver_), ballistic_solver_(ballistic_solver_),
    rest_frame_(rest_frame_), fps_counter(fps_counter), armor_class(armor_class) {
        // 初始化参数
        bullet_velocity_ = (*config_file_ptr)["bullet_velocity_"].as<float>();
        
        // yaw_rad_to_x_pixel_ratio = (*config_file_ptr)["yaw_rad_to_x_pixel_ratio"].as<float>(); 
        // pitch_rad_to_y_pixel_ratio = (*config_file_ptr)["pitch_rad_to_y_pixel_ratio"].as<float>(); 
        const YAML::Node& camera_matrix_Node = (*config_file_ptr)["camera_matrix"];
        yaw_rad_to_x_pixel_ratio = camera_matrix_Node[0][0].as<float>(); 
        pitch_rad_to_y_pixel_ratio = camera_matrix_Node[1][1].as<float>(); 

        reset_predictor_time = (*config_file_ptr)["reset_predictor_time"].as<float>(); 

        last_com_time = std::chrono::steady_clock::now();


        // ===== 通用装甲 EKF 几何参数（按装甲类型分） =====
        is_outpost_ = (armor_class == ArmorType::Outpost);

        // ===== 通用装甲 EKF 噪声参数 =====
        if ((*config_file_ptr)["ekf_params"]) {
            auto ep = (*config_file_ptr)["ekf_params"];
            if (ep["s2qx"])   ekf_params_.s2qx   = ep["s2qx"].as<double>();
            if (ep["s2qy"])   ekf_params_.s2qy   = ep["s2qy"].as<double>();
            if (ep["s2qz"])   ekf_params_.s2qz   = ep["s2qz"].as<double>();
            if (ep["s2qyaw"]) ekf_params_.s2qyaw = ep["s2qyaw"].as<double>();
            if (ep["s2qr"])   ekf_params_.s2qr   = ep["s2qr"].as<double>();
            if (ep["s2qdz"])  ekf_params_.s2qdz  = ep["s2qdz"].as<double>();

            if (ep["r_x"])    ekf_params_.r_x    = ep["r_x"].as<double>();
            if (ep["r_y"])    ekf_params_.r_y    = ep["r_y"].as<double>();
            if (ep["r_z"])    ekf_params_.r_z    = ep["r_z"].as<double>();
            if (ep["r_yaw"])  ekf_params_.r_yaw  = ep["r_yaw"].as<double>();

            if (ep["p0"])     ekf_params_.p0     = ep["p0"].as<double>();
        }

        if ((*config_file_ptr)["ekf_extra_delay"]) {
            ekf_extra_delay_ = (*config_file_ptr)["ekf_extra_delay"].as<double>();
        }

        
        // ====== 读取匹配阈值（四装甲跳变判据）======
        // 优先从 tracker 节点读取；如果没有，则尝试顶层键；最后落回默认值
        if ((*config_file_ptr)["tracker"]) {
            auto tr = (*config_file_ptr)["tracker"];
            if (tr["max_match_distance_mm"]) {
                ekf_max_match_distance_mm_ = tr["max_match_distance_mm"].as<double>();
            }
            if (tr["max_match_yaw_diff_rad"]) {
                ekf_max_match_yaw_diff_rad_ = tr["max_match_yaw_diff_rad"].as<double>();
            }
        } else {
            if ((*config_file_ptr)["ekf_max_match_distance_mm"]) {
                ekf_max_match_distance_mm_ = (*config_file_ptr)["ekf_max_match_distance_mm"].as<double>();
            }
            if ((*config_file_ptr)["ekf_max_match_yaw_diff_rad"]) {
                ekf_max_match_yaw_diff_rad_ = (*config_file_ptr)["ekf_max_match_yaw_diff_rad"].as<double>();
            }
        }


        // 初始化通用 EKF Tracker，dt 先随便给一个，后面每帧会更新
        armor_tracker_ = std::make_shared<armor_ekf::Tracker>(1.0 / 30.0, ekf_params_);


        // ========== 初始化结束 ==========


        oscilloscope_common_ = std::make_shared<Oscilloscope>(640, 480, "Common Debug Oscilloscope", 2);
        oscilloscope_common_ -> setScale(2.0);
        oscilloscope_common_ -> setOffset(-1.0);

        armor_distance_filter_ = std::make_shared<SimpleDataFilter>(1);
        armor_distance_filter_ -> setExponentialAlpha((*config_file_ptr)["armor_distance_smooth_factor"].as<float>());

        predictor_switcher_ = std::make_shared<PredictorSwitcher>(config_file_ptr, node);

        RMM_fire_control_data.target_change_ceasefire_ms = (*config_file_ptr)["target_change_ceasefire_ms"].as<int>();
        RMM_fire_control_data.aim_center_vyaw_threshold = (*config_file_ptr)["aim_center_vyaw_threshold"].as<float>();
        RMM_fire_control_data.aim_center_yaw_bias_expand = (*config_file_ptr)["aim_center_yaw_bias_expand"].as<float>();

        pre_predict_time = (*config_file_ptr)["pre_predict_time"].as<float>();
    }

    PredictorResult step(std::vector<ArmorResult>& classifyResults, cv::Mat& frame, PredictorType::PredictorType control_predictor_type);
    void update_serial_info(float bullet_velocity, float last_pitch_rad_delayed, float last_yaw_rad_delayed, float total_yaw_rad_delayed);

    bool is_reset = false;

    std::chrono::steady_clock::time_point latest_predicting_start_time;
private:
    PredictorType::PredictorType using_predictor_type = PredictorType::None;
    ArmorType::ArmorType armor_class;
    bool armor_is_large;

    std::shared_ptr<YAML::Node> config_file_ptr; 
    rclcpp::Node* node;
    std::chrono::time_point<std::chrono::steady_clock> node_start_time;
    std::shared_ptr<ArmorSolver> armor_solver_;
    std::shared_ptr<BallisticSolver> ballistic_solver_;
    std::shared_ptr<RestFrame> rest_frame_;
    std::shared_ptr<FrameRateCounter> fps_counter;

    float last_total_delay_ = 0.0;

    std::shared_ptr<Oscilloscope> oscilloscope_common_;
    std::shared_ptr<SimpleDataFilter> armor_distance_filter_;

    std::shared_ptr<RotationMotionModel> rotation_motion_model_;

    float bullet_velocity_;
    float last_pitch_rad_delayed_ = 0;
    float last_yaw_rad_delayed_ = 0;
    float total_yaw_rad_delayed_ = 0;

    float yaw_rad_to_x_pixel_ratio;
    float pitch_rad_to_y_pixel_ratio;
    float reset_predictor_time;
    std::chrono::steady_clock::time_point last_com_time;
    cv::Point2f last_aim_yaw_pitch_;
    cv::Point2f last_aim_yaw_pitch_pixel_;

    std::shared_ptr<PredictorSwitcher> predictor_switcher_;

    cv::Point3f last_rest_frame_pos = {0.0, 0.0, 0.0};

    float last_pixel_horizontal_center_distance = 1e10;

    bool has_valid_ballistic = false;
    
    float init_r = 250.0;

    struct RMM_fire_control_data_t {
        int target_change_ceasefire_ms;
        float aim_center_vyaw_threshold;
        float aim_center_yaw_bias_expand;
        bool new_target = true;

        float last_target_yaw;
        std::chrono::steady_clock::time_point last_target_yaw_jump_time;
    } RMM_fire_control_data;

    RMM_fire_result_t RMM_fire_control(SimpleArmor chosen_armor, RotationMotionState RMM_state, float yaw_bias, bool is_large_armor);

    float latest_armor_distance = 1e10;

    float pre_predict_time;

    // 通用装甲 EKF（基于 FYT 10 维状态）
    armor_ekf::EKFParams ekf_params_;
    std::shared_ptr<armor_ekf::Tracker> armor_tracker_;
    double ekf_last_time_   = 0.20;   // 上一帧时间戳（s）
    double ekf_extra_delay_ = 0.0;  // 额外提前量（策略冗余）, s

    // 当前这个 AllPredictor 对应目标类型的几何信息
    int armor_num_         = 4;        // 一圈有几块装甲：普通车 4，前哨站 3
    double ekf_init_r_mm_  = 265.0;  // 中心到装甲板水平半径
    double ekf_init_dz_mm_ = 20.0;    // 中心到装甲板高度偏置
    bool is_outpost_       = false;

    // 四装甲匹配与跳变的阈值（可由 YAML 配置）
    double ekf_max_match_distance_mm_ = 350.0;
    double ekf_max_match_yaw_diff_rad_ = 0.40;
    
    int current_target_id_ = -1;      // 当前跟踪目标ID - EKF

};
