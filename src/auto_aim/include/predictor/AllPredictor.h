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
#include "predictor/PositionPredictor3D.h"
#include "predictor/PeriodicDataPredictor.h"
#include "utils/SimpleDataFilter.h"
#include "predictor/RotationMotionModel.h"
#include "predictor/PredictorSwitcher.h"
#include "macro/AutoAimMacro.h"
// EKF
#include "EKF/Tracker.h"
struct PredictorResult {
    bool reset = true;
    float command_pitch = 0.0;
    float command_yaw = 0.0;
    bool fire_flag = false;
    PredictorType::PredictorType predictor_type = PredictorType::None;
    ArmorType::ArmorType armor_type = ArmorType::Hero;
    float pixel_horizontal_center_distance = 1e10;
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
        
        yaw_rad_to_x_pixel_ratio = (*config_file_ptr)["yaw_rad_to_x_pixel_ratio"].as<float>(); 
        pitch_rad_to_y_pixel_ratio = (*config_file_ptr)["pitch_rad_to_y_pixel_ratio"].as<float>(); 

        RESET_DISTANCE_THRESHOLD = (*config_file_ptr)["RESET_DISTANCE_THRESHOLD"].as<float>(); 
        MAX_LOST_TIME = (*config_file_ptr)["MAX_LOST_TIME"].as<float>(); 

        reset_com_time = (*config_file_ptr)["reset_com_time"].as<float>(); 

        predictor3d_fit_step = (*config_file_ptr)["predictor3d_fit_step"].as<int>();
        predictor3d_predict_step = (*config_file_ptr)["predictor3d_predict_step"].as<int>();
        predictor3d_fourier_fit_order = (*config_file_ptr)["predictor3d_fourier_fit_order"].as<int>();

        fire_distance = (*config_file_ptr)["fire_distance"].as<float>();
        fire_data_predictor_fit_step = (*config_file_ptr)["fire_data_predictor_fit_step"].as<int>();

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


        // std::string world_log_path = "/home/dyj/rm2026/transistor_rm2026_algorithm_visual_ws/录制数据/world_coords.csv";
        // world_log_.open(world_log_path, std::ios::out | std::ios::trunc);
        // if (world_log_.is_open()) {
        //     world_log_ << "timestamp_ms,x_mm,y_mm,z_mm,yaw_rad\n";
        // }
        
        // /// ========== 新的 EKF 和 Tracker 初始化 (9D模型修改) ==========

        // // ========== EKF 和 Tracker 初始化 (6D模型修改) ==========
        // double dt = 1.0 / std::max(1.0f, frame_rate_);
        
        // // 1. 从配置文件加载新的EKF参数
        // EKFParams ekf_params;
        // const auto& ekf_config = (*config_file_ptr)["ekf_params"];
        
        // // 过程噪声，对应加速度的标准差
        // ekf_params.s2qx = ekf_config["sigma2_q_x"].as<double>();
        // ekf_params.s2qy = ekf_config["sigma2_q_y"].as<double>();
        // ekf_params.s2qz = ekf_config["sigma2_q_z"].as<double>();
        
        // // 测量噪声，对应位置的标准差
        // ekf_params.r_x = ekf_config["r_x_coeff"].as<double>();
        // ekf_params.r_y = ekf_config["r_y_coeff"].as<double>();
        // ekf_params.r_z = ekf_config["r_z_coeff"].as<double>();

        // ekf_params.p0 = ekf_config["p0_init_val"].as<double>();

        // // 2. 创建Tracker，传入新参数
        // EKF_tracker_ = std::make_unique<Tracker>(dt, ekf_params);
        // RCLCPP_INFO(this->get_logger(), "New 6D EKF Tracker initialized with dt=%.4f and params from config.", dt);
        // // ========== 初始化结束 ==========
        
        predictor3d = std::make_shared<PositionPredictor3D>(predictor3d_fit_step);
        predictor3dArmorPredictions.push_back(cv::Point3f(0,0,0));
        predictor3dCenterPredictions.push_back(cv::Point3f(0,0,0));


        oscilloscope_fire_ = std::make_shared<Oscilloscope>(640, 120, "Fire Data Oscilloscope");
        oscilloscope_fire_ -> setScale(1.0);
        oscilloscope_fire_ -> setOffset(-0.5);

        oscilloscope_common_ = std::make_shared<Oscilloscope>(640, 480, "Common Debug Oscilloscope", 30);
        oscilloscope_common_ -> setScale(2.0);
        oscilloscope_common_ -> setOffset(-1.0);

        fire_data_predictor_ = std::make_shared<PeriodicDataPredictor>(fire_data_predictor_fit_step, 5);
        fire_data_predictor_ -> setPeriod(1);
        pred_fire_data_filter_ = std::make_shared<SimpleDataFilter>(1);
        pred_fire_data_filter_ -> setExponentialAlpha((*config_file_ptr)["pred_fire_data_smooth_factor"].as<float>());
        pred_fire_data_filter_ -> addPoint(0.0);

        armor_distance_filter_ = std::make_shared<SimpleDataFilter>(1);
        armor_distance_filter_ -> setExponentialAlpha((*config_file_ptr)["armor_distance_smooth_factor"].as<float>());

        predictor_switcher_ = std::make_shared<PredictorSwitcher>(config_file_ptr, node, predictor_switcher_check_frames_, rest_frame_);

        total_yaw_rad_delayed_filter_ = std::make_shared<SimpleDataFilter>(1);
        total_yaw_rad_delayed_filter_ -> setExponentialAlpha((*config_file_ptr)["total_yaw_rad_delayed_smooth_factor"].as<float>());
    }

    PredictorResult step(std::vector<ArmorResult>& classifyResults, cv::Mat& frame, PredictorType::PredictorType control_predictor_type);
    void update_serial_info(float bullet_velocity, float last_pitch_rad_delayed, float last_yaw_rad_delayed, float total_yaw_rad_delayed);

    bool is_reset = false;

private:
    PredictorType::PredictorType using_predictor_type = PredictorType::None;
    ArmorType::ArmorType armor_class;

    std::shared_ptr<YAML::Node> config_file_ptr; 
    rclcpp::Node* node;
    std::chrono::time_point<std::chrono::steady_clock> node_start_time;
    std::shared_ptr<ArmorSolver> armor_solver_;
    std::shared_ptr<BallisticSolver> ballistic_solver_;
    std::shared_ptr<RestFrame> rest_frame_;
    std::shared_ptr<FrameRateCounter> fps_counter;

    int current_target_id_ = -1;      // 当前跟踪目标ID - EKF

    float RESET_DISTANCE_THRESHOLD; // 单位：mm
    float MAX_LOST_TIME;              // 单位：秒

    float last_total_delay_ = 0.0;

    std::shared_ptr<Oscilloscope> oscilloscope_fire_;
    std::shared_ptr<PeriodicDataPredictor> fire_data_predictor_;
    std::shared_ptr<SimpleDataFilter> pred_fire_data_filter_;
    
    std::shared_ptr<Oscilloscope> oscilloscope_common_;
    std::shared_ptr<SimpleDataFilter> armor_distance_filter_;

    std::shared_ptr<RotationMotionModel> rotation_motion_model_;

    float bullet_velocity_;
    float last_pitch_rad_delayed_ = 0;
    float last_yaw_rad_delayed_ = 0;
    float total_yaw_rad_delayed_ = 0;
    std::shared_ptr<SimpleDataFilter> total_yaw_rad_delayed_filter_;

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

    float pitch_integration = 0.0;
    float yaw_integration = 0.0;
    float yaw_rad_to_x_pixel_ratio;
    float pitch_rad_to_y_pixel_ratio;
    float reset_com_time;
    std::chrono::steady_clock::time_point last_com_time;
    std::shared_ptr<PositionPredictor3D> predictor3d;
    int predictor3dPrediction_nowIndex = 0;
    std::vector<cv::Point3f> predictor3dArmorPredictions;
    std::vector<cv::Point3f> predictor3dCenterPredictions;
    int predictor3d_fit_step;
    int predictor3d_predict_step;
    int predictor3d_fourier_fit_order;
    float fire_distance;
    cv::Point2f last_aim_yaw_pitch_;
    cv::Point2f last_aim_yaw_pitch_pixel_;
    float last_command_pitch_;
    float last_command_yaw_;
    int fire_data_predictor_fit_step;

    std::shared_ptr<PredictorSwitcher> predictor_switcher_;
    int predictor_switcher_check_frames_ = 30;

    cv::Point3f last_rest_frame_pos = {0.0, 0.0, 0.0};

    float last_pixel_horizontal_center_distance = 1e10;

    bool has_valid_ballistic = false;
};