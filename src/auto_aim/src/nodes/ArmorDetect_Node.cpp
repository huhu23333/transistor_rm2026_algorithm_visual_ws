// ArmorDetect_Node.cpp
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include "camera/Camera.h"
#include "armor_detector/LightBarDetector.h"
#include "armor_detector/ArmorDetector.h"
#include "armor_detector/ArmorClassifier.h"
#include "armor_detector/ArmorSolver.h"
//#include "armor_detector/ArmorAngleKalman.h"

#include "EKF/Tracker.h"
#include <angles/angles.h>

//#include "auto_aim/msg/serial_data.hpp"
//#include "auto_aim/msg/gimbal_command.hpp"
#include <chrono>
#include <string>
#include <thread>
#include <armor_detector/BallisticSolver.h>
#include <yaml-cpp/yaml.h>
#include "utils/FrameRateCounter.h"
#include "utils/UnwarpUtils.h"
#include "test_codes/VideoInput.h"
#include "test_codes/ImagesInput.h"
#include <iostream>
#include <sstream>
#include <filesystem>
#include <unistd.h>
#include <limits.h>
#include <queue>
#include "utils/Com.h"
#include <csignal>
#include "test_codes/PredictionTrans.h"
#include "utils/RestFrame.h"
#include "utils/PositionPredictor3D.h"
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>
#include "test_codes/DataVisualizer.h"
#include "utils/PeriodicDataFitter.h"
#include "utils/SimpleDataFilter.h"

namespace fs = std::filesystem;


#define USE_VIDEO // 定义后使用视频而不是摄像头作为输入
//#define USE_IMAGES // 定义后使用图片而不是摄像头作为输入
//#define SAVE_IMG_FREQ 30 // 定义后将每n帧保存一次相机图片
#define USE_PREDICTOR3D // 定义后使用3D位置预测器而不是EKF
//#define DEBUG_CODE // 定义后将在初始化结束后、装甲板识别代码前运行debug代码

// 全局变量定义
cv::Mat g_image;
pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
bool g_bExit = false;
bool image_used = true;

class ArmorDetectNode : public rclcpp::Node {
public:
    ArmorDetectNode() : Node("armor_detect_node") {

        // 1. 获取可执行文件路径
        char exec_path[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
        if (len == -1) {
            perror("readlink");
            return;
        }
        exec_path[len] = '\0';
        RCLCPP_INFO(this->get_logger(), "info from C++ | Path: %s\n", exec_path);
        // 2. 转换为文件系统路径对象
        fs::path full_path(exec_path);
        std::string full_path_str = full_path.string();  // 转换为字符串便于查找
        // 3. 查找工作空间目录名
        const std::string ws_dir_name = "transistor_rm2026_algorithm_visual_ws";
        size_t pos = full_path_str.find(ws_dir_name);
        if (pos == std::string::npos) {
            std::cerr << "Error: Workspace directory not found in path" << std::endl;
            return;
        }
        // 4. 截取到工作空间目录结尾
        fs::path ws_dir_path = full_path_str.substr(0, pos + ws_dir_name.length());
        // 5. 拼接模型路径
        const std::string config_file_relatvie_path = "src/shared_files/config.yaml";
        fs::path config_file_path = ws_dir_path / config_file_relatvie_path;  // 使用文件系统的路径拼接

        // 加载配置文件
        config_file_ptr = std::make_shared<YAML::Node>(YAML::LoadFile(config_file_path));



        // 初始化串口通信器
        serial_communication_ = std::make_shared<SerialCommunicationClass>(this, std::bind(&ArmorDetectNode::serialDataCallback, this, std::placeholders::_1));

        // 初始化参数
        bullet_velocity_ = (*config_file_ptr)["bullet_velocity_"].as<float>();
        current_pitch_ = (*config_file_ptr)["current_pitch_"].as<float>();
        current_yaw_ = (*config_file_ptr)["current_yaw_"].as<float>();

        delta_x_ = (*config_file_ptr)["delta_x_"].as<float>();
        delta_y_ = (*config_file_ptr)["delta_y_"].as<float>();
        delta_z_ = (*config_file_ptr)["delta_z_"].as<float>();

        RESET_DISTANCE_THRESHOLD = (*config_file_ptr)["RESET_DISTANCE_THRESHOLD"].as<float>(); 
        MAX_LOST_TIME = (*config_file_ptr)["MAX_LOST_TIME"].as<float>(); 

        has_valid_target_ = false;
        enemy_color_ = (*config_file_ptr)["enemy_color"].as<std::string>();

        yaw_rad_to_x_pixel_ratio = (*config_file_ptr)["yaw_rad_to_x_pixel_ratio"].as<float>(); 
        pitch_rad_to_y_pixel_ratio = (*config_file_ptr)["pitch_rad_to_y_pixel_ratio"].as<float>(); 
        
        params_.min_light_height = (*config_file_ptr)["min_light_height"].as<int>();
        params_.light_slope_offset = (*config_file_ptr)["light_slope_offset"].as<int>();
        params_.light_min_area = (*config_file_ptr)["light_min_area"].as<int>();
        params_.max_light_wh_ratio = (*config_file_ptr)["max_light_wh_ratio"].as<float>();
        params_.min_light_wh_ratio = (*config_file_ptr)["min_light_wh_ratio"].as<float>();
        params_.light_max_tilt_angle = (*config_file_ptr)["light_max_tilt_angle"].as<float>();
        params_.min_light_delta_x = (*config_file_ptr)["min_light_delta_x"].as<int>();
        params_.min_light_dx_ratio = (*config_file_ptr)["min_light_dx_ratio"].as<float>();
        params_.max_light_dy_ratio = (*config_file_ptr)["max_light_dy_ratio"].as<float>();
        params_.max_light_delta_angle = (*config_file_ptr)["max_light_delta_angle"].as<float>();
        params_.near_face_v = (*config_file_ptr)["near_face_v"].as<int>();
        params_.max_lr_rate = (*config_file_ptr)["max_lr_rate"].as<float>();
        params_.max_wh_ratio = (*config_file_ptr)["max_wh_ratio"].as<float>();
        params_.min_wh_ratio = (*config_file_ptr)["min_wh_ratio"].as<float>();
        params_.small_armor_wh_threshold = (*config_file_ptr)["small_armor_wh_threshold"].as<float>();
        params_.bin_cls_thres = (*config_file_ptr)["bin_cls_thres"].as<int>();
        params_.target_max_angle = (*config_file_ptr)["target_max_angle"].as<int>();
        params_.goodToTotalRatio = (*config_file_ptr)["goodToTotalRatio"].as<float>();
        params_.matchDistThre = (*config_file_ptr)["matchDistThre"].as<int>();
        params_.wh_ratio_threshold = (*config_file_ptr)["wh_ratio_threshold"].as<float>();
        params_.wh_ratio_max = (*config_file_ptr)["wh_ratio_max"].as<float>();
        params_.M_YAW_THRES = (*config_file_ptr)["M_YAW_THRES"].as<int>();
        params_.K_YAW_THRES = (*config_file_ptr)["K_YAW_THRES"].as<float>();
        params_.MAX_DETECT_CNT = (*config_file_ptr)["MAX_DETECT_CNT"].as<int>();
        params_.MAX_LOST_CNT = (*config_file_ptr)["MAX_LOST_CNT"].as<int>();
        
        frame_rate_ = (*config_file_ptr)["frame_rate"].as<float>();
        /// ========== 新的 EKF 和 Tracker 初始化 (9D模型修改) ==========
        double dt = 1.0 / std::max(1.0f, frame_rate_);

        // 1. 新增：从配置文件加载9D EKF参数
        EKFParams ekf_params;
        const auto& ekf_config = (*config_file_ptr)["ekf_params"];

        ekf_params.s2qx = ekf_config["sigma2_q_x"].as<double>();
        ekf_params.s2qy = ekf_config["sigma2_q_y"].as<double>();
        ekf_params.s2qz = ekf_config["sigma2_q_z"].as<double>();
        ekf_params.s2qyaw = ekf_config["sigma2_q_yaw"].as<double>(); // 新增
        ekf_params.s2qr = ekf_config["sigma2_q_r"].as<double>();     // 新增

        ekf_params.r_x = ekf_config["r_x_coeff"].as<double>();
        ekf_params.r_y = ekf_config["r_y_coeff"].as<double>();
        ekf_params.r_z = ekf_config["r_z_coeff"].as<double>();
        ekf_params.r_yaw = ekf_config["r_yaw_val"].as<double>();   // 新增

        ekf_params.p0 = ekf_config["p0_init_val"].as<double>();

        // 2. 创建Tracker，传入新参数
        tracker_ = std::make_unique<Tracker>(dt, ekf_params);
        RCLCPP_INFO(this->get_logger(), "New 9D EKF Tracker initialized with dt=%.4f and params from config.", dt);
        // ========== 初始化结束 ==========

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
        // tracker_ = std::make_unique<Tracker>(dt, ekf_params);
        // RCLCPP_INFO(this->get_logger(), "New 6D EKF Tracker initialized with dt=%.4f and params from config.", dt);
        // // ========== 初始化结束 ==========



#ifdef USE_VIDEO
        video_input_ = std::make_shared<VideoInput>(ws_dir_path / (*config_file_ptr)["video_relative_path"].as<std::string>());
#else
#ifdef USE_IMAGES
        images_input_ = std::make_shared<ImagesInput>(ws_dir_path / (*config_file_ptr)["images_relative_path"].as<std::string>());
#else
        // 初始化相机和检测器
        camera_ = std::make_shared<Camera>((*config_file_ptr)["cam_ip"].as<std::string>(), (*config_file_ptr)["pc_ip"].as<std::string>());
        camera_->setExposureTime((*config_file_ptr)["camera_ExposureTime"].as<float>());
        camera_->setGain((*config_file_ptr)["camera_Gain"].as<float>());
#endif
#endif
        reset_com_time = (*config_file_ptr)["reset_com_time"].as<float>();
        serial_delay_time = (*config_file_ptr)["serial_delay_time"].as<float>();

        predictor3d_fit_step = (*config_file_ptr)["predictor3d_fit_step"].as<int>();
        predictor3d_predict_step = (*config_file_ptr)["predictor3d_predict_step"].as<int>();
        predictor3d_fourier_fit_order = (*config_file_ptr)["predictor3d_fourier_fit_order"].as<int>();
        predictor3d_fire_distance = (*config_file_ptr)["predictor3d_fire_distance"].as<float>();

        if (enemy_color_ == "RED") {
            params_.enemy_color = Params::RED;
        } else if (enemy_color_ == "BLUE") {
            params_.enemy_color = Params::BLUE;
        } else if (enemy_color_ == "GREEN") {
            params_.enemy_color = Params::GREEN;
        } else if (enemy_color_ == "BOTH") {
            params_.enemy_color = Params::BOTH;
        } else {
            // 处理错误情况，设置默认值
            enemy_color_ = "GREEN";
            params_.enemy_color = Params::GREEN;
        }

        light_detector_ = std::make_shared<LightBarDetector>(params_, config_file_ptr, this);
        armor_detector_ = std::make_shared<ArmorDetector>(config_file_ptr, this);
        classifier_ = std::make_shared<ArmorClassifier>(config_file_ptr, this);
        armor_solver_ = std::make_shared<ArmorSolver>(config_file_ptr, this);
        ballistic_solver_ = std::make_shared<BallisticSolver>(config_file_ptr, this);

        trans_pred_ = std::make_shared<Trans2DPredTo3DClass>(config_file_ptr);

        predictor3d = std::make_shared<PositionPredictor3D>(predictor3d_fit_step);
        predictor3dArmorPredictions.push_back(cv::Point3f(0,0,0));
        predictor3dCenterPredictions.push_back(cv::Point3f(0,0,0));

        rest_frame_ = std::make_shared<RestFrame>();
        rest_frame_ -> updateCamOrientation(0, 0, 0);
        rest_frame_ -> updateCamPosition(0, 0, 0);

        oscilloscope_fire_ = std::make_shared<Oscilloscope>(640, 120, "Fire Data Oscilloscope");
        oscilloscope_fire_ -> setScale(1.0);
        oscilloscope_fire_ -> setOffset(-0.5);

        oscilloscope_common_ = std::make_shared<Oscilloscope>(640, 120, "Common Debug Oscilloscope");
        oscilloscope_common_ -> setScale(1.0);
        oscilloscope_common_ -> setOffset(-0.5);

        fire_data_fitter_ = std::make_shared<PeriodicDataFitter>(predictor3d_fit_step);
        fire_data_fitter_ -> setPeriod(1);
        pred_fire_data_filter_ = std::make_shared<SimpleDataFilter>(1);
        pred_fire_data_filter_ -> setExponentialAlpha((*config_file_ptr)["pred_fire_data_smooth_factor"].as<float>());
        pred_fire_data_filter_ -> addPoint(0.0);

        armor_distance_filter_ = std::make_shared<SimpleDataFilter>(1);
        armor_distance_filter_ -> setExponentialAlpha((*config_file_ptr)["armor_distance_smooth_factor"].as<float>());
        armor_distance_filter_ -> addPoint(0.0);

        fps_counter = std::make_shared<FrameRateCounter>(30); // 30帧滑动窗口统计帧率

        com_timer_thread_ = std::thread(std::bind(&SerialCommunicationClass::timerThread, serial_communication_));
        com_timer_thread_.detach();

        // 串口通信下位机初始化
        serial_communication_->sendData(0, 0, false);

#ifdef DEBUG_CODE
        debug_code();
#endif

        // 创建定时器
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds((int)(1000/frame_rate_)), // 33
            std::bind(&ArmorDetectNode::processImage, this));


        RCLCPP_INFO(this->get_logger(), "ArmorDetectNode initialized");
    }

    ~ArmorDetectNode() {
        serial_communication_->~SerialCommunicationClass();
        cv::destroyAllWindows();
        pthread_mutex_destroy(&g_mutex);
        RCLCPP_INFO(this->get_logger(), "ArmorDetectNode destroyed");
    }

private:
    void debug_code() {
        while (true) {
            static double debug_time_count = 0.0;
            double debug_freq = 0.3;
            double debug_yaw = std::cos(debug_time_count*M_PI*debug_freq) * M_PI / 6;
            double debug_pitch = std::sin(debug_time_count*M_PI*debug_freq) * M_PI / 6;
            serial_communication_->sendData(debug_pitch, debug_yaw);
            RCLCPP_INFO(this->get_logger(), "send debug data: yaw[%.2f] pitch[%.2f]", debug_yaw, debug_pitch);
            RCLCPP_INFO(this->get_logger(), "received data: yaw[%.2f] pitch[%.2f]", last_yaw_rad_delayed_, last_pitch_rad_delayed_);
            cv::Mat frame;
            pthread_mutex_lock(&g_mutex);
            if (!g_image.empty()) {
                frame = g_image.clone();
                image_used = true;
            }
            pthread_mutex_unlock(&g_mutex);
            if (!frame.empty()) {
                cv::imshow("debug_code", frame);
                cv::waitKey(1);
            }
            auto start = std::chrono::steady_clock::now();
            std::this_thread::sleep_until(start + std::chrono::microseconds(33000));
            debug_time_count += 0.033;
        }
        /* std::thread([&]() {
            double debug_time_count = 0.0;
            while (true) {
                auto start = std::chrono::steady_clock::now();

                SerialData fakeSerialData;
                fakeSerialData.bullet_velocity = 25.0;  // 子弹速度
                fakeSerialData.bullet_angle = std::sin(debug_time_count * 0.5 * (2*M_PI)) * 1.8 / 30 * 15;    // 子弹角度
                fakeSerialData.gimbal_yaw = static_cast<int16_t>(std::cos(debug_time_count * 0.5 * (2*M_PI)) * 4095 / 180 * 15);       // 云台当前偏航角
                fakeSerialData.color = 1;            // 敌方颜色(0:红色, 1:蓝色)

                serialDataCallback(fakeSerialData);

                std::this_thread::sleep_until(start + std::chrono::microseconds(10000));  // 大约10ms周期
                debug_time_count += 0.01;
            }
        }).detach(); */
    }

    void serialDataCallback(const SerialData& msg) {
        bullet_velocity_ = msg.bullet_velocity;
        current_pitch_ = ((float)(msg.bullet_angle)) * 30 / 1.8 * M_PI / 180; // 测定pitch轴传入数据1.8大约对应30°
        current_yaw_ = ((float)(msg.gimbal_yaw)) * M_PI / 4096.0;  // 一圈对应[-4096, 4095]
        enemy_color_ = (msg.color == 0) ? "RED" : "BLUE";
        if (enemy_color_ == "RED") {
            params_.enemy_color = Params::RED;
        } else if (enemy_color_ == "BLUE") {
            params_.enemy_color = Params::BLUE;
        }
        if (light_detector_) {
            light_detector_->setEnemyColor(msg.color == 0 ? Params::RED : Params::BLUE);
        }

        if (current_yaw_ < -M_PI/2 && last_yaw_rad_ > M_PI/2) {
            yaw_circle_ += 1;
        } else if (current_yaw_ > M_PI/2 && last_yaw_rad_ < -M_PI/2) {
            yaw_circle_ -= 1;
        }

        total_yaw_rad_ = yaw_circle_ * 2 * M_PI + current_yaw_;
        last_pitch_rad_ = current_pitch_;
        last_yaw_rad_ = current_yaw_;

        RCLCPP_DEBUG(this->get_logger(), 
            "Received serial data: v=%.2f, pitch=%.2f, yaw=%.2f, color=%s \nyaw_circle=%d, total_yaw_rad=%.2f",
            bullet_velocity_, current_pitch_, current_yaw_, enemy_color_.c_str(),
            yaw_circle_, total_yaw_rad_);


        std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
        DelayInfos now_serial_infos;
        now_serial_infos.last_pitch_rad_ = last_pitch_rad_;
        now_serial_infos.last_yaw_rad_ = last_yaw_rad_;
        now_serial_infos.total_yaw_rad_ = total_yaw_rad_;
        now_serial_infos.push_time = current_time;
        serial_infos_delay_.push(now_serial_infos);
        while (serial_infos_delay_.size() > 1 && 
               std::chrono::duration_cast<std::chrono::milliseconds>(current_time - serial_infos_delay_.front().push_time).count() > serial_delay_time) {
            serial_infos_delay_.pop();
        }
        DelayInfos delayed_serial_infos = serial_infos_delay_.front();
        last_pitch_rad_delayed_ = delayed_serial_infos.last_pitch_rad_;
        last_yaw_rad_delayed_ = delayed_serial_infos.last_yaw_rad_;
        total_yaw_rad_delayed_ = delayed_serial_infos.total_yaw_rad_;

        ground_stable_point = cv::Point2f(500+total_yaw_rad_delayed_*yaw_rad_to_x_pixel_ratio, 500+last_pitch_rad_delayed_*pitch_rad_to_y_pixel_ratio);

        rest_frame_ -> updateCamOrientation(last_yaw_rad_delayed_, last_pitch_rad_delayed_, 0);
        rest_frame_ -> updateCamPosition(0, 0, 0); // 预留位置接口
        
        RCLCPP_DEBUG(this->get_logger(), "ground_stable_point: %.2f %.2f", ground_stable_point.x, ground_stable_point.y);

    }

    void drawResults(cv::Mat& image, 
                     const std::vector<Light>& lights,
                     const std::vector<Armor>& armors,
                     const std::vector<ArmorResult>& classifyResults,
                     const std::vector<ArmorResult>& classifyResults_forFourierPredict) {
        cv::Mat result = image.clone();

        // 0. 绘制平面地面系不动点（DEBUG）
        cv::circle(result, ground_stable_point, 10, cv::Scalar(0, 255, 0), 2);
        /* cv::circle(result, cv::Point2f(1000, 1000) - ground_stable_point, 10, cv::Scalar(0, 255, 0), 2);
        for (const auto& res : classifyResults) {
            for (size_t i = 0; i < res.corners.size() && i < 4; i++) {
                cv::line(result, res.corners[i] - ground_stable_point + cv::Point2f(500, 500), 
                        res.corners[(i+1)%4] - ground_stable_point + cv::Point2f(500, 500), 
                        cv::Scalar(0, 255, 0), 2);
            }    
        } */                       
        // 绘制3D面系不动点
        cv::Point3f test_point_pos;
        std::vector<float> rest_frame_test_point = {0, 1000, 0};
        std::vector<float> cam_normal_test_point = rest_frame_ -> getCamPositionFromWorld(rest_frame_test_point[0], rest_frame_test_point[1], rest_frame_test_point[2]);
        std::vector<float> pnp_pos_test_point = rest_frame_ -> normalToPnpResultFrame(cam_normal_test_point[0], cam_normal_test_point[1], cam_normal_test_point[2]);
        test_point_pos.x = pnp_pos_test_point[0];
        test_point_pos.y = pnp_pos_test_point[1];
        test_point_pos.z = pnp_pos_test_point[2];
        cv::Point2f test_point_pos_pixel = armor_solver_->project3DToPixel(test_point_pos);
        cv::circle(result, test_point_pos_pixel, 8, cv::Scalar(255, 0, 255), 2);

        // 1. 绘制灯条（绿色）
        for (const auto& light : lights) {
            cv::Point2f vertices[4];
            light.el.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(result, vertices[i], vertices[(i + 1) % 4], 
                        cv::Scalar(0, 255, 0), 2);
            }
        }

        // 2. 绘制装甲板候选区域（黄色）
        for (const auto& armor : armors) {
            for (size_t i = 0; i < armor.corners.size() && i < 4; i++) {
                cv::line(result, armor.corners[i], 
                        armor.corners[(i+1)%4], 
                        cv::Scalar(0, 255, 255), 2);
            }

            // 显示装甲板置信度
            if (!armor.corners.empty()) {
                std::string conf_str = cv::format("conf: %.2f", armor.confidence);
                cv::Point text_pos(armor.corners[0].x, armor.corners[0].y - 10);
                cv::putText(result, conf_str, text_pos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                        cv::Scalar(0, 255, 255), 1);
            }
        }

        // 3. 绘制最终识别结果（红色）和跟踪信息
        /* for (const auto& res : classifyResults_forFourierPredict) {
            if (res.is_steady_tracked) {
                for (auto& prediction : res.predictions) {
                    cv::circle(result, prediction, 3, cv::Scalar(0, 255, 0), -1);
                }
            }
        } */
        for (const auto& res : classifyResults) {
            // 绘制装甲板轮廓
            if (res.is_tracked_now) {
                for (size_t i = 0; i < res.corners.size() && i < 4; i++) {
                    cv::line(result, res.corners[i], 
                            res.corners[(i+1)%4], 
                            cv::Scalar(0, 0, 255), 2);
                }    
            } else {
                for (size_t i = 0; i < res.corners.size() && i < 4; i++) {
                    cv::line(result, res.corners[i], 
                            res.corners[(i+1)%4], 
                            cv::Scalar(255, 0, 255), 2);
                }    
            }

            // 绘制预测中心点
            for (auto& prediction : res.predictions) {
                cv::circle(result, prediction, 3, cv::Scalar(255, 0, 255), -1);
            }
            cv::circle(result, res.center_predicted, 3, cv::Scalar(0, 255, 255), -1);

            // 绘制中心点和编号
            cv::Point2f center = res.center;
            cv::circle(result, center, 3, cv::Scalar(0, 0, 255), -1);

            std::string text = cv::format("N%d (%.2f)", 
                                        res.number, 
                                        res.confidence);
            cv::Point text_pos(res.corners[1].x, res.corners[1].y - 10);

            // 使用黑色描边使文字更清晰
            cv::putText(result, text, text_pos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                        cv::Scalar(0, 0, 0), 3);
            cv::putText(result, text, text_pos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                        cv::Scalar(0, 0, 255), 1);

            // 添加跟踪状态显示
            std::string track_text = "TRACKING";
            cv::Point track_pos(center.x - 30, center.y + 30);
            cv::putText(result, track_text, track_pos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 1);
        }

        // 在窗口中显示图像
        cv::imshow("Armor Detection", result);
        cv::waitKey(1);  // 确保窗口刷新
    }

    void processImage() {
        
        cv::Mat frame;
#ifdef USE_VIDEO
        while (image_used)
        {
            usleep(1000);
        }
#endif
#ifdef USE_IMAGES
        while (image_used)
        {
            usleep(1000);
        }
#endif
        pthread_mutex_lock(&g_mutex);
        if (!g_image.empty()) {
            frame = g_image.clone();
            image_used = true;
        }
        pthread_mutex_unlock(&g_mutex);

        if (!frame.empty()) {
            
#ifdef SAVE_IMG_FREQ
            frame_count_ += 1;
            if (frame_count_ % SAVE_IMG_FREQ == 0 && frame_count_ / SAVE_IMG_FREQ < 2000) {
                // 创建保存目录
                fs::create_directories("camera_images");
                // 生成文件名（00001.jpg 格式）
                std::ostringstream filename;
                filename << "camera_images/"
                        << std::setw(5) << std::setfill('0') << (frame_count_ / SAVE_IMG_FREQ)
                        << ".jpg";
                cv::imwrite(filename.str(), frame);
            }
#endif

            //cv::flip(frame, frame, -1);  // 翻转图像（上下翻转）

            std::vector<Light> lights;
            std::vector<Armor> armors;
            std::vector<ArmorResult> classifyResults;
            std::vector<ArmorResult> classifyResults_forFourierPredict;
            std::vector<std::vector<ArmorResult>> classifyResults_expanded;

            // 检测灯条
            light_detector_->detectLights({frame});
            light_detector_->processLights();
            lights = light_detector_->getLights();
            
            // 检测装甲板
            armors = armor_detector_->detectArmors(lights);
            has_valid_target_ = false;


            classifyResults_expanded = classifier_->classify(frame, armors, ground_stable_point);
            classifyResults = classifyResults_expanded[0];
            classifyResults_forFourierPredict = classifyResults_expanded[1];

            if (classifyResults.empty()) {
                if (tracker_->state != Tracker::LOST) {
                    tracker_->predict();
                }

                bool fire_flag = false;
            	if (
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_com_time).count() >= reset_com_time) {
                	serial_communication_->sendData(0, 0, false);
                	pitch_integration = 0; // 积分项重置
                	predictor3d -> clearHistory(); 
                    fire_data_fitter_ -> clearHistory();
                    pred_fire_data_filter_ -> clearHistory();
                } else {
                    pred_fire_data_filter_ -> addPoint(0.0);
                    fire_flag = pred_fire_data_filter_ -> getExponentialValue() > 0.5;
                    serial_communication_->sendData(last_command_pitch_, last_command_yaw_, fire_flag);
                    if (fire_flag) {
                        cv::circle(frame, last_aim_yaw_pitch_pixel_, 8, cv::Scalar(0, 0, 255), 2);
                    } else {
                        cv::circle(frame, last_aim_yaw_pitch_pixel_, 8, cv::Scalar(255, 255, 0), 2);
                    }
                    cv::Point3f predicted_armor_pos = predictor3dArmorPredictions[predictor3dPrediction_nowIndex];
                    cv::Point3f predicted_aim_pos = predictor3dCenterPredictions[predictor3dPrediction_nowIndex];
                	predictor3d -> addPoint(predicted_armor_pos);
                	if (predictor3dPrediction_nowIndex < predictor3dArmorPredictions.size()-1) {
                	    predictor3dPrediction_nowIndex += 1;
                	}
#ifdef USE_PREDICTOR3D
                    // 转换回pnp相机坐标系
                    std::vector<float> rest_frame_aim_pos_pred = {predicted_aim_pos.x, predicted_aim_pos.y, predicted_aim_pos.z};
                    std::vector<float> cam_normal_aim_pos_pred = rest_frame_ -> getCamPositionFromWorld(rest_frame_aim_pos_pred[0], rest_frame_aim_pos_pred[1], rest_frame_aim_pos_pred[2]);
                    std::vector<float> pnp_aim_pos_pred = rest_frame_ -> normalToPnpResultFrame(cam_normal_aim_pos_pred[0], cam_normal_aim_pos_pred[1], cam_normal_aim_pos_pred[2]);
                    predicted_aim_pos.x = pnp_aim_pos_pred[0];
                    predicted_aim_pos.y = pnp_aim_pos_pred[1];
                    predicted_aim_pos.z = pnp_aim_pos_pred[2];

                    // 转换回pnp相机坐标系
                    std::vector<float> rest_frame_armor_pos_pred = {predicted_armor_pos.x, predicted_armor_pos.y, predicted_armor_pos.z};
                    std::vector<float> cam_normal_armor_pos_pred = rest_frame_ -> getCamPositionFromWorld(rest_frame_armor_pos_pred[0], rest_frame_armor_pos_pred[1], rest_frame_armor_pos_pred[2]);
                    std::vector<float> pnp_armor_pos_pred = rest_frame_ -> normalToPnpResultFrame(cam_normal_armor_pos_pred[0], cam_normal_armor_pos_pred[1], cam_normal_armor_pos_pred[2]);
                    predicted_armor_pos.x = pnp_armor_pos_pred[0];
                    predicted_armor_pos.y = pnp_armor_pos_pred[1];
                    predicted_armor_pos.z = pnp_armor_pos_pred[2];
                    // 绘制瞄准预测点（黄色）
                    cv::Point2f pred_aim_pixel = armor_solver_->project3DToPixel(predicted_aim_pos);
                    cv::circle(frame, pred_aim_pixel, 8, cv::Scalar(0, 255, 255), 2);
                    // 绘制装甲板预测点（蓝色）
                    cv::Point2f pred_armor_pixel = armor_solver_->project3DToPixel(predicted_armor_pos);
                    cv::circle(frame, pred_armor_pixel, 8, cv::Scalar(255, 0, 0), 2);

                    fire_data_fitter_ -> setPeriod(predictor3d->getFourierPeriod());
                    fire_data_fitter_ -> addPoint(0.0);
#endif
                }
#ifdef USE_PREDICTOR3D
                oscilloscope_fire_ -> addDataPoint(fire_flag);
#endif
            } 
            
            if (!classifyResults.empty()) {
                last_com_time = std::chrono::steady_clock::now();
                // 选择最佳目标（置信度最高）
                auto it = std::max_element(
                    classifyResults.begin(), classifyResults.end(),
                    [](const ArmorResult& a, const ArmorResult& b) {
                        return a.confidence < b.confidence;
                    }
                );
                if (it != classifyResults.end()) {
                    auto best_result = *it;
                    AimResult aim = armor_solver_->solveArmor(best_result, last_pitch_rad_delayed_, last_yaw_rad_delayed_);
                    if (aim.valid) {
                        // 查看并滤波z轴距离轴数据
                        armor_distance_filter_ -> addPoint(aim.position.z);
                        aim.position.z = armor_distance_filter_ -> getExponentialValue();
                        oscilloscope_common_ -> addDataPoint(aim.position.z / 6000);

                        // 将pnp结果转换至静止坐标系以稳定预测
                        std::vector<float> cam_normal_pos = rest_frame_ -> pnpResultToNormalFrame(aim.position.x, aim.position.y, aim.position.z);
                        std::vector<float> rest_frame_pos = rest_frame_ -> getWorldPositionFromCam(cam_normal_pos[0], cam_normal_pos[1], cam_normal_pos[2]);
                        std::vector<float> rest_frame_euler_angles = rest_frame_ -> getWorldEulerAnglesFromCam(
                            aim.normal_euler_angles[0], aim.normal_euler_angles[1], aim.normal_euler_angles[2]
                        );

                        // ========== EKF 9D ==========
                        // 1. 构造4维测量向量 z = [xa, ya, za, yaw_a]
                        Tracker::Measurement z;
                        z << rest_frame_pos[0], rest_frame_pos[1], rest_frame_pos[2], aim.normal_euler_angles[1]; // 只用yaw角

                         RCLCPP_INFO(this->get_logger(), "EKF Pre-prediction (Measurement): x=%.3f, y=%.3f, z=%.3f, yaw=%.3f",
                                    z(0), z(1), z(2), z(3));

                        // 2. EKF 状态机逻辑
                        if (tracker_->state == Tracker::LOST) {
                            tracker_->reset(z);
                            current_target_id_ = best_result.number;
                        } else {
                            // 跳变处理：通过ID或距离判断
                            Eigen::Vector3d pred_armor_pos = tracker_->getArmorPosition();
                            double position_diff = (pred_armor_pos - Eigen::Vector3d(rest_frame_pos[0], rest_frame_pos[1], rest_frame_pos[2])).norm();

                            if (best_result.number != current_target_id_ || position_diff > RESET_DISTANCE_THRESHOLD) {
                                if(best_result.number != current_target_id_) {
                                    RCLCPP_WARN(this->get_logger(), "ID switched, resetting tracker.");
                                } else {
                                    RCLCPP_WARN(this->get_logger(), "Position jumped (%.f mm), resetting tracker.", position_diff);
                                }
                                tracker_->reset(z);
                                current_target_id_ = best_result.number;
                            } else {
                                tracker_->predict();
                                tracker_->update(z);
                            }
                        }

                        // 3. 提前预测与弹道解算
                        constexpr float image_latency = 0.013f;
                        constexpr float comm_latency  = 0.010f;
                        float bullet_time = (bullet_velocity_ > 1.0f) ? (std::abs(aim.position.z) / 1000.0f / bullet_velocity_) : 0.0f;
                        float extra_time = 0.100f;
                        float total_delay = image_latency + comm_latency + bullet_time + extra_time;

                        // 获取提前预测后的机器人中心状态
                        Tracker::State future_state = tracker_->predictAhead(total_delay);
                        
                        // 从预测的机器人中心状态，反解出未来时刻装甲板的位置
                        double future_xc = future_state(0), future_yc = future_state(2), future_zc = future_state(4);
                        double future_yaw = future_state(6), future_r = future_state(8);
                        cv::Point3f predicted_aim_pos(
                            future_xc - future_r * sin(future_yaw),
                            future_yc + future_r * cos(future_yaw),
                            future_zc 
                        );

                         RCLCPP_INFO(this->get_logger(), "EKF Post-prediction (Target):  x=%.3f, y=%.3f, z=%.3f",
                                    predicted_aim_pos.x, predicted_aim_pos.y, predicted_aim_pos.z);
                        
                        
                        // // ========== EKF 逻辑 (6D模型修改) ==========

                        // // 1. 构造3维测量向量 z = [xa, ya, za]
                        // Tracker::Measurement z;
                        // z << rest_frame_pos[0], rest_frame_pos[1], rest_frame_pos[2];

                        // // 2. EKF 状态机逻辑
                        // if (tracker_->state == Tracker::LOST) {
                        //     // 如果是丢失状态，用当前测量值重置滤波器
                        //     tracker_->reset(z);
                        //     current_target_id_ = best_result.number;
                        // } else {
                        //     // 跳变处理
                        //     Eigen::Vector3d pred_armor_pos = tracker_->getArmorPosition();
                        //     double position_diff = (pred_armor_pos - Eigen::Vector3d(rest_frame_pos[0], rest_frame_pos[1], rest_frame_pos[2])).norm();

                        //     if (best_result.number != current_target_id_ || position_diff > RESET_DISTANCE_THRESHOLD) {
                        //         if(best_result.number != current_target_id_) {
                        //             RCLCPP_WARN(this->get_logger(), "ID switched, resetting tracker.");
                        //         } else {
                        //             RCLCPP_WARN(this->get_logger(), "Position jumped (%.f mm), resetting tracker.", position_diff);
                        //         }
                        //         tracker_->reset(z);
                        //         current_target_id_ = best_result.number;
                        //     } else {
                        //         tracker_->predict();
                        //         tracker_->update(z);
                        //     }
                        // }

                        // // 3. 提前预测与弹道解算
                        // // 计算总延迟 (这部分逻辑不变)
                        // constexpr float image_latency = 0.013f;
                        // constexpr float comm_latency  = 0.010f;
                        // float bullet_time = (bullet_velocity_ > 1.0f) ? (std::abs(aim.position.z) / 1000.0f / bullet_velocity_) : 0.0f;
                        // float extra_time = 0.100f;
                        // float total_delay = image_latency + comm_latency + bullet_time + extra_time;

                        // // 获取提前预测后的装甲板状态
                        // Tracker::State future_state = tracker_->predictAhead(total_delay);
                        
                        // // 直接从未来状态中提取预测位置
                        // cv::Point3f predicted_aim_pos(
                        //     future_state(0),
                        //     future_state(2),
                        //     future_state(4)
                        // );
                        

                        RCLCPP_DEBUG(this->get_logger(), "yaw: %.2f" , aim.yaw );
                        RCLCPP_DEBUG(this->get_logger(), "distance: %.2f" , aim.distance );
                        RCLCPP_DEBUG(this->get_logger(), "position: (%.2f, %.2f, %.2f)" , aim.position.x, aim.position.y, aim.position.z);
                        RCLCPP_DEBUG(this->get_logger(), "Future armor pos: (%.2f, %.2f, %.2f)",
                                    predicted_aim_pos.x, predicted_aim_pos.y, predicted_aim_pos.z);

                        bool fire_flag = true;

                        // 预测未来位置（旧卡尔曼滤波）
                        // cv::Point3f predicted_aim_pos = angle_kalman_->predictKalmanFilter(total_delay);

                        //cv::Point3f predicted_aim_pos = trans_pred_->trans2DPredTo3D(best_result, aim.position, classifyResults_forFourierPredict,
                        //                                                         total_delay, fps_counter->fps());
                        
                        // 测试静止坐标系
                        /* std::vector<float> cam_normal_pos = rest_frame_ -> pnpResultToNormalFrame(predicted_aim_pos.x, predicted_aim_pos.y, predicted_aim_pos.z);
                        std::vector<float> rest_frame_pos = rest_frame_ -> getWorldPositionFromCam(cam_normal_pos[0], cam_normal_pos[1], cam_normal_pos[2]);

                        std::vector<float> rest_frame_pos_new = rest_frame_pos;

                        std::vector<float> cam_normal_pos_new = rest_frame_ -> getCamPositionFromWorld(rest_frame_pos_new[0], rest_frame_pos_new[1], rest_frame_pos_new[2]);
                        std::vector<float> pnp_pos_new = rest_frame_ -> normalToPnpResultFrame(cam_normal_pos_new[0], cam_normal_pos_new[1], cam_normal_pos_new[2]); */

                        // 3D位置预测器
                        cv::Point3f rest_frame_pos_Point3f(rest_frame_pos[0], rest_frame_pos[1], rest_frame_pos[2]);
                        predictor3d -> addPoint(rest_frame_pos_Point3f);
                        predictor3d -> fitFourier(predictor3d_fit_step, predictor3d_fourier_fit_order);
                        predictor3dCenterPredictions = std::vector<cv::Point3f>(predictor3d_predict_step, cv::Point3f(predictor3d -> getAveragePosition())); //predictor3d -> predictLinear(predictor3d_predict_step); // predictFourier | predictLinear
                        predictor3dArmorPredictions = predictor3d -> predictFourier(predictor3d_predict_step); // predictFourier | predictLinear
                        predictor3dPrediction_nowIndex = 0;
                        size_t predictor3dPrediction_indexToAim = std::min(predictor3d_predict_step-1, (int)(total_delay * fps_counter->fps())); // total_delay
#ifdef USE_PREDICTOR3D
                        predicted_aim_pos = predictor3dCenterPredictions[predictor3dPrediction_indexToAim]; // todo

                        cv::Point3f predicted_armor_pos = predictor3dArmorPredictions[predictor3dPrediction_indexToAim]; // todo

                        // 计算弹道最近点并绘制（大紫色圈）
                        std::vector<float> cam_position = rest_frame_ -> getCamPosition();
                        cv::Point3f bullet_nearest_point = ballistic_solver_ -> calcNearestPointWithAirResistance( // todo
                            rest_frame_pos_Point3f / 1000, {cam_position[0], cam_position[1], cam_position[2]}, last_aim_yaw_pitch_, bullet_velocity_) * 1000;
                        std::vector<float> cam_normal_bullet_nearest_point = rest_frame_ -> getCamPositionFromWorld(bullet_nearest_point.x, bullet_nearest_point.y, bullet_nearest_point.z);
                        std::vector<float> pnp_bullet_nearest_point = rest_frame_ -> normalToPnpResultFrame(cam_normal_bullet_nearest_point[0], cam_normal_bullet_nearest_point[1], cam_normal_bullet_nearest_point[2]);
                        cv::Point2f bullet_nearest_point_pixel = armor_solver_->project3DToPixel({pnp_bullet_nearest_point[0], pnp_bullet_nearest_point[1], pnp_bullet_nearest_point[2]});
                        cv::circle(frame, bullet_nearest_point_pixel, 15, cv::Scalar(255, 0, 255), 2);
                        RCLCPP_DEBUG(this->get_logger(), "bullet_nearest_point: (%.2f, %.2f, %.2f)",
                                    bullet_nearest_point.x, bullet_nearest_point.y, bullet_nearest_point.z);

                        bool armor_near_flag = cv::norm(bullet_nearest_point - rest_frame_pos_Point3f) < predictor3d_fire_distance; // todo

                        //oscilloscope_fire_ -> addDataPoint(cv::norm(bullet_nearest_point - predicted_armor_pos)/400);

                        // 转换回pnp相机坐标系
                        std::vector<float> rest_frame_armor_pos_pred = {predicted_armor_pos.x, predicted_armor_pos.y, predicted_armor_pos.z};
                        std::vector<float> cam_normal_armor_pos_pred = rest_frame_ -> getCamPositionFromWorld(rest_frame_armor_pos_pred[0], rest_frame_armor_pos_pred[1], rest_frame_armor_pos_pred[2]);
                        std::vector<float> pnp_armor_pos_pred = rest_frame_ -> normalToPnpResultFrame(cam_normal_armor_pos_pred[0], cam_normal_armor_pos_pred[1], cam_normal_armor_pos_pred[2]);
                        predicted_armor_pos.x = pnp_armor_pos_pred[0];
                        predicted_armor_pos.y = pnp_armor_pos_pred[1];
                        predicted_armor_pos.z = pnp_armor_pos_pred[2];

                        fire_data_fitter_ -> setPeriod(predictor3d->getFourierPeriod());
                        fire_data_fitter_ -> addPoint(armor_near_flag);
                        pred_fire_data_filter_ -> addPoint(fire_data_fitter_ -> isUpper(predictor3dPrediction_indexToAim, 0.1) || fire_data_fitter_ -> getA0() > 0.8);
                        fire_flag = pred_fire_data_filter_ -> getExponentialValue() > 0.5;
                        //oscilloscope_fire_ -> addDataPoint(pred_fire_data_filter_ -> getExponentialValue());
#endif

                        // 转换回pnp相机坐标系
                        std::vector<float> rest_frame_aim_pos_pred = {predicted_aim_pos.x, predicted_aim_pos.y, predicted_aim_pos.z};
                        std::vector<float> cam_normal_aim_pos_pred = rest_frame_ -> getCamPositionFromWorld(rest_frame_aim_pos_pred[0], rest_frame_aim_pos_pred[1], rest_frame_aim_pos_pred[2]);
                        std::vector<float> pnp_aim_pos_pred = rest_frame_ -> normalToPnpResultFrame(cam_normal_aim_pos_pred[0], cam_normal_aim_pos_pred[1], cam_normal_aim_pos_pred[2]);
                        predicted_aim_pos.x = pnp_aim_pos_pred[0];
                        predicted_aim_pos.y = pnp_aim_pos_pred[1];
                        predicted_aim_pos.z = pnp_aim_pos_pred[2];

                        // 弹道解算
                        RCLCPP_DEBUG(this->get_logger(), "aim pos: (%.2f, %.2f, %.2f)",
                                    predicted_aim_pos.x, predicted_aim_pos.y, predicted_aim_pos.z);
                        BallisticInfo ballistic_result = ballistic_solver_ -> calcBallisticAngle(
                            predicted_aim_pos.x, 
                            predicted_aim_pos.y, 
                            predicted_aim_pos.z,
                            delta_x_,
                            delta_y_,
                            delta_z_,
                            bullet_velocity_,
                            last_pitch_rad_delayed_,//pitch_integration | last_pitch_rad_delayed_ #todo
                            last_yaw_rad_delayed_
                        );
                        
                        if (ballistic_result.valid) {
                            // RCLCPP_INFO(this->get_logger(), "Target detected, publishing command");
                            has_valid_target_ = true;

                            pitch_integration += ballistic_result.delta_pitch_rad * 0.03;

                            if (pitch_integration > 0.3) {
                                pitch_integration = 0.3;
                            }
                            if (pitch_integration < -0.3) {
                                pitch_integration = -0.3;
                            }
                            
                            // 发布云台控制命令
                            float command_pitch = last_pitch_rad_delayed_ + ballistic_result.delta_pitch_rad * 0.5 + pitch_integration; // PI控制
                            float command_yaw = ballistic_result.target_yaw_rad;
                            last_command_pitch_ = command_pitch;
                            last_command_yaw_ = command_yaw;
                            serial_communication_->sendData(command_pitch, command_yaw, fire_flag);

                            RCLCPP_DEBUG(this->get_logger(),
                                "Target %d: Position[%.2f, %.2f, %.2f] mm, "
                                "Command[pitch: %.2f, yaw: %.2f] rad",
                                best_result.number,
                                predicted_aim_pos.x, predicted_aim_pos.y, predicted_aim_pos.z,
                                command_pitch, command_yaw);
                            
                            // 绘制瞄准预测点（黄色）
                            cv::Point2f pred_aim_pixel = armor_solver_->project3DToPixel(predicted_aim_pos);
                            cv::circle(frame, pred_aim_pixel, 8, cv::Scalar(0, 255, 255), 2);

                            // 计算并绘制瞄准时目标画面中心（天蓝色：未开火 | 红色：开火）
                            cv::Point2f aim_yaw_pitch = cv::Point2f(ballistic_result.target_yaw_rad, last_pitch_rad_delayed_ + ballistic_result.delta_pitch_rad);
                            cv::Point2f aim_yaw_pitch_pixel = cv::Point2f(
                                frame.cols / 2 - (aim_yaw_pitch.x - last_yaw_rad_delayed_) * yaw_rad_to_x_pixel_ratio, 
                                frame.rows / 2 - (aim_yaw_pitch.y - last_pitch_rad_delayed_) * pitch_rad_to_y_pixel_ratio);
                            last_aim_yaw_pitch_ = aim_yaw_pitch;
                            last_aim_yaw_pitch_pixel_ = aim_yaw_pitch_pixel;
                            if (fire_flag) {
                                cv::circle(frame, aim_yaw_pitch_pixel, 8, cv::Scalar(0, 0, 255), 2);
                            } else {
                                cv::circle(frame, aim_yaw_pitch_pixel, 8, cv::Scalar(255, 255, 0), 2);
                            }
                            RCLCPP_DEBUG(this->get_logger(), "aim center yaw pitch: (%.2f, %.2f)",
                                    aim_yaw_pitch.x, aim_yaw_pitch.y);
#ifdef USE_PREDICTOR3D
                            cv::Point2f pred_armor_pixel = armor_solver_->project3DToPixel(predicted_armor_pos);
                            // 绘制装甲板预测点（蓝色）
                            cv::circle(frame, pred_armor_pixel, 8, cv::Scalar(255, 0, 0), 2);
                            oscilloscope_fire_ -> addDataPoint(fire_flag);
                            //oscilloscope_fire_ -> addDataPoint(fire_data_fitter_ -> smooth(0));
                            //oscilloscope_fire_ -> addDataPoint(fire_data_fitter_ -> isRising(0));
#endif
                        }
                    }
                    
                }
            }
            drawResults(frame, lights, armors, classifyResults, classifyResults_forFourierPredict);
#ifdef USE_PREDICTOR3D
            oscilloscope_fire_ -> update();
            oscilloscope_fire_ -> putText("period:"+std::to_string(predictor3d->getFourierPeriod()), cv::Point2f(500, 20), cv::Scalar(0, 255, 0), 0.7);
            oscilloscope_fire_ -> show();
#endif
            oscilloscope_common_ -> update();
            oscilloscope_common_ -> show();

            //计算帧率
            fps_counter->tick();
            
            // // 显示当前参数状态
            // cv::putText(frame, 
            //     cv::format("V: %.1f m/s, P: %.1f, Y: %.1f", 
            //         bullet_velocity_, last_pitch_rad_delayed_, last_yaw_rad_delayed_),
            //     cv::Point(10, 60),
            //     cv::FONT_HERSHEY_SIMPLEX, 0.5,
            //     cv::Scalar(0, 255, 0), 1);
        }        

        // 获取处理帧率
        RCLCPP_INFO(this->get_logger(), "frame rate: %.1f fps\n" , fps_counter->fps());
    }

    // 参数文件
    std::shared_ptr<YAML::Node> config_file_ptr; 

    // 处理目标丢失情况
    // void handleTargetLost() {
    //     if (!is_target_lost_) {
    //         RCLCPP_WARN(get_logger(), "Target lost!");
    //         is_target_lost_ = true;
    //         last_track_time_ = this->now();
    //     }
    // }
    // 新增成员变量
    int current_target_id_ = -1;      // 当前跟踪目标ID
    cv::Point3f last_observed_pos_;   // 上一帧观测位置
    bool is_target_lost_ = false;     // 目标丢失标志
    rclcpp::Time last_track_time_;    // 最后有效跟踪时间

    // 配置参数
    // static constexpr float RESET_DISTANCE_THRESHOLD = 400.0f; // 单位：mm
    // static constexpr float MAX_LOST_TIME = 0.5f;              // 单位：秒
    float RESET_DISTANCE_THRESHOLD; // 单位：mm
    float MAX_LOST_TIME;              // 单位：秒
    // 成员变量
    rclcpp::TimerBase::SharedPtr timer_;
    std::thread com_timer_thread_;
    
    std::shared_ptr<Camera> camera_;
    std::shared_ptr<LightBarDetector> light_detector_;
    std::shared_ptr<ArmorDetector> armor_detector_;
    std::shared_ptr<ArmorSolver> armor_solver_;
    std::shared_ptr<ArmorClassifier> classifier_;
    std::shared_ptr<BallisticSolver> ballistic_solver_;

    std::shared_ptr<VideoInput> video_input_;
    std::shared_ptr<ImagesInput> images_input_;
    float frame_rate_;

    std::shared_ptr<Trans2DPredTo3DClass> trans_pred_;
    std::shared_ptr<RestFrame> rest_frame_;
    std::shared_ptr<Oscilloscope> oscilloscope_fire_;
    std::shared_ptr<PeriodicDataFitter> fire_data_fitter_;
    std::shared_ptr<SimpleDataFilter> pred_fire_data_filter_;
    
    std::shared_ptr<Oscilloscope> oscilloscope_common_;
    std::shared_ptr<SimpleDataFilter> armor_distance_filter_;
    
    float bullet_velocity_;
    float current_pitch_;
    float current_yaw_;
    float delta_x_;
    float delta_y_;
    float delta_z_;
    int yaw_circle_ = 0;
    float last_pitch_rad_ = 0;
    float last_yaw_rad_ = 0;
    float total_yaw_rad_ = 0;
    float last_pitch_rad_delayed_ = 0;
    float last_yaw_rad_delayed_ = 0;
    float total_yaw_rad_delayed_ = 0;
    struct DelayInfos {
        float last_pitch_rad_;
        float last_yaw_rad_;
        float total_yaw_rad_;
        std::chrono::steady_clock::time_point push_time;
    };
    std::queue<DelayInfos> serial_infos_delay_;
    float serial_delay_time;
    bool has_valid_target_;
    std::string enemy_color_;
    Params params_;

    // EKF/Tracker 相关新增成员
    std::unique_ptr<Tracker> tracker_;
    double last_continuous_yaw_ = 0.0; // 用于连续化yaw角

    // 帧率计算器
    std::shared_ptr<FrameRateCounter> fps_counter;
#ifdef SAVE_IMG_FREQ
    long long frame_count_ = 0;
#endif
    float pitch_integration = 0.0;
    cv::Point2f ground_stable_point;
    std::shared_ptr<SerialCommunicationClass> serial_communication_;
    float reset_com_time;
    std::chrono::steady_clock::time_point last_com_time = std::chrono::steady_clock::now();
    float yaw_rad_to_x_pixel_ratio;
    float pitch_rad_to_y_pixel_ratio;
    std::shared_ptr<PositionPredictor3D> predictor3d;
    int predictor3dPrediction_nowIndex = 0;
    std::vector<cv::Point3f> predictor3dArmorPredictions;
    std::vector<cv::Point3f> predictor3dCenterPredictions;
    int predictor3d_fit_step;
    int predictor3d_predict_step;
    int predictor3d_fourier_fit_order;
    float predictor3d_fire_distance;
    cv::Point2f last_aim_yaw_pitch_;
    cv::Point2f last_aim_yaw_pitch_pixel_;
    float last_command_pitch_;
    float last_command_yaw_;
};

std::shared_ptr<ArmorDetectNode> node;
void signalHandler(int signum) {
    if (node) {
        rclcpp::shutdown();
    }
}
int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    node = std::make_shared<ArmorDetectNode>();
    signal(SIGINT, signalHandler);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
