// ArmorDetect_Node.cpp
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include "camera/Camera.h"
#include "2d_armor_detector/LightBarDetector.h"
#include "2d_armor_detector/ArmorDetector.h"
#include "2d_armor_detector/ArmorClassifier.h"
#include "3d_processing/ArmorSolver.h"
//#include "armor_detector/ArmorAngleKalman.h"

//#include "auto_aim/msg/serial_data.hpp"
//#include "auto_aim/msg/gimbal_command.hpp"
#include <chrono>
#include <string>
#include <thread>
#include <3d_processing/BallisticSolver.h>
#include <yaml-cpp/yaml.h>
#include "utils/FrameRateCounter.h"
#include "2d_armor_detector/UnwarpUtils.h"
#include "other_input/VideoInput.h"
#include "other_input/ImagesInput.h"
#include <iostream>
#include <sstream>
#include <filesystem>
#include <unistd.h>
#include <limits.h>
#include <queue>
#include "communication/Com.h"
#include <csignal>
#include "3d_processing/RestFrame.h"
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>
#include "predictor/PredictorMain.h"
#include "2d_armor_detector/YOLOPoseArmorDetector.h"
#include "predictor/PredictorSwitcher.h"
#include "2d_armor_detector/Armor.h"
#include "communication/WatchdogClient.h"
#include "visualizer/RestFrameDraw.h"
#include "communication/HeadIMU.h"
#include "visualizer/YawVisualizer.h"

namespace fs = std::filesystem;

#include "macro/AutoAimMacro.h"

// 全局变量定义
cv::Mat g_image;
pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
bool g_bExit = false;
bool image_used = true;

class ArmorDetectNode : public rclcpp::Node {
public:
    ArmorDetectNode() : Node("armor_detect_node") {
        node_start_time = std::chrono::steady_clock::now();

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



        // 初始化参数
        bullet_velocity_ = (*config_file_ptr)["bullet_velocity_"].as<float>();
        enemy_color_ = (*config_file_ptr)["init_enemy_color"].as<std::string>();

        use_yolo_pose = (*config_file_ptr)["use_yolo_pose"].as<bool>();

        // 根据相机内参自动提取，不再需要手动输入
        // yaw_rad_to_x_pixel_ratio = (*config_file_ptr)["yaw_rad_to_x_pixel_ratio"].as<float>(); 
        // pitch_rad_to_y_pixel_ratio = (*config_file_ptr)["pitch_rad_to_y_pixel_ratio"].as<float>(); 
        const YAML::Node& camera_matrix_Node = (*config_file_ptr)["camera_matrix"];
        yaw_rad_to_x_pixel_ratio = camera_matrix_Node[0][0].as<float>(); 
        pitch_rad_to_y_pixel_ratio = camera_matrix_Node[1][1].as<float>(); 


        max_armor_position_height = (*config_file_ptr)["max_armor_position_height"].as<float>(); 
        
        params_.min_light_height = (*config_file_ptr)["min_light_height"].as<int>();
        params_.light_min_area = (*config_file_ptr)["light_min_area"].as<int>();
        params_.light_max_area = (*config_file_ptr)["light_max_area"].as<int>();
        params_.max_light_wh_ratio = (*config_file_ptr)["max_light_wh_ratio"].as<float>();
        params_.min_light_wh_ratio = (*config_file_ptr)["min_light_wh_ratio"].as<float>();
        params_.light_max_tilt_angle = (*config_file_ptr)["light_max_tilt_angle"].as<float>();
        
        frame_rate_ = (*config_file_ptr)["frame_rate"].as<float>();


#ifdef USE_VIDEO
        video_input_ = std::make_shared<VideoInput>(ws_dir_path / (*config_file_ptr)["video_relative_path"].as<std::string>());
#else
#ifdef USE_IMAGES
        images_input_ = std::make_shared<ImagesInput>(ws_dir_path / (*config_file_ptr)["images_relative_path"].as<std::string>());
#else
        // 初始化相机和检测器
        camera_ = std::make_shared<Camera>((*config_file_ptr)["cam_ip"].as<std::string>(), (*config_file_ptr)["pc_ip"].as<std::string>());
        //camera_ = std::make_shared<Camera>(0);
        camera_->setExposureTime((*config_file_ptr)["camera_ExposureTime"].as<float>());
        camera_->setGain((*config_file_ptr)["camera_Gain"].as<float>());
        camera_ -> start();
#endif
#endif
        serial_delay_time = (*config_file_ptr)["serial_delay_time"].as<float>();

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
        classifier_ = std::make_shared<ArmorClassifier>(config_file_ptr, this, ws_dir_path);
        armor_solver_ = std::make_shared<ArmorSolver>(config_file_ptr, this);
        ballistic_solver_ = std::make_shared<BallisticSolver>(config_file_ptr, this);

        rest_frame_ = std::make_shared<RestFrame>();
        rest_frame_ -> updateCamOrientation(0, 0, 0);
        rest_frame_ -> updateCamPosition(0, 0, 0);

        fps_counter = std::make_shared<FrameRateCounter>(30); // 30帧滑动窗口统计帧率

        predictor_main_ = std::make_shared<PredictorMain>(
            config_file_ptr, this, node_start_time, armor_solver_,
            ballistic_solver_, rest_frame_, fps_counter);

        yolo_pose_armor_detector = std::make_shared<YOLOPoseArmorDetector>(config_file_ptr, this);

        if (yolo_pose_armor_detector) {
            yolo_pose_armor_detector->setEnemyColor(enemy_color_ == "RED" ? Params::RED : Params::BLUE);
        }

        yaw_visualizer_ = std::make_shared<YawVisualizer>();

        // 初始化串口通信器
        serial_communication_ = std::make_shared<SerialCommunicationClass>(this, std::bind(&ArmorDetectNode::serialDataCallback, this, std::placeholders::_1));

        com_timer_thread_ = std::thread(std::bind(&SerialCommunicationClass::timerThread, serial_communication_));
        // com_timer_thread_.detach();

        headIMUInfos.headIMU_communication_ = std::make_shared<HeadIMUSerialCommunicationClass>(std::bind(&ArmorDetectNode::headIMUSerialDataCallback, this, std::placeholders::_1));
        headIMUInfos.headIMU_timer_thread_ = std::thread(std::bind(&HeadIMUSerialCommunicationClass::timerThread, headIMUInfos.headIMU_communication_));

        // 串口通信下位机初始化
        serial_communication_->sendData(0, 0, false);

        watchdog_client = std::make_shared<WatchdogClient>();
        watchdog_client -> init();
        watchdog_client -> feed();
        last_feed_dog_time = std::chrono::steady_clock::now();

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
        // while (true) {
        //     static double debug_time_count = 0.0;
        //     double debug_freq = 0.3;
        //     double debug_yaw = std::cos(debug_time_count*M_PI*debug_freq) * M_PI / 6;
        //     double debug_pitch = std::sin(debug_time_count*M_PI*debug_freq) * M_PI / 6;
        //     serial_communication_->sendData(debug_pitch, debug_yaw);
        //     RCLCPP_INFO(this->get_logger(), "send debug data: yaw[%.2f] pitch[%.2f]", debug_yaw, debug_pitch);
        //     RCLCPP_INFO(this->get_logger(), "received data: yaw[%.2f] pitch[%.2f]", last_yaw_rad_delayed_, last_pitch_rad_delayed_);
        //     cv::Mat frame;
        //     pthread_mutex_lock(&g_mutex);
        //     if (!g_image.empty()) {
        //         frame = g_image.clone();
        //         image_used = true;
        //     }
        //     pthread_mutex_unlock(&g_mutex);
        //     if (!frame.empty()) {
        //         cv::imshow("debug_code", frame);
        //         cv::waitKey(1);
        //     }
        //     auto start = std::chrono::steady_clock::now();
        //     std::this_thread::sleep_until(start + std::chrono::microseconds(33000));
        //     debug_time_count += 0.033;
        // }
        std::thread([&]() {
            double debug_time_count = 0.0;
            while (true) {
                auto start = std::chrono::steady_clock::now();

                SerialData fakeSerialData;
                fakeSerialData.bullet_velocity = 25.0;  // 子弹速度
                fakeSerialData.bullet_angle = std::sin(debug_time_count * 0.5 * (2*M_PI)) * 1.8 / 30 * 15;    // 子弹角度
                fakeSerialData.gimbal_yaw =  
                    // static_cast<int16_t>(60.0 * 4095.0 / 180.0);
                    // static_cast<int16_t>(std::atan2(std::sin(debug_time_count * 2 * M_PI), std::cos(debug_time_count * 2 * M_PI)) * 4095.0 / M_PI / 12); 
                    // static_cast<int16_t>(static_cast<float>((std::atan2(std::sin(debug_time_count * 1.0), std::cos(debug_time_count * 1.0)) > 0) - 0.5) * 4095); 
                    static_cast<int16_t>(std::atan2(std::sin(debug_time_count * 0.3), std::cos(debug_time_count * 0.3)) * 4095.0 / M_PI);
                    // static_cast<int16_t>(std::cos(debug_time_count * 0.5 * (2*M_PI)) * 4095 / 180 * 15);       // 云台当前偏航角
                fakeSerialData.color = 1;            // 敌方颜色(0:红色, 1:蓝色)

                serialDataCallback(fakeSerialData);

                std::this_thread::sleep_until(start + std::chrono::microseconds(10000));  // 大约10ms周期
                debug_time_count += 0.01;
            }
        }).detach();
    }

    void recalibrateHeadIMU() {
        if (predictor_main_) {
            predictor_main_ -> reset_yaw_integration();
        }

        float reset_command_yaw = last_yaw_rad_delayed_;
        for (int i = 0; i < 60; i++) {
            serial_communication_ -> sendData(0.0, reset_command_yaw, false);
            usleep(30*1000);
        }

        headIMUInfos.to_mcu_delta_yaw = reset_command_yaw - last_yaw_rad_delayed_;
    }

    void headIMUSerialDataCallback(const HeadIMUSerialData& msg) {


        float current_pitch_;
        float current_yaw_;
        float last_pitch_rad_;
        float last_yaw_rad_;
        float total_yaw_rad_;


        current_pitch_ = msg.euler_pitch;
        current_yaw_ = msg.euler_yaw;


        headIMUInfos.head_imu_yaw = msg.euler_yaw;
        headIMUInfos.head_imu_pitch = msg.euler_pitch;
        headIMUInfos.head_imu_roll = msg.euler_roll;
        headIMUInfos.to_mcu_delta_yaw = headIMUInfos.mcu_yaw - headIMUInfos.latest_head_imu_yaw_when_mcu_yaw_update;
        headIMUInfos.to_mcu_delta_pitch = headIMUInfos.mcu_pitch - headIMUInfos.head_imu_pitch;

        while (current_yaw_ < -M_PI) {
            current_yaw_ += 2 * M_PI;
        }
        while (current_yaw_ > M_PI) {
            current_yaw_ -= 2 * M_PI;
        }
        
        if (current_yaw_ < -M_PI/2 && last_yaw_rad_imu_ > M_PI/2) {
            current_yaw_circle_imu_ += 1;
        } else if (current_yaw_ > M_PI/2 && last_yaw_rad_imu_ < -M_PI/2) {
            current_yaw_circle_imu_ -= 1;
        }

        total_yaw_rad_imu_ = current_yaw_circle_imu_ * 2 * M_PI + current_yaw_;
        last_pitch_rad_imu_ = current_pitch_;
        last_yaw_rad_imu_ = current_yaw_;

        if (headIMUInfos.use_head_imu) {
            std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
            DelayInfos now_serial_infos;
            // now_serial_infos.last_pitch_rad_ = last_pitch_rad_imu_;
            now_serial_infos.last_pitch_rad_ = last_pitch_rad_mcu_;
            now_serial_infos.last_pitch_rad_ = last_pitch_rad_imu_;
            now_serial_infos.last_yaw_rad_ = last_yaw_rad_imu_;
            now_serial_infos.total_yaw_rad_ = total_yaw_rad_imu_;
            now_serial_infos.push_time = current_time;
            serial_infos_delay_.push(now_serial_infos);
        }
    }

    void serialDataCallback(const SerialData& msg) {


        float current_pitch_;
        float current_yaw_;


        bullet_velocity_ = msg.bullet_velocity;
        current_pitch_ = ((float)(msg.bullet_angle)) * 30 / 1.8 * M_PI / 180; // 测定pitch轴传入数据1.8大约对应30°
        current_yaw_ = ((float)(msg.gimbal_yaw)) * M_PI / 4096.0;  // 一圈对应[-4096, 4095]


        headIMUInfos.mcu_yaw = current_yaw_;
        headIMUInfos.mcu_pitch = current_pitch_;
        if (headIMUInfos.last_mcu_yaw != headIMUInfos.mcu_yaw) {
            headIMUInfos.latest_head_imu_yaw_when_mcu_yaw_update = headIMUInfos.head_imu_yaw;
            headIMUInfos.last_mcu_yaw = current_yaw_;
            headIMUInfos.last_mcu_yaw_update_time = std::chrono::steady_clock::now();
            headIMUInfos.mcu_yaw_online = true;
            headIMUInfos.latest_mcu_command_yaw_when_mcu_yaw_update = headIMUInfos.last_mcu_command_yaw;
        }
        headIMUInfos.to_mcu_delta_yaw = headIMUInfos.mcu_yaw - headIMUInfos.latest_head_imu_yaw_when_mcu_yaw_update;
        headIMUInfos.to_mcu_delta_pitch = headIMUInfos.mcu_pitch - headIMUInfos.head_imu_pitch;


        while (current_yaw_ < -M_PI) {
            current_yaw_ += 2 * M_PI;
        }
        while (current_yaw_ > M_PI) {
            current_yaw_ -= 2 * M_PI;
        }
        enemy_color_ = (msg.color == 0) ? "RED" : "BLUE";
        if (enemy_color_ == "RED") {
            params_.enemy_color = Params::RED;
        } else if (enemy_color_ == "BLUE") {
            params_.enemy_color = Params::BLUE;
        }
        if (light_detector_) {
            light_detector_->setEnemyColor(msg.color == 0 ? Params::RED : Params::BLUE);
        }
        if (yolo_pose_armor_detector) {
            yolo_pose_armor_detector->setEnemyColor(msg.color == 0 ? Params::RED : Params::BLUE);
        }

        if (current_yaw_ < -M_PI/2 && last_yaw_rad_mcu_ > M_PI/2) {
            current_yaw_circle_mcu_ += 1;
        } else if (current_yaw_ > M_PI/2 && last_yaw_rad_mcu_ < -M_PI/2) {
            current_yaw_circle_mcu_ -= 1;
        }

        total_yaw_rad_mcu_ = current_yaw_circle_mcu_ * 2 * M_PI + current_yaw_;
        last_pitch_rad_mcu_ = current_pitch_;
        last_yaw_rad_mcu_ = current_yaw_;

        RCLCPP_DEBUG(this->get_logger(), 
            "Received serial data: v=%.2f, pitch=%.2f, yaw=%.2f, color=%s \nyaw_circle=%d, total_yaw_rad=%.2f",
            bullet_velocity_, current_pitch_, current_yaw_, enemy_color_.c_str(),
            current_yaw_circle_mcu_, total_yaw_rad_mcu_);


        if (!headIMUInfos.use_head_imu) {
            std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
            DelayInfos now_serial_infos;
            now_serial_infos.last_pitch_rad_ = last_pitch_rad_mcu_;
            now_serial_infos.last_yaw_rad_ = last_yaw_rad_mcu_;
            now_serial_infos.total_yaw_rad_ = total_yaw_rad_mcu_;
            now_serial_infos.push_time = current_time;
            serial_infos_delay_.push(now_serial_infos);
        }
    }

    void drawResults(cv::Mat& image, 
                     const std::vector<Light>& lights,
                     const std::vector<Armor>& armors,
                     const std::vector<ArmorResult>& classifyResults) {
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
        cv::Point3f test_point_pos = rest_frame_ -> worldToPnpP3f({0, 1000, 0});
        cv::Point2f test_point_pos_pixel = armor_solver_ -> project3DToPixel(test_point_pos);
        cv::circle(result, test_point_pos_pixel, 8, cv::Scalar(255, 0, 255), 2);

        // // 1. 绘制灯条（绿色）
        // for (const auto& light : lights) {
        //     cv::Point2f vertices[4];
        //     light.el.points(vertices);
        //     for (int i = 0; i < 4; i++) {
        //         cv::line(result, vertices[i], vertices[(i + 1) % 4], 
        //                 cv::Scalar(0, 255, 0), 2);
        //     }
        // }

        // // 2. 绘制装甲板候选区域（黄色）
        // for (const auto& armor : armors) {
        //     for (size_t i = 0; i < armor.corners.size() && i < 4; i++) {
        //         cv::line(result, armor.corners[i], 
        //                 armor.corners[(i+1)%4], 
        //                 cv::Scalar(0, 255, 255), 2);
        //     }

        //     // 显示装甲板置信度
        //     if (!armor.corners.empty()) {
        //         std::string conf_str = cv::format("conf: %.2f", armor.confidence);
        //         cv::Point text_pos(armor.corners[0].x, armor.corners[0].y - 10);
        //         cv::putText(result, conf_str, text_pos,
        //                 cv::FONT_HERSHEY_SIMPLEX, 0.5, 
        //                 cv::Scalar(0, 255, 255), 1);
        //     }

        //     // 绘制灯条顶点
        //     for (size_t i = 0; i < armor.light_bar_corners.size() && i < 4; i++) {
        //         cv::line(result, armor.light_bar_corners[i], 
        //                 armor.light_bar_corners[(i+1)%4], 
        //                 cv::Scalar(255, 0, 0), 2);
        //     }
        // }

        // 3. 绘制最终识别结果（红色）和跟踪信息
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

            // 绘制灯条顶点
            for (size_t i = 0; i < res.armor.light_bar_corners.size() && i < 4; i++) {
                cv::line(result, res.armor.light_bar_corners[i], 
                        res.armor.light_bar_corners[(i+1)%4], 
                        cv::Scalar(0, 255, 255), 2);
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
#ifdef SHOW_WINDOWS
        cv::imshow("Armor Detection", result);
        cv::waitKey(1);  // 确保窗口刷新
#endif
    }

    void processImage() {
    

        if (std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - headIMUInfos.last_mcu_yaw_update_time).count() > 3000
        ) {
            if (fabs(headIMUInfos.last_mcu_command_yaw - headIMUInfos.latest_mcu_command_yaw_when_mcu_yaw_update)
                > 5.0 * M_PI / 180.0
            ) {
                headIMUInfos.mcu_yaw_online = false;
            }
        }



        while (serial_infos_delay_.size() > 1 && 
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - serial_infos_delay_.front().push_time).count() > serial_delay_time + extra_info_delay_time_ms) {
            serial_infos_delay_.pop();
        }
        DelayInfos delayed_serial_infos = serial_infos_delay_.front();
        last_pitch_rad_delayed_ = delayed_serial_infos.last_pitch_rad_;
        last_yaw_rad_delayed_ = delayed_serial_infos.last_yaw_rad_;
        total_yaw_rad_delayed_ = delayed_serial_infos.total_yaw_rad_;
        ground_stable_point = cv::Point2f(500+total_yaw_rad_delayed_*yaw_rad_to_x_pixel_ratio, 500+last_pitch_rad_delayed_*pitch_rad_to_y_pixel_ratio);
        rest_frame_ -> updateCamOrientation(last_yaw_rad_delayed_, last_pitch_rad_delayed_, 0);
        rest_frame_ -> updateCamPosition(0, 0, 0); // 预留位置接口
        predictor_main_ -> update_serial_info(bullet_velocity_, last_pitch_rad_delayed_, last_yaw_rad_delayed_, total_yaw_rad_delayed_);
        RCLCPP_DEBUG(this->get_logger(), "ground_stable_point: %.2f %.2f", ground_stable_point.x, ground_stable_point.y);




        
        cv::Mat frame;
#if defined(USE_VIDEO) || defined(USE_IMAGES) || defined(SYNC_CAMERA_FPS)
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
            //cv::resize(frame, frame, cv::Size(768, 512), 0, 0, cv::INTER_LINEAR);

            //cv::flip(frame, frame, -1);  // 翻转图像（上下翻转）

            std::vector<Light> lights;
            std::vector<Armor> armors;
            std::vector<ArmorResult> classifyResults;

            // 检测灯条
            light_detector_->detectLights(frame);
            light_detector_->processLights();
            lights = light_detector_->getLights();
            
            // 检测装甲板
            armors = armor_detector_->detectArmors(lights);

            if (use_yolo_pose) {
                std::vector<Armor> yolo_armors;
                now_history_frame_identifier += 1;
                if (now_history_frame_identifier == history_frame_identifier_loop) {
                    now_history_frame_identifier = 0;
                }
                history_frames.push_back(HistoryFrame({now_history_frame_identifier, frame.clone()}));
                if (history_frames.size() == max_history_frame) {
                    history_frames.pop_front();
                }
                yolo_armors = yolo_pose_armor_detector -> detectArmors(frame, false, now_history_frame_identifier);
                int history_frame_index = history_frames.size() - 1 - yolo_delay_frame;
                if (yolo_armors.size() > 0) {
                    Armor& yolo_armor = yolo_armors[0];
                    for (int delay_frame = 0; delay_frame < history_frames.size(); delay_frame += 1) {
                        history_frame_index = history_frames.size() - 1 - delay_frame;
                        yolo_delay_frame = delay_frame;
                        if (yolo_armor.delayed_result.history_frame_identifier == history_frames[history_frame_index].identifier) {
                            break;
                        }
                    }
                }
                std::vector<Armor> true_yolo_armors;
                for (Armor& yolo_armor : yolo_armors) {
                    if (yolo_armor.is_true_yolo_armor(history_frames[history_frame_index].frame))
                    {
                        true_yolo_armors.push_back(yolo_armor);
                    }
                }
                RCLCPP_INFO(this->get_logger(), "yolo_delay_frame: %d", yolo_delay_frame);
                extra_info_delay_time_ms = fps_counter -> avg_frame_time() * yolo_delay_frame * 1000.0;
                classifyResults = classifier_->classify(history_frames[history_frame_index].frame, true_yolo_armors, ground_stable_point);
            } else {
                extra_info_delay_time_ms = 0.0;
                classifyResults = classifier_->classify(frame, armors, ground_stable_point);
            }

            std::vector<ArmorResult> classifyResults_withSolveArmorResult;
            for (ArmorResult &classify_result : classifyResults) {
                AimResult solve_armor_result = armor_solver_->solveArmor(classify_result, last_pitch_rad_delayed_, last_yaw_rad_delayed_);
                cv::Point3f rest_frame_pos = rest_frame_ -> pnpToWorldP3f(solve_armor_result.position);
                if (rest_frame_pos.z < max_armor_position_height && solve_armor_result.valid) { // 高度高于一定值视为无效
                    classifyResults_withSolveArmorResult.emplace_back(classify_result);
                    classifyResults_withSolveArmorResult.back().solve_armor_result = solve_armor_result;
                }
            }

            bool auto_aim_switch = true;
            PredictorResult predictor_result = predictor_main_ -> step(classifyResults_withSolveArmorResult, frame, 
                                                                       PredictorType::AutoSwitch, ArmorType::Middle, 
                                                                       auto_aim_switch, headIMUInfos.mcu_yaw_online); // Todo
            cv::putText(frame, 
                "aiming "+ArmorType::ArmorTypeStrings[predictor_result.armor_type]+": "+PredictorType::PredictorTypeStrings[predictor_result.predictor_type], 
                cv::Point2f(0, 100), 
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            float mcu_command_pitch = predictor_result.command_pitch;
            float mcu_command_yaw = predictor_result.command_yaw;
            if (headIMUInfos.use_head_imu) {
                mcu_command_pitch = predictor_result.command_pitch; // + headIMUInfos.to_mcu_delta_pitch;
                mcu_command_yaw = predictor_result.command_yaw + headIMUInfos.to_mcu_delta_yaw;
            }
            headIMUInfos.last_mcu_command_yaw = mcu_command_yaw;
            if (predictor_result.reset) {
                // RCLCPP_INFO(this->get_logger(), "send data: yaw[%.2f] pitch[%.2f] fire[%d]", 0.0, 0.0, false);
                serial_communication_->sendData(0.0, 0.0, false);
            } else {
                // RCLCPP_INFO(this->get_logger(), "send data: yaw[%.2f] pitch[%.2f] fire[%d]", predictor_result.command_pitch, predictor_result.command_yaw, predictor_result.fire_flag);
                serial_communication_->sendData(mcu_command_pitch, mcu_command_yaw, predictor_result.fire_flag);
            }
            
            // 显示当前参数状态
            cv::putText(frame, 
                cv::format("V: %.1f m/s, P: %.1f, Y: %.1f", 
                    bullet_velocity_, last_pitch_rad_delayed_, last_yaw_rad_delayed_),
                cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 1);

            drawRestFrame(frame, rest_frame_, armor_solver_);

            drawResults(frame, lights, armors, classifyResults_withSolveArmorResult);

            yaw_visualizer_ -> update(last_yaw_rad_delayed_ + (headIMUInfos.use_head_imu ? headIMUInfos.to_mcu_delta_yaw : 0.0), mcu_command_yaw);
            yaw_visualizer_ -> show();

            //计算帧率
            fps_counter->tick();

            if (std::chrono::steady_clock::now() - last_feed_dog_time >= std::chrono::seconds(3)) {
                watchdog_client -> feed();
                last_feed_dog_time = std::chrono::steady_clock::now();
            } // 正常运行时，每3秒喂一次狗
        }

        // 获取处理帧率
        RCLCPP_INFO(this->get_logger(), "frame rate: %.1f fps\n" , fps_counter->fps());
    }

    // 参数文件
    std::shared_ptr<YAML::Node> config_file_ptr; 

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

    std::shared_ptr<RestFrame> rest_frame_;

    std::chrono::time_point<std::chrono::steady_clock> node_start_time;
    
    float bullet_velocity_;


    float last_pitch_rad_mcu_;
    float last_yaw_rad_mcu_;
    float total_yaw_rad_mcu_;
    int current_yaw_circle_mcu_ = 0;
    
    float last_pitch_rad_imu_;
    float last_yaw_rad_imu_;
    float total_yaw_rad_imu_;
    int current_yaw_circle_imu_ = 0;

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
    std::string enemy_color_;
    Params params_;

    // 帧率计算器
    std::shared_ptr<FrameRateCounter> fps_counter;
#ifdef SAVE_IMG_FREQ
    long long frame_count_ = 0;
#endif
    cv::Point2f ground_stable_point;
    std::shared_ptr<SerialCommunicationClass> serial_communication_;
    float yaw_rad_to_x_pixel_ratio;
    float pitch_rad_to_y_pixel_ratio;

    std::shared_ptr<PredictorMain> predictor_main_;

    std::shared_ptr<YOLOPoseArmorDetector> yolo_pose_armor_detector;
    bool use_yolo_pose;
    int max_history_frame = 10;
    int history_frame_identifier_loop = 30;
    int now_history_frame_identifier = 0;
    struct HistoryFrame {
        int identifier;
        cv::Mat frame;
    };
    std::deque<HistoryFrame> history_frames;
    int yolo_delay_frame = 0;
    float extra_info_delay_time_ms = 0.0;

    float max_armor_position_height;

    std::shared_ptr<WatchdogClient> watchdog_client;
    std::chrono::steady_clock::time_point last_feed_dog_time;

    struct {
        std::shared_ptr<HeadIMUSerialCommunicationClass> headIMU_communication_;
        std::thread headIMU_timer_thread_;

        bool use_head_imu = true;

        float head_imu_yaw;
        float head_imu_pitch;
        float head_imu_roll;

        float mcu_yaw;
        float mcu_pitch;
        
        float last_mcu_yaw;
        float latest_head_imu_yaw_when_mcu_yaw_update;
        std::chrono::steady_clock::time_point last_mcu_yaw_update_time;
        bool mcu_yaw_online = true;
        float last_mcu_command_yaw;
        float latest_mcu_command_yaw_when_mcu_yaw_update;

        float to_mcu_delta_yaw;
        float to_mcu_delta_pitch;

    } headIMUInfos;

    std::shared_ptr<YawVisualizer> yaw_visualizer_;
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
