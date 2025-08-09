// ArmorDetect_Node.cpp
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include "camera/Camera.h"
#include "armor_detector/LightBarDetector.h"
#include "armor_detector/ArmorDetector.h"
#include "armor_detector/ArmorClassifier.h"
#include "armor_detector/ArmorAngleKalman.h"
#include "auto_aim/msg/serial_data.hpp"
#include "auto_aim/msg/gimbal_command.hpp"
#include <chrono>
#include <string>
#include <thread>
#include <armor_detector/BallisticSolver.h>
#include <yaml-cpp/yaml.h>
#include "test_codes/FrameRateCounter.h"
#include "test_codes/UnwarpUtils.h"

// #define USE_VIDEO

// 全局变量定义
cv::Mat g_image;
pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
bool g_bExit = false;

class ArmorDetectNode : public rclcpp::Node {
public:
    ArmorDetectNode() : Node("armor_detect_node") {

        config_file_ptr = std::make_shared<YAML::Node>(YAML::LoadFile("/home/rm1/rm2026/transistor_rm2026_algorithm_visual_ws/src/auto_aim/config.yaml"));

        // 初始化发布者和订阅者
        gimbal_command_pub_ = this->create_publisher<auto_aim::msg::GimbalCommand>(
            "gimbal_command", 10);
            
        serial_data_sub_ = this->create_subscription<auto_aim::msg::SerialData>(
            "serial_data", 10,
            std::bind(&ArmorDetectNode::serialDataCallback, this, std::placeholders::_1));

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

        #ifdef USE_VIDEO
        
        #endif
        #ifndef USE_VIDEO
        // 初始化相机和检测器
        camera_ = std::make_shared<Camera>((*config_file_ptr)["cam_ip"].as<std::string>(), (*config_file_ptr)["pc_ip"].as<std::string>());
        camera_->setExposureTime((*config_file_ptr)["camera_ExposureTime"].as<float>());
        camera_->setGain((*config_file_ptr)["camera_Gain"].as<float>());
        #endif

        if (enemy_color_ == "RED") {
            params_.enemy_color = Params::RED;
        } else if (enemy_color_ == "BLUE") {
            params_.enemy_color = Params::BLUE;
        } else if (enemy_color_ == "GREEN") {
            params_.enemy_color = Params::GREEN;
        } else {
            // 处理错误情况，设置默认值
            enemy_color_ = "RED";
            params_.enemy_color = Params::RED;
        }

        RCLCPP_DEBUG(this->get_logger(), "LibTorch version: %s \n", TORCH_VERSION);

        light_detector_ = std::make_shared<LightBarDetector>(params_, config_file_ptr, this);
        armor_detector_ = std::make_shared<ArmorDetector>(config_file_ptr, this);
        classifier_ = std::make_shared<ArmorClassifier>(config_file_ptr, false, this);
        angle_kalman_ = std::make_shared<ArmorAngleKalman>();

        // 创建定时器
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), // 33
            std::bind(&ArmorDetectNode::processImage, this));

        fps_counter = std::make_shared<FrameRateCounter>(30); // 30帧滑动窗口统计帧率

        RCLCPP_INFO(this->get_logger(), "ArmorDetectNode initialized");
    }

    ~ArmorDetectNode() {
        cv::destroyAllWindows();
        pthread_mutex_destroy(&g_mutex);
        RCLCPP_INFO(this->get_logger(), "ArmorDetectNode destroyed");
    }

private:
    void serialDataCallback(const auto_aim::msg::SerialData::SharedPtr msg) {
        static float last_bullet_velocity = 0.0f;
        static float last_bullet_angle = 0.0f;
        static float last_gimbal_yaw = 0.0f;
        static uint8_t last_color = 0;

        // 检查所有关键数据是否完全相同
        if (msg->bullet_velocity == last_bullet_velocity &&
            msg->bullet_angle == last_bullet_angle &&
            msg->gimbal_yaw == last_gimbal_yaw &&
            msg->color == last_color) {
            return;  // 跳过重复数据
        }

        // 更新上一次的数据
        last_bullet_velocity = msg->bullet_velocity;
        last_bullet_angle = msg->bullet_angle;
        last_gimbal_yaw = msg->gimbal_yaw;
        last_color = msg->color;

        // 原有的处理逻辑
        bullet_velocity_ = msg->bullet_velocity;
        current_pitch_ = msg->bullet_angle;
        current_yaw_ = msg->gimbal_yaw;
        enemy_color_ = (msg->color == 0) ? "RED" : "BLUE";
        
        if (light_detector_) {
            light_detector_->setEnemyColor(msg->color == 0 ? Params::RED : Params::BLUE);
        }

        RCLCPP_DEBUG(this->get_logger(), 
            "Received serial data: v=%.2f, pitch=%.2f, yaw=%.2f, color=%s",
            bullet_velocity_, current_pitch_, current_yaw_, enemy_color_.c_str());
    }

    void drawResults(cv::Mat& image, 
                     const std::vector<Light>& lights,
                     const std::vector<Armor>& armors,
                     const std::vector<ArmorResult>& classifyResults) {
        cv::Mat result = image.clone();

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
            cv::Point text_pos(res.corners[0].x, res.corners[0].y - 10);

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

        pthread_mutex_lock(&g_mutex);
        if (!g_image.empty()) {
            frame = g_image.clone();
        }
        pthread_mutex_unlock(&g_mutex);

        if (!frame.empty()) {

        //    cv::flip(frame, frame, -1);  // 翻转图像（上下翻转）

            std::vector<Light> lights;
            std::vector<Armor> armors;
            std::vector<ArmorResult> classifyResults;

            // 检测灯条
            light_detector_->detectLights({frame});
            light_detector_->processLights();
            lights = light_detector_->getLights();
            
            // 检测装甲板
            armors = armor_detector_->detectArmors(lights);
            has_valid_target_ = false;

            classifyResults = classifier_->classify(frame, armors);

            if (!armors.empty()) {

                // 选择最佳目标（置信度最高）
                auto it = std::max_element(
                    classifyResults.begin(), classifyResults.end(),
                    [](const ArmorResult& a, const ArmorResult& b) {
                        return a.confidence < b.confidence;
                    }
                );
                if (it != classifyResults.end()) {
                    auto best_result = *it;
                    AimResult aim = armor_detector_->solveArmor(best_result);                    
                    if (aim.valid) {
                        if (!angle_kalman_->isInitialized()) {
                            angle_kalman_->reset(aim.position); // 使用当前观测位置初始化
                        }
                        cv::Point3f current_predicted_pos = angle_kalman_->predictKalmanFilter(0);
                        float pos_diff = cv::norm(current_predicted_pos - aim.position);
                        // ========== 新增目标切换检测逻辑 ==========
                        // 情况1：检测到新目标ID
                        if (best_result.number != current_target_id_) {
                            // RCLCPP_INFO(get_logger(), "Target changed: %d -> %d, reset Kalman", 
                                    // current_target_id_, best_result.number);
                            angle_kalman_->reset();
                            current_target_id_ = best_result.number;
                            is_target_lost_ = false;
                        } else if (pos_diff > RESET_DISTANCE_THRESHOLD) {
                            // RCLCPP_WARN(get_logger(), "Position jump: %.2f mm, reset Kalman", pos_diff);
                            angle_kalman_->reset();
                            is_target_lost_ = false;
                        } else if ((this->now() - last_track_time_).seconds() > MAX_LOST_TIME) {
                            // RCLCPP_WARN(get_logger(), "Target lost timeout, reset Kalman");
                            angle_kalman_->reset();
                            current_target_id_ = -1;
                        }
                        angle_kalman_->updateKalmanFilter(aim.position);
                        last_observed_pos_ = aim.position;
                        last_track_time_ = this->now();
                        is_target_lost_ = false;
                        // 计算总延迟（图像处理+通信+弹丸飞行）
                        constexpr float image_latency = 0.043f; // 33ms处理延迟
                        constexpr float comm_latency = 0.040f;  // 10ms通信延迟
                        float bullet_time = abs(aim.position.z) / 1000 / bullet_velocity_;
                        float total_delay = image_latency + comm_latency + bullet_time;
                        
                        // 预测未来位置
                        cv::Point3f predicted_pos = angle_kalman_->predictKalmanFilter(total_delay);
                        
                        // 弹道解算
                        BallisticInfo ballistic_result = calcBallisticAngle(
                            predicted_pos.x, 
                            predicted_pos.y, 
                            predicted_pos.z,
                            delta_x_,
                            delta_y_,
                            delta_z_,
                            bullet_velocity_,
                            current_pitch_,
                            current_yaw_
                        );
                        
                        if (ballistic_result.valid) {
                            // RCLCPP_INFO(this->get_logger(), "Target detected, publishing command");
                            has_valid_target_ = true;
                            latest_pitch_angle_ = ballistic_result.pitch_angle;
                            latest_yaw_angle_ = ballistic_result.yaw_angle;
                            
                            // 发布云台控制命令
                            auto command_msg = auto_aim::msg::GimbalCommand();
                            command_msg.pitch = latest_pitch_angle_;
                            command_msg.yaw = latest_yaw_angle_;
                            gimbal_command_pub_->publish(command_msg);

                            RCLCPP_INFO(this->get_logger(),
                                "Target %d: Position[%.2f, %.2f, %.2f] mm, "
                                "Command[pitch: %.2f, yaw: %.2f] deg",
                                best_result.number,
                                predicted_pos.x, predicted_pos.y, predicted_pos.z,
                                latest_pitch_angle_, latest_yaw_angle_);
                            
                                // 绘制预测点（黄色）
                                cv::Point2f pred_pixel = armor_detector_->project3DToPixel(predicted_pos);
                                cv::circle(frame, pred_pixel, 8, cv::Scalar(0, 255, 255), 2);
                        }
                    } 
                    
                }
            }

            drawResults(frame, lights, armors, classifyResults);

            //计算帧率
            fps_counter->tick();
            
            // // 显示当前参数状态
            // cv::putText(frame, 
            //     cv::format("V: %.1f m/s, P: %.1f, Y: %.1f", 
            //         bullet_velocity_, current_pitch_, current_yaw_),
            //     cv::Point(10, 60),
            //     cv::FONT_HERSHEY_SIMPLEX, 0.5,
            //     cv::Scalar(0, 255, 0), 1);
        }        

        // 获取处理帧率
        RCLCPP_DEBUG(this->get_logger(), "frame rate: %.1f fps\n" , fps_counter->fps());
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
    rclcpp::Publisher<auto_aim::msg::GimbalCommand>::SharedPtr gimbal_command_pub_;
    rclcpp::Subscription<auto_aim::msg::SerialData>::SharedPtr serial_data_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::shared_ptr<Camera> camera_;
    std::shared_ptr<LightBarDetector> light_detector_;
    std::shared_ptr<ArmorDetector> armor_detector_;
    std::shared_ptr<ArmorClassifier> classifier_;
    std::shared_ptr<ArmorAngleKalman> angle_kalman_;
    
    float bullet_velocity_;
    float current_pitch_;
    float current_yaw_;
    float delta_x_;
    float delta_y_;
    float delta_z_;
    float latest_pitch_angle_;
    float latest_yaw_angle_;
    bool has_valid_target_;
    std::string enemy_color_;
    Params params_;

    // 帧率计算器
    std::shared_ptr<FrameRateCounter> fps_counter;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorDetectNode>());
    rclcpp::shutdown();
    return 0;
}
