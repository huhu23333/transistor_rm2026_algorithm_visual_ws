#include "predictor/AllPredictor.h"

void AllPredictor::update_serial_info(float bullet_velocity, float last_pitch_rad_delayed, float last_yaw_rad_delayed, float total_yaw_rad_delayed) {
    bullet_velocity_ = bullet_velocity;
    last_pitch_rad_delayed_ = last_pitch_rad_delayed;
    last_yaw_rad_delayed_ = last_yaw_rad_delayed;
    total_yaw_rad_delayed_ = total_yaw_rad_delayed;
}

PredictorResult AllPredictor::step(std::vector<ArmorResult>& classifyResults, cv::Mat& frame, PredictorType::PredictorType control_predictor_type)
{
    PredictorResult result;
    total_yaw_rad_delayed_filter_ -> addPoint(total_yaw_rad_delayed_);
    
    bool pnp_valid_flag = false;
    bool ballistic_valid_flag = false;
    if (!classifyResults.empty()) {
        // 选择最佳目标（置信度最高）
        auto it = std::max_element(
            classifyResults.begin(), classifyResults.end(),
            [](const ArmorResult& a, const ArmorResult& b) {
                if (a.is_tracked_now && !b.is_tracked_now) {
                    return false;
                }
                if (!a.is_tracked_now && b.is_tracked_now) {
                    return true;
                }
                return a.confidence < b.confidence;
            }
        );
        if (it != classifyResults.end()) {
            auto best_result = *it;
            AimResult aim = armor_solver_->solveArmor(best_result, last_pitch_rad_delayed_, last_yaw_rad_delayed_);
            if (aim.valid) {
                pnp_valid_flag = true;

                is_reset = false;
                last_com_time = std::chrono::steady_clock::now();

                last_pixel_horizontal_center_distance = std::abs(best_result.center.x - static_cast<float>(frame.cols)/2.0);
                
                // 查看并滤波z轴距离轴数据
                armor_distance_filter_ -> addPoint(aim.position.z);
                aim.position.z = armor_distance_filter_ -> getExponentialValue();
                oscilloscope_common_ -> addDataPoint(aim.position.z / 10000);

                // 将pnp结果转换至静止坐标系以稳定预测
                cv::Point3f rest_frame_pos = rest_frame_ -> pnpToWorldP3f(aim.position);
                // std::vector<float> rest_frame_euler_angles = rest_frame_ -> getWorldEulerAnglesFromCam(
                //     aim.normal_euler_angles[0], aim.normal_euler_angles[1], aim.normal_euler_angles[2]
                // );
                std::vector<float> rest_frame_euler_angles = {
                    static_cast<float>(aim.ba_global_ypr[0]),
                    static_cast<float>(aim.ba_global_ypr[1]),
                    static_cast<float>(aim.ba_global_ypr[2])
                };
                RCLCPP_DEBUG(node->get_logger(), "camera euler angles: yaw=%.2f, pitch=%.2f, roll=%.2f", aim.normal_euler_angles[0], aim.normal_euler_angles[1], aim.normal_euler_angles[2]);
                RCLCPP_DEBUG(node->get_logger(), "Rest frame pos: x=%.2f, y=%.2f, z=%.2f, yaw=%.2f", rest_frame_pos.x, rest_frame_pos.y, rest_frame_pos.z, rest_frame_euler_angles[0]);

                last_rest_frame_pos = rest_frame_pos;

                // 提前预测与弹道解算
                constexpr float image_latency = 0.013f;
                constexpr float comm_latency  = 0.010f;
                float bullet_time = (bullet_velocity_ > 1.0f) ? (std::abs(aim.position.z) / 1000.0f / bullet_velocity_) : 0.0f;
                float extra_time = 0.500f; // 0.300f
                float total_delay = image_latency + comm_latency + bullet_time + extra_time;
                last_total_delay_ = total_delay;

                // 默认使用 None （直接瞄准装甲板）
                cv::Point3f predicted_armor_pos = rest_frame_pos;
                cv::Point3f predicted_aim_pos = predicted_armor_pos;
                bool fire_flag = true;

                PBEKF_ObservedData observed_data = {
                    fps_counter -> avg_frame_time(),
                    rest_frame_pos.x,
                    rest_frame_pos.y,
                    rest_frame_pos.z,
                    rest_frame_euler_angles[0]
                };
                if (!PBEKFTracker) {
                    PBEKFTracker = std::make_shared<PBEKF_EKFTracker>(observed_data);
                } else {
                    if (best_result.is_tracked_now) {
                        PBEKFTracker -> update(observed_data);
                    } else {
                        std::array<double, 4> predicted_result = PBEKFTracker -> predict(fps_counter -> avg_frame_time());
                        PBEKF_ObservedData predicted_observed_data = {
                            fps_counter -> avg_frame_time(),
                            predicted_result[0],
                            predicted_result[1],
                            predicted_result[2],
                            predicted_result[3]
                        };
                        PBEKFTracker -> update(predicted_observed_data);
                    }
                }

                std::array<double, 4> predicted_result = PBEKFTracker -> predict(total_delay);
                predicted_armor_pos = {
                    predicted_result[0],
                    predicted_result[1],
                    predicted_result[2]
                };

                cv::Mat PBEKF_visualize_frame = cv::Mat::zeros(800, 800, CV_8UC3);
                cv::circle(PBEKF_visualize_frame, cv::Point2f(400+observed_data.x/10, 400-observed_data.y/10), 8, cv::Scalar(255, 255, 0), 2);
                cv::line(PBEKF_visualize_frame, 
                    cv::Point2f(400 + observed_data.x/10, 400-observed_data.y/10), 
                    cv::Point2f(400 + observed_data.x/10 + std::sin(observed_data.yaw)*50, 
                                400 - (observed_data.y/10 - std::cos(observed_data.yaw)*50)),
                    cv::Scalar(255, 255, 0), 2);
                std::vector<double> PBEKFStateForVisualization = PBEKFTracker -> getStateForVisualization();
                float xc = PBEKFStateForVisualization[0];
                float yc = PBEKFStateForVisualization[2];
                float zc = PBEKFStateForVisualization[4];
                float yaw_now = PBEKFStateForVisualization[5];
                float vyaw = PBEKFStateForVisualization[6];
                float r = PBEKFStateForVisualization[7];
                cv::circle(PBEKF_visualize_frame, cv::Point2f(400+xc/10, 400-yc/10), 10, cv::Scalar(0, 255, 0), 3);
                cv::line(PBEKF_visualize_frame, 
                    cv::Point2f(400 + xc/10, 400-yc/10), 
                    cv::Point2f(400 + xc/10 + std::sin(yaw_now)*50, 
                                400 - (yc/10 - std::cos(yaw_now)*50)),
                    cv::Scalar(0, 255, 0), 2);
                for (int i = 0; i < 3; i++) {
                    float yaw = PBEKFStateForVisualization[8+i];
                    float xa = xc + r * std::sin(yaw);
                    float ya = yc - r * std::cos(yaw);
                    cv::Point2f PBEKF_pixel = armor_solver_->project3DToPixel(rest_frame_ -> worldToPnpP3f(cv::Point3f(xa, ya, zc)));
                    cv::circle(frame, PBEKF_pixel, 8, cv::Scalar(0, 255, 0), 2);
                    cv::circle(PBEKF_visualize_frame, cv::Point2f(400+xa/10, 400-ya/10), 8, cv::Scalar(0, 255, 0), 2);
                }
                cv::putText(PBEKF_visualize_frame, 
                    "vyaw:"+std::to_string(vyaw), 
                    cv::Point2f(20,20), 
                    cv::FONT_HERSHEY_COMPLEX, 0.7, 
                    cv::Scalar(0, 255, 0), 1, 8, false);
                cv::putText(PBEKF_visualize_frame, 
                    "r:"+std::to_string(r), 
                    cv::Point2f(20,50), 
                    cv::FONT_HERSHEY_COMPLEX, 0.7, 
                    cv::Scalar(0, 255, 0), 1, 8, false);
#ifdef SHOW_WINDOWS
                cv::imshow("PBEKF visualize", PBEKF_visualize_frame);
#endif

                // 统一转换回pnp相机坐标系    
                predicted_aim_pos = rest_frame_ -> worldToPnpP3f(predicted_aim_pos);
                predicted_armor_pos = rest_frame_ -> worldToPnpP3f(predicted_armor_pos);

                // 弹道解算
                RCLCPP_DEBUG(node->get_logger(), "aim pos: (%.2f, %.2f, %.2f)",
                            predicted_aim_pos.x, predicted_aim_pos.y, predicted_aim_pos.z);
                BallisticInfo ballistic_result = ballistic_solver_ -> calcBallisticAngle(
                    predicted_aim_pos.x, 
                    predicted_aim_pos.y, 
                    predicted_aim_pos.z,
                    bullet_velocity_,
                    last_pitch_rad_delayed_,//pitch_integration | last_pitch_rad_delayed_ #todo
                    last_yaw_rad_delayed_
                );
                
                if (ballistic_result.valid) {
                    ballistic_valid_flag = true;
                    has_valid_ballistic = true;
                    // RCLCPP_INFO(node->get_logger(), "Target detected, publishing command");
                    // has_valid_target_ = true;

                    pitch_integration += ballistic_result.delta_pitch_rad * 0.1;
                    yaw_integration += ballistic_result.delta_yaw_rad * 0.1;

                    if (pitch_integration > 10.0 * M_PI / 180.0) {
                        pitch_integration = 10.0 * M_PI / 180.0;
                    }
                    if (pitch_integration < -10.0 * M_PI / 180.0) {
                        pitch_integration = -10.0 * M_PI / 180.0;
                    }

                    if (yaw_integration > 20.0 * M_PI / 180.0) {
                        yaw_integration = 20.0 * M_PI / 180.0;
                    }
                    if (yaw_integration < -20.0 * M_PI / 180.0) {
                        yaw_integration = -20.0 * M_PI / 180.0;
                    }
                    
                    // 发布云台控制命令
                    float command_pitch = last_pitch_rad_delayed_ + ballistic_result.delta_pitch_rad * 0.8 + pitch_integration; // PI控制
                    float command_yaw = last_yaw_rad_delayed_ + ballistic_result.delta_yaw_rad + yaw_integration; // 缓解yaw轴输入数据掉线问题
                    last_command_pitch_ = command_pitch;
                    last_command_yaw_ = command_yaw;
                    //serial_communication_->sendData(command_pitch, command_yaw, fire_flag);
                    result.reset = false;
                    result.command_pitch = command_pitch;
                    result.command_yaw = command_yaw;
                    result.fire_flag = fire_flag;

                    RCLCPP_DEBUG(node->get_logger(),
                        "Target %d: Position[%.2f, %.2f, %.2f] mm, "
                        "Command[pitch: %.2f, yaw: %.2f] rad",
                        static_cast<int>(armor_class),
                        predicted_aim_pos.x, predicted_aim_pos.y, predicted_aim_pos.z,
                        command_pitch, command_yaw);
                    
                    // 绘制瞄准预测点（黄色）
                    cv::Point2f pred_aim_pixel = armor_solver_->project3DToPixel(predicted_aim_pos);
                    cv::circle(frame, pred_aim_pixel, 8, cv::Scalar(0, 255, 255), 2);

                    // 计算并绘制瞄准时目标画面中心（天蓝色：未开火 | 红色：开火）
                    cv::Point2f aim_yaw_pitch = cv::Point2f(last_yaw_rad_delayed_ + ballistic_result.delta_yaw_rad, last_pitch_rad_delayed_ + ballistic_result.delta_pitch_rad);
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
                    RCLCPP_DEBUG(node->get_logger(), "aim center yaw pitch: (%.2f, %.2f)",
                            aim_yaw_pitch.x, aim_yaw_pitch.y);
//#ifdef USE_PREDICTOR3D
                    cv::Point2f pred_armor_pixel = armor_solver_->project3DToPixel(predicted_armor_pos);
                    // 绘制装甲板预测点（天蓝色）
                    cv::circle(frame, pred_armor_pixel, 8, cv::Scalar(255, 255, 0), 2);
                    oscilloscope_fire_ -> addDataPoint(fire_flag);
                    //oscilloscope_fire_ -> addDataPoint(fire_data_predictor_ -> smooth(0));
                    //oscilloscope_fire_ -> addDataPoint(fire_data_predictor_ -> isRising(0));

                    // 绘制不同时间预测开火波形
                    for (size_t debug_pred_fire_index = 1; debug_pred_fire_index < 6; debug_pred_fire_index += 1) {
                        oscilloscope_common_ -> addDataPoint(
                            ((float)(fire_data_predictor_ -> smooth(debug_pred_fire_index*5)))/11.0 + 
                            (float)(debug_pred_fire_index - 1) / 10,
                            debug_pred_fire_index);
                    }
//#endif
                }
                using_predictor_type = PredictorType::EKF;
            }
        }
    }

    if (!pnp_valid_flag) {
        if (EKF_tracker_->state != Tracker::LOST) {
            EKF_tracker_->predict();
        }

        if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_com_time).count() >= reset_com_time) {
            result.reset = true;
            pitch_integration = 0.0; // 积分项重置
            yaw_integration = 0.0;
            predictor3d -> clearHistory(); 
            fire_data_predictor_ -> clearHistory();
            pred_fire_data_filter_ -> clearHistory();
            armor_distance_filter_ -> clearHistory();
            rotation_motion_model_.reset();
            is_reset = true;
            predictor_switcher_ -> clearHistory();
            last_pixel_horizontal_center_distance = 1e10;
            has_valid_ballistic = false;

            PBEKFTracker.reset();
        } else {
            
            result.reset = false;
            result.command_pitch = last_command_pitch_;
            result.command_yaw = last_command_yaw_;
            result.fire_flag = false;

            bool fire_flag = false;

            if (result.fire_flag) {
                cv::circle(frame, last_aim_yaw_pitch_pixel_, 8, cv::Scalar(0, 0, 255), 2);
            } else {
                cv::circle(frame, last_aim_yaw_pitch_pixel_, 8, cv::Scalar(255, 255, 0), 2);
            }

            if (PBEKFTracker) {
                std::array<double, 4> predicted_result = PBEKFTracker -> predict(fps_counter -> avg_frame_time());
                PBEKF_ObservedData predicted_observed_data = {
                    fps_counter -> avg_frame_time(),
                    predicted_result[0],
                    predicted_result[1],
                    predicted_result[2],
                    predicted_result[3]
                };
                PBEKFTracker -> update(predicted_observed_data);

                cv::Mat PBEKF_visualize_frame = cv::Mat::zeros(800, 800, CV_8UC3);
                std::vector<double> PBEKFStateForVisualization = PBEKFTracker -> getStateForVisualization();
                float xc = PBEKFStateForVisualization[0];
                float yc = PBEKFStateForVisualization[2];
                float zc = PBEKFStateForVisualization[4];
                float yaw_now = PBEKFStateForVisualization[5];
                float vyaw = PBEKFStateForVisualization[6];
                float r = PBEKFStateForVisualization[7];
                cv::circle(PBEKF_visualize_frame, cv::Point2f(400+xc/10, 400-yc/10), 10, cv::Scalar(0, 255, 0), 3);
                cv::line(PBEKF_visualize_frame, 
                    cv::Point2f(400 + xc/10, 400-yc/10), 
                    cv::Point2f(400 + xc/10 + std::sin(yaw_now)*50, 
                                400 - (yc/10 - std::cos(yaw_now)*50)),
                    cv::Scalar(0, 255, 0), 2);
                for (int i = 0; i < 3; i++) {
                    float yaw = PBEKFStateForVisualization[8+i];
                    float xa = xc + r * std::sin(yaw);
                    float ya = yc - r * std::cos(yaw);
                    cv::Point2f PBEKF_pixel = armor_solver_->project3DToPixel(rest_frame_ -> worldToPnpP3f(cv::Point3f(xa, ya, zc)));
                    cv::circle(frame, PBEKF_pixel, 8, cv::Scalar(0, 255, 0), 2);
                    cv::circle(PBEKF_visualize_frame, cv::Point2f(400+xa/10, 400-ya/10), 8, cv::Scalar(0, 255, 0), 2);
                }
                cv::putText(PBEKF_visualize_frame, 
                    "vyaw:"+std::to_string(vyaw), 
                    cv::Point2f(20,20), 
                    cv::FONT_HERSHEY_COMPLEX, 0.7, 
                    cv::Scalar(0, 255, 0), 1, 8, false);
                cv::putText(PBEKF_visualize_frame, 
                    "r:"+std::to_string(r), 
                    cv::Point2f(20,50), 
                    cv::FONT_HERSHEY_COMPLEX, 0.7, 
                    cv::Scalar(0, 255, 0), 1, 8, false);
#ifdef SHOW_WINDOWS
                cv::imshow("PBEKF visualize", PBEKF_visualize_frame);
#endif
            }
        }
        oscilloscope_fire_ -> addDataPoint(result.fire_flag);

        using_predictor_type = PredictorType::EKF;

    } else if ((!ballistic_valid_flag) && has_valid_ballistic) {
        result.reset = false;
        result.command_pitch = last_command_pitch_;
        result.command_yaw = last_command_yaw_;
        result.fire_flag = false;
    }

//#ifdef USE_PREDICTOR3D
    oscilloscope_fire_ -> update();
    oscilloscope_fire_ -> putText("period_3d:"+std::to_string(predictor3d->getFourierPeriod()), cv::Point2f(240, 20), cv::Scalar(0, 255, 0), 0.7);
    oscilloscope_fire_ -> putText("period_fire:"+std::to_string(fire_data_predictor_->getPeriod()), cv::Point2f(440, 20), cv::Scalar(0, 255, 0), 0.7);
    oscilloscope_fire_ -> show();
//#endif
    oscilloscope_common_ -> update();
    oscilloscope_common_ -> show();

    std::string using_predictor_type_string = PredictorType::PredictorTypeStrings[using_predictor_type];
    cv::putText(frame, 
        "Class "+std::to_string(armor_class)+": "+using_predictor_type_string, 
        cv::Point2f(frame.cols - 200, 50 + 30 * armor_class), 
        cv::FONT_HERSHEY_COMPLEX, 0.7, 
        cv::Scalar(0, 255, 0), 1, 8, false);
    result.predictor_type = using_predictor_type;
    result.armor_type = armor_class;
    result.pixel_horizontal_center_distance = last_pixel_horizontal_center_distance;
    return result;
}