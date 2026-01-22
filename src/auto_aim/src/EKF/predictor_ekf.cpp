#include "EKF/predict_ekf.hpp"

void PredictorEKF::update_serial_info(float bullet_velocity, float last_pitch_rad_delayed, float last_yaw_rad_delayed, float total_yaw_rad_delayed) {
    bullet_velocity_ = bullet_velocity;
    last_pitch_rad_delayed_ = last_pitch_rad_delayed;
    last_yaw_rad_delayed_ = last_yaw_rad_delayed;
    total_yaw_rad_delayed_ = total_yaw_rad_delayed;
}

PredictorResult PredictorEKF::step(std::vector<ArmorResult>& classifyResults, cv::Mat& frame, PredictorType::PredictorType control_predictor_type)
{
    PredictorResult result = {};

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
            AimResult aim = best_result.solve_armor_result;// armor_solver_->solveArmor(best_result, last_pitch_rad_delayed_, last_yaw_rad_delayed_);

            is_reset = false;
            last_com_time_ = std::chrono::steady_clock::now();

            last_pixel_horizontal_center_distance_ = std::abs(best_result.center.x - static_cast<float>(frame.cols)/2.0);
            
            // 查看并滤波z轴距离轴数据
            armor_distance_filter_ -> addPoint(aim.position.z);
            aim.position.z = armor_distance_filter_ -> getExponentialValue();
            oscilloscope_common_ -> addDataPoint(aim.position.z / 10000, 0);

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
            RCLCPP_DEBUG(node_->get_logger(), "camera euler angles: yaw=%.2f, pitch=%.2f, roll=%.2f", aim.normal_euler_angles[0], aim.normal_euler_angles[1], aim.normal_euler_angles[2]);
            RCLCPP_DEBUG(node_->get_logger(), "Rest frame pos: x=%.2f, y=%.2f, z=%.2f, yaw=%.2f", rest_frame_pos.x, rest_frame_pos.y, rest_frame_pos.z, rest_frame_euler_angles[0]);

            last_rest_frame_pos_ = rest_frame_pos;

            // 计算延迟并存至成员变量，用于弹道解算
            constexpr float image_latency = 0.013f;
            constexpr float comm_latency  = 0.010f;
            float bullet_time = (bullet_velocity_ > 1.0f) ? (std::abs(aim.position.z) / 1000.0f / bullet_velocity_) : 0.0f;
            float extra_time = 0.300f; // 0.300f
            float total_delay = image_latency + comm_latency + bullet_time + extra_time;
            last_total_delay_ = total_delay;

            cv::Point3f predicted_armor_pos = rest_frame_pos;
            cv::Point3f predicted_aim_pos = predicted_armor_pos;
            bool fire_flag = true;
            
            // EKF预测
            std::vector<float> cam_position = rest_frame_ -> getCamPosition();
            cv::Point3f bullet_nearest_point = ballistic_solver_ -> calcNearestPointWithAirResistance( // todo
                rest_frame_pos / 1000, {cam_position[0], cam_position[1], cam_position[2]}, last_aim_yaw_pitch_, bullet_velocity_) * 1000;
            cv::Point3f pnp_bullet_nearest_point = rest_frame_ -> worldToPnpP3f(bullet_nearest_point);
            cv::Point2f bullet_nearest_point_pixel = armor_solver_->project3DToPixel(pnp_bullet_nearest_point);
            cv::circle(frame, bullet_nearest_point_pixel, 15, cv::Scalar(255, 0, 255), 2);
            RCLCPP_DEBUG(node_->get_logger(), "bullet_nearest_point: (%.2f, %.2f, %.2f)",
                        bullet_nearest_point.x, bullet_nearest_point.y, bullet_nearest_point.z);

            double EKF_update_time = (std::chrono::steady_clock::now() - node_start_time_).count() / 1e9;
            Tracker::Measurement z = {rest_frame_pos.x, rest_frame_pos.y, rest_frame_pos.z, rest_frame_euler_angles[0]};

            if (ekf_init == false) {
                tracker_ -> reset(z);

                RCLCPP_INFO(node_->get_logger(), "EKF initialized.");

                if (is_outpost) {
                    n_armors_ = 3;
                    r_now_ = 276.5;
                    r_another_ = 276.5;
                } else {
                    n_armors_ = 4;
                    r_now_ = tracker_ -> r_init_;
                    r_another_ = tracker_ -> r_init_;
                }
                jump_rad_ = M_PI * 2.0 / n_armors_;

                ekf_init = true;
                return result;
            }

            Tracker::State state = tracker_ -> getTargetState();
            PredictResult EKF_pred_now_data = state_convert_to_PredictResult(state);

            tracker_ -> predict(); tracker_ -> update(z);
  
            // 可视化EKF预测结果
            cv::Point3f EKF_pred_now_center_p3f = rest_frame_ -> worldToPnpP3f({
                static_cast<float>(EKF_pred_now_data.center_x), 
                static_cast<float>(EKF_pred_now_data.center_y), 
                static_cast<float>(EKF_pred_now_data.center_z)
            });
            cv::Point2f EKF_pred_now_center_pixel = armor_solver_->project3DToPixel(EKF_pred_now_center_p3f);
            if (best_result.is_tracked_now) {
                cv::circle(frame, EKF_pred_now_center_pixel, 10, cv::Scalar(0, 255, 0), 2);
            } else {
                cv::circle(frame, EKF_pred_now_center_pixel, 10, cv::Scalar(255, 0, 255), 2);
            }
            cv::Mat EKF_visualize_frame = cv::Mat::zeros(800, 800, CV_8UC3);
            if (best_result.is_tracked_now) {
                cv::circle(EKF_visualize_frame, cv::Point2f(400+EKF_pred_now_data.center_x/10, 400-EKF_pred_now_data.center_y/10), 8, cv::Scalar(0, 255, 0), 2);
            } else {
                cv::circle(EKF_visualize_frame, cv::Point2f(400+EKF_pred_now_data.center_x/10, 400-EKF_pred_now_data.center_y/10), 8, cv::Scalar(255, 0, 255), 2);
            }
            for (int EKF_pred_now_armor_i = 0; EKF_pred_now_armor_i < EKF_pred_now_data.armors.size(); EKF_pred_now_armor_i += 1) {
                SimpleArmor& EKF_pred_now_armor = EKF_pred_now_data.armors[EKF_pred_now_armor_i];
                cv::Point3f EKF_pred_now_armor_p3f = rest_frame_ -> worldToPnpP3f({
                    static_cast<float>(EKF_pred_now_armor.x), 
                    static_cast<float>(EKF_pred_now_armor.y), 
                    static_cast<float>(EKF_pred_now_armor.z)
                });
                cv::Point2f EKF_pred_now_armor_pixel = armor_solver_->project3DToPixel(EKF_pred_now_armor_p3f);
                cv::circle(frame, EKF_pred_now_armor_pixel, 6, cv::Scalar(0, 255, 0), 2);
                // cv::line(frame, EKF_pred_now_center_pixel, EKF_pred_now_armor_pixel, cv::Scalar(0, 255, 0), 2);
                
                cv::circle(EKF_visualize_frame, cv::Point2f(400+EKF_pred_now_armor.x/10, 400-EKF_pred_now_armor.y/10), 8, 
                    cv::Scalar(0, 255 - EKF_pred_now_armor_i * 80, EKF_pred_now_armor_i * 80), 2);
                // cv::line(EKF_visualize_frame, 
                //     cv::Point2f(400+EKF_pred_now_data.center_x/10, 400-EKF_pred_now_data.center_y/10), 
                //     cv::Point2f(400+EKF_pred_now_armor.x/10, 400-EKF_pred_now_armor.y/10), 
                //     cv::Scalar(0, 255 - EKF_pred_now_armor_i * 80, EKF_pred_now_armor_i * 80), 2);
            }
            cv::circle(EKF_visualize_frame, cv::Point2f(400+rest_frame_pos.x/10, 400-rest_frame_pos.y/10), 8, cv::Scalar(255, 255, 0), 2);
            cv::line(EKF_visualize_frame, 
                cv::Point2f(400 + rest_frame_pos.x/10, 400-rest_frame_pos.y/10), 
                cv::Point2f(400 + rest_frame_pos.x/10 + std::sin(rest_frame_euler_angles[0])*50, 
                            400 - (rest_frame_pos.y/10 - std::cos(rest_frame_euler_angles[0])*50)),
                cv::Scalar(255, 255, 0), 2);

            cv::line(EKF_visualize_frame, 
                cv::Point2f(400, 400), 
                cv::Point2f(400 - std::sin(total_yaw_rad_delayed_)*150, 
                            400 - std::cos(total_yaw_rad_delayed_)*150),
                cv::Scalar(255, 255, 0), 2);
            cv::putText(EKF_visualize_frame, 
                "total_yaw:"+std::to_string(total_yaw_rad_delayed_), 
                cv::Point2f(20,140), 
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(EKF_visualize_frame, 
                "vx:"+std::to_string(EKF_state.center_vx), 
                cv::Point2f(20,200), 
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(EKF_visualize_frame, 
                "vy:"+std::to_string(EKF_state.center_vy), 
                cv::Point2f(20,230), 
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::line(EKF_visualize_frame, 
                cv::Point2f(400 + EKF_pred_now_data.center_x/10, 400 - EKF_pred_now_data.center_y/10), 
                cv::Point2f(400 + (EKF_pred_now_data.center_x/10 + EKF_state.center_vx/5), 
                            400 - (EKF_pred_now_data.center_y/10 + EKF_state.center_vy/5)),
                cv::Scalar(255, 255, 0), 2);
#ifdef SHOW_WINDOWS
            cv::imshow("EKF visualize "+std::to_string(armor_class_), EKF_visualize_frame);
#endif

            // 统一转换回pnp相机坐标系    
            predicted_aim_pos = rest_frame_ -> worldToPnpP3f(predicted_aim_pos);
            predicted_armor_pos = rest_frame_ -> worldToPnpP3f(predicted_armor_pos);

            // 弹道解算
            RCLCPP_DEBUG(node_->get_logger(), "aim pos: (%.2f, %.2f, %.2f)",
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
                has_valid_ballistic_ = true;
                // RCLCPP_INFO(node_->get_logger(), "Target detected, publishing command");
                // has_valid_target_ = true;
                
                // 发布云台控制命令
                //serial_communication_->sendData(command_pitch, command_yaw, fire_flag);
                result.reset = false;
                result.command_delta_pitch = ballistic_result.delta_pitch_rad;
                result.command_delta_yaw = ballistic_result.delta_yaw_rad;
                result.fire_flag = fire_flag;
                
                // 绘制瞄准预测点（黄色）
                cv::Point2f pred_aim_pixel = armor_solver_->project3DToPixel(predicted_aim_pos);
                cv::circle(frame, pred_aim_pixel, 8, cv::Scalar(0, 255, 255), 2);

                // 计算并绘制瞄准时目标画面中心（天蓝色：未开火 | 红色：开火）
                cv::Point2f aim_yaw_pitch = cv::Point2f(last_yaw_rad_delayed_ + ballistic_result.delta_yaw_rad, last_pitch_rad_delayed_ + ballistic_result.delta_pitch_rad);
                cv::Point2f aim_yaw_pitch_pixel = cv::Point2f(
                    frame.cols / 2 - (aim_yaw_pitch.x - last_yaw_rad_delayed_) * yaw_rad_to_x_pixel_ratio_, 
                    frame.rows / 2 - (aim_yaw_pitch.y - last_pitch_rad_delayed_) * pitch_rad_to_y_pixel_ratio_);
                last_aim_yaw_pitch_ = aim_yaw_pitch;
                last_aim_yaw_pitch_pixel_ = aim_yaw_pitch_pixel;
                if (fire_flag) {
                    cv::circle(frame, aim_yaw_pitch_pixel, 8, cv::Scalar(0, 0, 255), 2);
                } else {
                    cv::circle(frame, aim_yaw_pitch_pixel, 8, cv::Scalar(255, 255, 0), 2);
                }
                RCLCPP_DEBUG(node_->get_logger(), "aim center yaw pitch: (%.2f, %.2f)",
                        aim_yaw_pitch.x, aim_yaw_pitch.y);
                
                cv::Point2f pred_armor_pixel = armor_solver_->project3DToPixel(predicted_armor_pos);
                // 绘制装甲板预测点（天蓝色）
                cv::circle(frame, pred_armor_pixel, 8, cv::Scalar(255, 255, 0), 2);
            }
        }
    }

    else {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_com_time).count() >= reset_predictor_time) {
            result.reset = true;
            armor_distance_filter_ -> clearHistory();
            if (rotation_motion_model_) {
                RotationMotionState EKFstate = rotation_motion_model_ -> getState();
                if (EKFstate.update_frames > 90) init_r = (EKFstate.r_now, EKFstate.r_another) / 2.0;
                rotation_motion_model_.reset();
            }
            is_reset = true;
            last_pixel_horizontal_center_distance = 1e10;
            has_valid_ballistic = false;
        } else {
            
            result.reset = false;
            result.command_delta_pitch = 0.0;
            result.command_delta_yaw = 0.0;
            result.fire_flag = false;

            bool fire_flag = false;

            if (result.fire_flag) {
                cv::circle(frame, last_aim_yaw_pitch_pixel_, 8, cv::Scalar(0, 0, 255), 2);
            } else {
                cv::circle(frame, last_aim_yaw_pitch_pixel_, 8, cv::Scalar(255, 255, 0), 2);
            }

            if (armor_class_ != ArmorType::Base) {
                cv::Point3f predicted_armor_pos;
                cv::Point3f predicted_aim_pos;
                // ==========================RotationMotionModel===========================
                if (rotation_motion_model_) {
                    double EKF_update_time = (std::chrono::steady_clock::now() - node_start_time_).count() / 1e9;
                    rotation_motion_model_ -> emptyUpdate(EKF_update_time);
                    
                    PredictResult EKF_pred_now_data = rotation_motion_model_ -> predict(0.0);

                    cv::Point3f EKF_pred_now_center_p3f = rest_frame_ -> worldToPnpP3f({
                        static_cast<float>(EKF_pred_now_data.center_x), 
                        static_cast<float>(EKF_pred_now_data.center_y), 
                        static_cast<float>(EKF_pred_now_data.center_z)}
                    );
                    cv::Point2f EKF_pred_now_center_pixel = armor_solver_->project3DToPixel(EKF_pred_now_center_p3f);
                    cv::circle(frame, EKF_pred_now_center_pixel, 10, cv::Scalar(255, 0, 255), 2);

                    cv::Mat EKF_visualize_frame = cv::Mat::zeros(800, 800, CV_8UC3);
                    cv::circle(EKF_visualize_frame, cv::Point2f(400+EKF_pred_now_data.center_x/10, 400-EKF_pred_now_data.center_y/10), 8, cv::Scalar(255, 0, 255), 2);
                    for (int EKF_pred_now_armor_i = 0; EKF_pred_now_armor_i < EKF_pred_now_data.armors.size(); EKF_pred_now_armor_i += 1) {
                        SimpleArmor& EKF_pred_now_armor = EKF_pred_now_data.armors[EKF_pred_now_armor_i];
                        cv::Point3f EKF_pred_now_armor_p3f = rest_frame_ -> worldToPnpP3f({
                            static_cast<float>(EKF_pred_now_armor.x), 
                            static_cast<float>(EKF_pred_now_armor.y), 
                            static_cast<float>(EKF_pred_now_armor.z)
                        });
                        cv::Point2f EKF_pred_now_armor_pixel = armor_solver_->project3DToPixel(EKF_pred_now_armor_p3f);
                        cv::circle(frame, EKF_pred_now_armor_pixel, 6, cv::Scalar(0, 255, 0), 2);
                        // cv::line(frame, EKF_pred_now_center_pixel, EKF_pred_now_armor_pixel, cv::Scalar(0, 255, 0), 2);
                        
                        cv::circle(EKF_visualize_frame, cv::Point2f(400+EKF_pred_now_armor.x/10, 400-EKF_pred_now_armor.y/10), 8, 
                            cv::Scalar(0, 255 - EKF_pred_now_armor_i * 80, EKF_pred_now_armor_i * 80), 2);
                        // cv::line(EKF_visualize_frame, 
                        //     cv::Point2f(400+EKF_pred_now_data.center_x/10, 400-EKF_pred_now_data.center_y/10), 
                        //     cv::Point2f(400+EKF_pred_now_armor.x/10, 400-EKF_pred_now_armor.y/10), 
                        //     cv::Scalar(0, 255 - EKF_pred_now_armor_i * 80, EKF_pred_now_armor_i * 80), 2);
                    }
                    RotationMotionState EKF_state = rotation_motion_model_ -> getState();
                    cv::putText(EKF_visualize_frame, 
                        "EKF_state vyaw:"+std::to_string(EKF_state.vyaw), 
                        cv::Point2f(20,50), 
                        cv::FONT_HERSHEY_COMPLEX, 0.7, 
                        cv::Scalar(0, 255, 0), 1, 8, false);
                    cv::putText(EKF_visualize_frame, 
                        "T:"+std::to_string(EKF_update_time), 
                        cv::Point2f(20,80), 
                        cv::FONT_HERSHEY_COMPLEX, 0.7, 
                        cv::Scalar(0, 255, 0), 1, 8, false);
                    if (using_predictor_type == PredictorType::RotationMotionModel) {
                        PredictResult EKF_pred_aim_data = rotation_motion_model_ -> predict(last_total_delay_);
                        std::vector<float> cam_position = rest_frame_ -> getCamPosition();
                        cv::Point2d cam_to_center_vector = {EKF_pred_aim_data.center_x - cam_position[0], EKF_pred_aim_data.center_y - cam_position[1]};
                        std::vector<double> center_v_dot_yaw(EKF_pred_aim_data.armors.size());
                        float choose_armor_yaw_bias = M_PI / 180.0 * 0.0;
                        choose_armor_yaw_bias *= static_cast<float>(EKF_pred_aim_data.rotation_direction);
                        for (int EKF_pred_aim_armor_i = 0; EKF_pred_aim_armor_i < EKF_pred_aim_data.armors.size(); EKF_pred_aim_armor_i += 1) {
                            SimpleArmor& EKF_pred_aim_armor = EKF_pred_aim_data.armors[EKF_pred_aim_armor_i];
                            cv::Point2d yaw_vector = {std::sin(EKF_pred_aim_armor.yaw + choose_armor_yaw_bias), -std::cos(EKF_pred_aim_armor.yaw + choose_armor_yaw_bias)};
                            center_v_dot_yaw[EKF_pred_aim_armor_i] = cam_to_center_vector.dot(yaw_vector);
                        }
                        int nearest_idx = std::distance(center_v_dot_yaw.begin(), std::min_element(center_v_dot_yaw.begin(), center_v_dot_yaw.end()));
                        auto nearest_armor = EKF_pred_aim_data.armors[nearest_idx];
                        predicted_armor_pos = {
                            static_cast<float>(nearest_armor.x),
                            static_cast<float>(nearest_armor.y),
                            static_cast<float>(nearest_armor.z) 
                        };
                        predicted_aim_pos = predicted_armor_pos;
                        float nearest_armor_yaw_bias = (nearest_armor.yaw - (rotation_motion_model_ -> getCamToCenterYaw())) * static_cast<float>(EKF_pred_aim_data.rotation_direction);
                        while (nearest_armor_yaw_bias < -M_PI) {
                            nearest_armor_yaw_bias += 2*M_PI;
                        }
                        while (nearest_armor_yaw_bias > M_PI) {
                            nearest_armor_yaw_bias -= 2*M_PI;
                        }
                        fire_flag = (nearest_armor_yaw_bias > -30.0 * M_PI / 180.0) && (nearest_armor_yaw_bias < 30.0 * M_PI / 180.0);
                        cv::circle(EKF_visualize_frame, 
                            cv::Point2f(400+nearest_armor.x/10, 400-nearest_armor.y/10), 8, 
                            cv::Scalar(0, 0, 255), 2);
                        cv::putText(EKF_visualize_frame, 
                            "r_now:"+std::to_string(EKF_pred_aim_data.r_now), 
                            cv::Point2f(20,110), 
                            cv::FONT_HERSHEY_COMPLEX, 0.7, 
                            cv::Scalar(0, 255, 0), 1, 8, false);
                        cv::putText(EKF_visualize_frame, 
                            "r_another:"+std::to_string(EKF_pred_aim_data.r_another), 
                            cv::Point2f(300,110), 
                            cv::FONT_HERSHEY_COMPLEX, 0.7, 
                            cv::Scalar(0, 255, 0), 1, 8, false);
                    }
                    cv::line(EKF_visualize_frame, 
                        cv::Point2f(400, 400), 
                        cv::Point2f(400 - std::sin(total_yaw_rad_delayed_)*150, 
                                    400 - std::cos(total_yaw_rad_delayed_)*150),
                        cv::Scalar(255, 255, 0), 2);
                    cv::putText(EKF_visualize_frame, 
                        "total_yaw:"+std::to_string(total_yaw_rad_delayed_), 
                        cv::Point2f(20,140), 
                        cv::FONT_HERSHEY_COMPLEX, 0.7, 
                        cv::Scalar(0, 255, 0), 1, 8, false);
                    cv::putText(EKF_visualize_frame, 
                        "vx:"+std::to_string(EKF_state.center_vx), 
                        cv::Point2f(20,200), 
                        cv::FONT_HERSHEY_COMPLEX, 0.7, 
                        cv::Scalar(0, 255, 0), 1, 8, false);
                    cv::putText(EKF_visualize_frame, 
                        "vy:"+std::to_string(EKF_state.center_vy), 
                        cv::Point2f(20,230), 
                        cv::FONT_HERSHEY_COMPLEX, 0.7, 
                        cv::Scalar(0, 255, 0), 1, 8, false);
                    cv::line(EKF_visualize_frame, 
                        cv::Point2f(400 + EKF_pred_now_data.center_x/10, 400 - EKF_pred_now_data.center_y/10), 
                        cv::Point2f(400 + (EKF_pred_now_data.center_x/10 + EKF_state.center_vx/5), 
                                    400 - (EKF_pred_now_data.center_y/10 + EKF_state.center_vy/5)),
                        cv::Scalar(255, 255, 0), 2);
#ifdef SHOW_WINDOWS
                    cv::imshow("EKF visualize "+std::to_string(armor_class_), EKF_visualize_frame);
#endif
                }
                // ==========================RotationMotionModel=========================== END
                // 统一转换回pnp相机坐标系
                predicted_aim_pos = rest_frame_ -> worldToPnpP3f(predicted_aim_pos);
                predicted_armor_pos = rest_frame_ -> worldToPnpP3f(predicted_armor_pos);

                // 弹道解算
                RCLCPP_DEBUG(node_->get_logger(), "aim pos: (%.2f, %.2f, %.2f)",
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
                    has_valid_ballistic_ = true;
                    // RCLCPP_INFO(node_->get_logger(), "Target detected, publishing command");
                    // has_valid_target_ = true;

                    // 发布云台控制命令
                    //serial_communication_->sendData(command_pitch, command_yaw, fire_flag);
                    result.reset = false;
                    result.command_delta_pitch = ballistic_result.delta_pitch_rad;
                    result.command_delta_yaw = ballistic_result.delta_yaw_rad;
                    result.fire_flag = fire_flag;
                }

                // 绘制瞄准预测点（黄色）
                cv::Point2f pred_aim_pixel = armor_solver_->project3DToPixel(predicted_aim_pos);
                cv::circle(frame, pred_aim_pixel, 8, cv::Scalar(0, 255, 255), 2);
                // 绘制装甲板预测点（天蓝色）
                cv::Point2f pred_armor_pixel = armor_solver_->project3DToPixel(predicted_armor_pos);
                cv::circle(frame, pred_armor_pixel, 8, cv::Scalar(255, 255, 0), 2);
            }

            if ((!ballistic_valid_flag) && has_valid_ballistic_) {
                result.reset = false;
                result.command_delta_pitch = 0.0;
                result.command_delta_yaw = 0.0;
                result.fire_flag = false;
            }
            if (result.fire_flag) {
                cv::circle(frame, last_aim_yaw_pitch_pixel_, 8, cv::Scalar(0, 0, 255), 2);
            } else {
                cv::circle(frame, last_aim_yaw_pitch_pixel_, 8, cv::Scalar(255, 255, 0), 2);
            }
        }
    }
    return result;
}

PredictResult PredictorEKF::state_convert_to_PredictResult(Tracker::State state) {
    // 根据当前状态提取装甲板位置
    double xc = state(0);
    double yc = state(2);
    double zc = state(4);
    double r_now = state(8);
    double yaw = state(6);
    double rotation_direction = (state(7) >= 0) ? 1.0 : -1.0;
    std::vector<SimpleArmor> armors;

    for (int i = 0; i < n_armors_; i++) {
        double armor_yaw = yaw - i * rotation_direction * jump_rad_;
        double r_using = is_outpost ? r_now_ : ((i%2==0) ? r_now_ : r_another_);
        double z_using = is_outpost ? zc : ((i%2==0) ? zc : z_another_);
        armors.push_back(SimpleArmor({
            xc + r_using * std::sin(armor_yaw),
            yc - r_using * std::cos(armor_yaw),
            z_using,
            r_using,
            armor_yaw
        }));
    }

    PredictResult EKF_pred_now_data = {
        xc, yc, zc, z_another_,
        r_now, r_another_,
        yaw, rotation_direction, armors
    };
    return EKF_pred_now_data;
}

void visualize(){
    
}