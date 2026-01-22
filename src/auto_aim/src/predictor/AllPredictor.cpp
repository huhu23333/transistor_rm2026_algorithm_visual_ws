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

    if (armor_class != ArmorType::Base) {
        // using_predictor_type = predictor_switcher_ -> step();
        if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - latest_predicting_start_time).count()
            < pre_predict_time) {

            using_predictor_type = PredictorType::None;
        } else {
            using_predictor_type = PredictorType::EKF;// predictor_switcher_ -> step();
        }
    }

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
            armor_is_large = best_result.is_large;

            is_reset = false;
            last_com_time = std::chrono::steady_clock::now();

            last_pixel_horizontal_center_distance = std::abs(best_result.center.x - static_cast<float>(frame.cols)/2.0);
            latest_armor_distance = std::sqrt(aim.distance);
            
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
            RCLCPP_DEBUG(node->get_logger(), "camera euler angles: yaw=%.2f, pitch=%.2f, roll=%.2f", aim.normal_euler_angles[0], aim.normal_euler_angles[1], aim.normal_euler_angles[2]);
            RCLCPP_DEBUG(node->get_logger(), "Rest frame pos: x=%.2f, y=%.2f, z=%.2f, yaw=%.2f", rest_frame_pos.x, rest_frame_pos.y, rest_frame_pos.z, rest_frame_euler_angles[0]);

            last_rest_frame_pos = rest_frame_pos;

            // 提前预测与弹道解算
            constexpr float image_latency = 0.013f;
            constexpr float comm_latency  = 0.010f;
            float bullet_time = (bullet_velocity_ > 1.0f) ? (std::abs(aim.position.z) / 1000.0f / bullet_velocity_) : 0.0f;
            float extra_time = 0.300f; // 0.300f
            float total_delay = image_latency + comm_latency + bullet_time + extra_time;
            last_total_delay_ = total_delay;

            // 默认使用 None （直接瞄准装甲板）
            cv::Point3f predicted_armor_pos = rest_frame_pos;
            cv::Point3f predicted_aim_pos = predicted_armor_pos;
            bool fire_flag = true;
            

            if (armor_class != ArmorType::Base) {
                std::vector<float> cam_position = rest_frame_ -> getCamPosition();
                cv::Point3f bullet_nearest_point = ballistic_solver_ -> calcNearestPointWithAirResistance( // todo
                    rest_frame_pos / 1000, {cam_position[0], cam_position[1], cam_position[2]}, last_aim_yaw_pitch_, bullet_velocity_) * 1000;
                cv::Point3f pnp_bullet_nearest_point = rest_frame_ -> worldToPnpP3f(bullet_nearest_point);
                cv::Point2f bullet_nearest_point_pixel = armor_solver_->project3DToPixel(pnp_bullet_nearest_point);
                cv::circle(frame, bullet_nearest_point_pixel, 15, cv::Scalar(255, 0, 255), 2);
                RCLCPP_DEBUG(node->get_logger(), "bullet_nearest_point: (%.2f, %.2f, %.2f)",
                            bullet_nearest_point.x, bullet_nearest_point.y, bullet_nearest_point.z);


                // ========================== EKF 逻辑 (9D模型修改) ===========================
                // 1) 构造 4 维量测 z = [xa, ya, za, yaw_a]
                armor_ekf::Tracker::Measurement z;
                z << rest_frame_pos.x, rest_frame_pos.y, rest_frame_pos.z, rest_frame_euler_angles[0];

                // 2) 初始化 / 状态机
                if (armor_tracker_->trackState() == armor_ekf::Tracker::TrackState::LOST) {
                    // 车型/装甲数量（没有更细分类时，默认 4 板）
                    armor_tracker_->setArmorsNum(armor_ekf::Tracker::ArmorsNum::NORMAL_4);

                    // 由测量 + 先验几何反解中心并初始化（单位 mm）
                    armor_tracker_->resetFromArmor(z, ekf_init_r_mm_, ekf_init_dz_mm_);

                    // 阈值从配置设定
                    armor_tracker_->setMatchThresholds(ekf_max_match_distance_mm_, ekf_max_match_yaw_diff_rad_);

                    current_target_id_ = best_result.number;
                } else {
                    // 3) yaw 跳变处理（四装甲切换）——先做几何纠正，再 predict/update
                    armor_tracker_->handleArmorJump(
                        /*measured_yaw*/ z(3),
                        /*measured_pos*/ Eigen::Vector3d(rest_frame_pos.x, rest_frame_pos.y, rest_frame_pos.z)
                    );

                    // 4) 常规 EKF
                    armor_tracker_->predict();
                    armor_tracker_->update(z);
                }

                if (using_predictor_type == PredictorType::EKF) {
                    // 5) 时延前推，predictAhead 返回 [xc, yc, zc, yaw]^T（4x1）
                    const auto future_c = armor_tracker_->predictAhead(total_delay);
                    const double xc_f  = future_c(0);
                    const double yc_f  = future_c(1);
                    const double zc_f  = future_c(2);
                    const double yaw_f = future_c(3);
                    const double r     = armor_tracker_->state()(8);
                    const double dz    = armor_tracker_->state()(9);

                    double switch_strategy = 0.35;

                    Eigen::Vector3d best_pos = armor_tracker_->predictBestArmorPosition(total_delay, switch_strategy);

                    // 由中心状态反解“该块装甲”的未来位置
                    predicted_armor_pos = {
                        static_cast<float>(xc_f + r * std::sin(yaw_f)),
                        static_cast<float>(yc_f - r * std::cos(yaw_f)),
                        static_cast<float>(zc_f)
                    };

                    // 如你的代码有“瞄准点=装甲中心”的逻辑，保持一致：
                    predicted_aim_pos = predicted_armor_pos;
                }

                float ekf_center_x, ekf_v_x, ekf_center_y, ekf_v_y, ekf_center_z, ekf_v_z, ekf_yaw, ekf_v_yaw, ekf_r, ekf_dz;
                Eigen::Matrix<double, 10, 1> ekf_state = armor_tracker_->state();
                ekf_center_x = ekf_state(0);
                ekf_v_x = ekf_state(1);
                ekf_center_y = ekf_state(2);
                ekf_v_y = ekf_state(3);
                ekf_center_z = ekf_state(4);
                ekf_v_z = ekf_state(5);
                ekf_yaw = ekf_state(6);
                ekf_v_yaw = ekf_state(7);
                ekf_r = ekf_state(8);
                ekf_dz = ekf_state(9);
                // ========================== EKF 逻辑 (9D模型修改) =========================== END


/*
                cv::Mat RMM_visualize_frame = cv::Mat::zeros(800, 800, CV_8UC3);
                if (best_result.is_tracked_now) {
                    cv::circle(RMM_visualize_frame, cv::Point2f(400+ekf_center_x/10, 400-ekf_center_y/10), 8, cv::Scalar(0, 255, 0), 2);
                } else {
                    cv::circle(RMM_visualize_frame, cv::Point2f(400+ekf_center_x/10, 400-ekf_center_y/10), 8, cv::Scalar(255, 0, 255), 2);
                }
                for (int RMM_pred_now_armor_i = 0; RMM_pred_now_armor_i < RMM_pred_now_data.armors.size(); RMM_pred_now_armor_i += 1) {
                    SimpleArmor& RMM_pred_now_armor = RMM_pred_now_data.armors[RMM_pred_now_armor_i];
                    cv::Point3f RMM_pred_now_armor_p3f = rest_frame_ -> worldToPnpP3f({
                        static_cast<float>(RMM_pred_now_armor.x), 
                        static_cast<float>(RMM_pred_now_armor.y), 
                        static_cast<float>(RMM_pred_now_armor.z)
                    });
                    cv::Point2f RMM_pred_now_armor_pixel = armor_solver_->project3DToPixel(RMM_pred_now_armor_p3f);
                    cv::circle(frame, RMM_pred_now_armor_pixel, 6, cv::Scalar(0, 255, 0), 2);
                    // cv::line(frame, RMM_pred_now_center_pixel, RMM_pred_now_armor_pixel, cv::Scalar(0, 255, 0), 2);
                    
                    cv::circle(RMM_visualize_frame, cv::Point2f(400+RMM_pred_now_armor.x/10, 400-RMM_pred_now_armor.y/10), 8, 
                        cv::Scalar(0, 255 - RMM_pred_now_armor_i * 80, RMM_pred_now_armor_i * 80), 2);
                    // cv::line(RMM_visualize_frame, 
                    //     cv::Point2f(400+RMM_pred_now_data.center_x/10, 400-RMM_pred_now_data.center_y/10), 
                    //     cv::Point2f(400+RMM_pred_now_armor.x/10, 400-RMM_pred_now_armor.y/10), 
                    //     cv::Scalar(0, 255 - RMM_pred_now_armor_i * 80, RMM_pred_now_armor_i * 80), 2);
                }
                cv::circle(RMM_visualize_frame, cv::Point2f(400+rest_frame_pos.x/10, 400-rest_frame_pos.y/10), 8, cv::Scalar(255, 255, 0), 2);
                cv::line(RMM_visualize_frame, 
                    cv::Point2f(400 + rest_frame_pos.x/10, 400-rest_frame_pos.y/10), 
                    cv::Point2f(400 + rest_frame_pos.x/10 + std::sin(rest_frame_euler_angles[0])*50, 
                                400 - (rest_frame_pos.y/10 - std::cos(rest_frame_euler_angles[0])*50)),
                    cv::Scalar(255, 255, 0), 2);
                double theoretic_yaw = rotation_motion_model_ -> getTheoreticYaw(rest_frame_pos.x, rest_frame_pos.y);
                cv::line(RMM_visualize_frame, 
                    cv::Point2f(400 + rest_frame_pos.x/10, 400-rest_frame_pos.y/10), 
                    cv::Point2f(400 + rest_frame_pos.x/10 + std::sin(theoretic_yaw)*50, 
                                400 - (rest_frame_pos.y/10 - std::cos(theoretic_yaw)*50)),
                    cv::Scalar(0, 255, 0), 2);
                RotationMotionState RMM_state = rotation_motion_model_ -> getState();
                cv::putText(RMM_visualize_frame, 
                    "RMM_state vyaw:"+std::to_string(RMM_state.vyaw), 
                    cv::Point2f(20,50), 
                    cv::FONT_HERSHEY_COMPLEX, 0.7, 
                    cv::Scalar(0, 255, 0), 1, 8, false);
                cv::putText(RMM_visualize_frame, 
                    "T:"+std::to_string(RMM_update_time), 
                    cv::Point2f(20,80), 
                    cv::FONT_HERSHEY_COMPLEX, 0.7, 
                    cv::Scalar(0, 255, 0), 1, 8, false);
                if (using_predictor_type == PredictorType::RotationMotionModel) {
                    PredictResult RMM_pred_aim_data = rotation_motion_model_ -> predict(total_delay);
                    cv::Point2d cam_to_center_vector = {RMM_pred_aim_data.center_x - cam_position[0], RMM_pred_aim_data.center_y - cam_position[1]};
                    std::vector<double> center_v_dot_yaw(RMM_pred_aim_data.armors.size());
                    float choose_armor_yaw_bias = M_PI / 180.0 * 0.0;
                    choose_armor_yaw_bias *= static_cast<float>(RMM_pred_aim_data.rotation_direction);
                    for (int RMM_pred_aim_armor_i = 0; RMM_pred_aim_armor_i < RMM_pred_aim_data.armors.size(); RMM_pred_aim_armor_i += 1) {
                        SimpleArmor& RMM_pred_aim_armor = RMM_pred_aim_data.armors[RMM_pred_aim_armor_i];
                        cv::Point2d yaw_vector = {std::sin(RMM_pred_aim_armor.yaw + choose_armor_yaw_bias), -std::cos(RMM_pred_aim_armor.yaw + choose_armor_yaw_bias)};
                        center_v_dot_yaw[RMM_pred_aim_armor_i] = cam_to_center_vector.dot(yaw_vector);
                    }
                    int nearest_idx = std::distance(center_v_dot_yaw.begin(), std::min_element(center_v_dot_yaw.begin(), center_v_dot_yaw.end()));
                    auto nearest_armor = RMM_pred_aim_data.armors[nearest_idx];
                    predicted_armor_pos = {
                        static_cast<float>(nearest_armor.x),
                        static_cast<float>(nearest_armor.y),
                        static_cast<float>(nearest_armor.z) 
                    };
                    float nearest_armor_yaw_bias = (nearest_armor.yaw - (rotation_motion_model_ -> getCamToCenterYaw())) * static_cast<float>(RMM_pred_aim_data.rotation_direction);
                    while (nearest_armor_yaw_bias < -M_PI) {
                        nearest_armor_yaw_bias += 2*M_PI;
                    }
                    while (nearest_armor_yaw_bias > M_PI) {
                        nearest_armor_yaw_bias -= 2*M_PI;
                    }

                    RMM_fire_result_t RMM_fire_result = RMM_fire_control(nearest_armor, RMM_state, nearest_armor_yaw_bias, armor_is_large);
                    if (RMM_fire_result.aim_center) {
                        predicted_aim_pos = {
                            static_cast<float>(RMM_pred_aim_data.center_x),
                            static_cast<float>(RMM_pred_aim_data.center_y),
                            static_cast<float>(nearest_armor.z) 
                        };
                    } else {
                        predicted_aim_pos = predicted_armor_pos;
                    }
                    fire_flag = RMM_fire_result.fire;
                    
                    cv::circle(RMM_visualize_frame, 
                        cv::Point2f(400+nearest_armor.x/10, 400-nearest_armor.y/10), 8, 
                        cv::Scalar(0, 0, 255), 2);
                    cv::putText(RMM_visualize_frame, 
                        "r_now:"+std::to_string(RMM_pred_aim_data.r_now), 
                        cv::Point2f(20,110), 
                        cv::FONT_HERSHEY_COMPLEX, 0.7, 
                        cv::Scalar(0, 255, 0), 1, 8, false);
                    cv::putText(RMM_visualize_frame, 
                        "r_another:"+std::to_string(RMM_pred_aim_data.r_another), 
                        cv::Point2f(300,110), 
                        cv::FONT_HERSHEY_COMPLEX, 0.7, 
                        cv::Scalar(0, 255, 0), 1, 8, false);
                    cv::putText(RMM_visualize_frame, 
                        "center_z:"+std::to_string(RMM_pred_aim_data.center_z), 
                        cv::Point2f(20,140), 
                        cv::FONT_HERSHEY_COMPLEX, 0.7, 
                        cv::Scalar(0, 255, 0), 1, 8, false);
                    cv::putText(RMM_visualize_frame, 
                        "z_another:"+std::to_string(RMM_pred_aim_data.z_another), 
                        cv::Point2f(300,140), 
                        cv::FONT_HERSHEY_COMPLEX, 0.7, 
                        cv::Scalar(0, 255, 0), 1, 8, false);
                }
                cv::line(RMM_visualize_frame, 
                    cv::Point2f(400, 400), 
                    cv::Point2f(400 - std::sin(total_yaw_rad_delayed_)*150, 
                                400 - std::cos(total_yaw_rad_delayed_)*150),
                    cv::Scalar(255, 255, 0), 2);
                cv::putText(RMM_visualize_frame, 
                    "total_yaw:"+std::to_string(total_yaw_rad_delayed_), 
                    cv::Point2f(20,170), 
                    cv::FONT_HERSHEY_COMPLEX, 0.7, 
                    cv::Scalar(0, 255, 0), 1, 8, false);
                cv::putText(RMM_visualize_frame, 
                    "vx:"+std::to_string(RMM_state.center_vx), 
                    cv::Point2f(20,200), 
                    cv::FONT_HERSHEY_COMPLEX, 0.7, 
                    cv::Scalar(0, 255, 0), 1, 8, false);
                cv::putText(RMM_visualize_frame, 
                    "vy:"+std::to_string(RMM_state.center_vy), 
                    cv::Point2f(20,230), 
                    cv::FONT_HERSHEY_COMPLEX, 0.7, 
                    cv::Scalar(0, 255, 0), 1, 8, false);
                cv::line(RMM_visualize_frame, 
                    cv::Point2f(400 + RMM_pred_now_data.center_x/10, 400 - RMM_pred_now_data.center_y/10), 
                    cv::Point2f(400 + (RMM_pred_now_data.center_x/10 + RMM_state.center_vx/5), 
                                400 - (RMM_pred_now_data.center_y/10 + RMM_state.center_vy/5)),
                    cv::Scalar(255, 255, 0), 2);
#ifdef SHOW_WINDOWS
                cv::imshow("RMM visualize "+std::to_string(armor_class), RMM_visualize_frame);
#endif
                // ========================== RotationMotionModsel =========================== END*/


            }

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
                
                cv::Point2f pred_armor_pixel = armor_solver_->project3DToPixel(predicted_armor_pos);
                // 绘制装甲板预测点（天蓝色）
                cv::circle(frame, pred_armor_pixel, 8, cv::Scalar(255, 255, 0), 2);
            }
        }
    }

    else {

        if (armor_tracker_->trackState() != armor_ekf::Tracker::TrackState::LOST) {
            armor_tracker_->predict();
        }

        if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_com_time).count() >= reset_predictor_time) {
            result.reset = true;
            armor_distance_filter_ -> clearHistory();
            if (rotation_motion_model_) {
                RotationMotionState RMMstate = rotation_motion_model_ -> getState();
                if (RMMstate.update_frames > 90) init_r = (RMMstate.r_now, RMMstate.r_another) / 2.0;
                rotation_motion_model_.reset();
            }
            is_reset = true;
            last_pixel_horizontal_center_distance = 1e10;
            has_valid_ballistic = false;
            RMM_fire_control_data.new_target = true;
            latest_armor_distance = 1e10;
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

            if (armor_class != ArmorType::Base) {
                cv::Point3f predicted_armor_pos;
                cv::Point3f predicted_aim_pos;
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

            if ((!ballistic_valid_flag) && has_valid_ballistic) {
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
    
    oscilloscope_common_ -> addDataPoint(((float)(result.fire_flag))/11.0, 1);
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
    result.latest_armor_distance = latest_armor_distance;






    // 调试代码：绘制EKF预测的整车模型 
    if (armor_tracker_ && armor_tracker_->trackState() != armor_ekf::Tracker::TrackState::LOST) {
        // 1. 获取当前后验状态 [xc, vxc, yc, vyc, zc, vzc, yaw, vyaw, r, dz]
        auto state = armor_tracker_->state();
        double xc = state(0);
        double yc = state(2);
        double zc = state(4);
        double yaw = state(6);
        double r = state(8);
        //double dz = state(9);

        // 2. 绘制车体中心 (红色实心点)
        // 先从世界系转回相机系(PnP系)，再投影到像素系
        cv::Point3f center_world(xc, yc, zc);
        cv::Point3f center_pnp = rest_frame_->worldToPnpP3f(center_world);
        cv::Point2f center_pixel = armor_solver_->project3DToPixel(center_pnp);
        cv::circle(frame, center_pixel, 6, cv::Scalar(0, 0, 255), -3); 
        cv::putText(frame, "Center", center_pixel, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);

        // 3. 绘制 4 块装甲板 (绿色空心圆 + 连线)
        // 注意：必须与 motion_model.hpp 中的 Measure 结构体逻辑严格一致
        // motion_model 中: xa = xc + OFFSET_SIGN * r * cos(yaw)
        //                  ya = yc - OFFSET_SIGN * r * sin(yaw)
        // 且 OFFSET_SIGN = -1.0
        double offset_sign = 1.0; // 务必与 motion_model.hpp 保持一致

        int debug_armor_num = 4; // 默认画4板
        for (int i = 0; i < debug_armor_num; ++i) {
            // 计算第 i 块板的 yaw
            double armor_yaw = yaw + i * (2.0 * M_PI / debug_armor_num);
            
            // 反解位置
            double xa = xc + offset_sign * r * std::sin(armor_yaw);
            double ya = yc - offset_sign * r * std::cos(armor_yaw);
            double za = zc;

            // 投影
            cv::Point3f armor_world_pt(xa, ya, za);
            cv::Point3f armor_pnp_pt = rest_frame_->worldToPnpP3f(armor_world_pt);
            cv::Point2f armor_pixel_pt = armor_solver_->project3DToPixel(armor_pnp_pt);

            // 绘制
            if (armor_pixel_pt.x > 0 && armor_pixel_pt.x < frame.cols && 
                armor_pixel_pt.y > 0 && armor_pixel_pt.y < frame.rows) {
                
                cv::circle(frame, armor_pixel_pt, 8, cv::Scalar(0, 255, 0), 4);
                cv::line(frame, center_pixel, armor_pixel_pt, cv::Scalar(255, 255, 0), 2); // 连线
                cv::putText(frame, std::to_string(i), armor_pixel_pt, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            }
        }
        
        // 在屏幕左上角打印当前状态信息
        std::stringstream ss;
        ss << "Yaw: " << std::fixed << std::setprecision(2) << yaw 
        << " vYaw: " << state(7) << " R: " << r << " dz: " << state(9);
        cv::putText(frame, ss.str(), cv::Point(20, 150), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
    }
    // ================= 调试代码结束 =================



    if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - latest_predicting_start_time).count()
        < pre_predict_time_not_aim) {

        result.fire_flag = false;
        result.reset = true;
    }

    return result;
}


RMM_fire_result_t AllPredictor::RMM_fire_control(SimpleArmor chosen_armor, RotationMotionState RMM_state, float yaw_bias, bool is_large_armor) {
    RMM_fire_result_t result;

    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    if (RMM_fire_control_data.new_target) {
        RMM_fire_control_data.last_target_yaw_jump_time = now;
        RMM_fire_control_data.new_target = false;
    } else {
        float yaw_diff = fabs(RMM_fire_control_data.last_target_yaw - chosen_armor.yaw);
        while (yaw_diff > M_PI) {
            yaw_diff -= 2 *  M_PI;
        }
        while (yaw_diff < -M_PI) {
            yaw_diff += 2 *  M_PI;
        }
        if (yaw_diff > M_PI / 4.0) {
            RMM_fire_control_data.last_target_yaw_jump_time = now;
        }
    }
    RMM_fire_control_data.last_target_yaw = chosen_armor.yaw;

    if (fabs(RMM_state.vyaw) > RMM_fire_control_data.aim_center_vyaw_threshold) {
        result.aim_center = true;
    } else {
        result.aim_center = false;
    }

    if (result.aim_center) {
        float max_yaw_bias = std::atan2((is_large_armor ? ArmorConstants::LARGE_ARMOR_WIDTH : ArmorConstants::SMALL_ARMOR_WIDTH) / 2.0, 
                                        chosen_armor.r) + RMM_fire_control_data.aim_center_yaw_bias_expand;
        if (fabs(yaw_bias) < max_yaw_bias) {
            result.fire = true;
        } else {
            result.fire = false;
        }
    } else {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - RMM_fire_control_data.last_target_yaw_jump_time).count() 
            < RMM_fire_control_data.target_change_ceasefire_ms) {

            result.fire = false;
        } else {
            result.fire = true;
        }
    }

    return result;
}