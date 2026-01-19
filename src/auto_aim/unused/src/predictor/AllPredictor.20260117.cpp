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
            AimResult aim = best_result.solve_armor_result;// armor_solver_->solveArmor(best_result, last_pitch_rad_delayed_, last_yaw_rad_delayed_);
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
                float extra_time = 0.300f; // 0.300f
                float total_delay = image_latency + comm_latency + bullet_time + extra_time;
                last_total_delay_ = total_delay;

                // 默认使用 None （直接瞄准装甲板）
                cv::Point3f predicted_armor_pos = rest_frame_pos;
                cv::Point3f predicted_aim_pos = predicted_armor_pos;
                bool fire_flag = true;
                

                if (armor_class != ArmorType::Base) {
                    // ========================== EKF 逻辑 (9D模型修改) ===========================
                    // 1. 构造4维测量向量 z = [xa, ya, za, yaw_a]
                    Tracker::Measurement z;
                    z << rest_frame_pos.x, rest_frame_pos.y, rest_frame_pos.z, rest_frame_euler_angles[0];

                    // 2. EKF 状态机逻辑
                    if (EKF_tracker_->state == Tracker::LOST) {
                        EKF_tracker_->reset(z);
                        current_target_id_ = best_result.number;
                    } else {
                        // 跳变处理：通过ID或距离判断
                        Eigen::Vector3d pred_armor_pos = EKF_tracker_->getArmorPosition();
                        double position_diff = (pred_armor_pos - Eigen::Vector3d(rest_frame_pos.x, rest_frame_pos.y, rest_frame_pos.z)).norm();

                        if (best_result.number != current_target_id_ || position_diff > RESET_DISTANCE_THRESHOLD) {
                            if(best_result.number != current_target_id_) {
                                RCLCPP_DEBUG(node->get_logger(), "ID switched, resetting tracker.");
                            } else {
                                RCLCPP_DEBUG(node->get_logger(), "Position jumped (%.f mm), resetting tracker.", position_diff);
                            }
                            EKF_tracker_->reset(z);
                            current_target_id_ = best_result.number;
                        } else {
                            EKF_tracker_->predict();
                            EKF_tracker_->update(z);
                        }
                    }
                    
                    if (using_predictor_type == PredictorType::EKF) {

                        // 获取提前预测后的机器人中心状态
                        Tracker::State future_state = EKF_tracker_->predictAhead(total_delay);
                        
                        // 从预测的机器人中心状态，反解出未来时刻装甲板的位置
                        double future_xc = future_state(0), future_yc = future_state(2), future_zc = future_state(4);
                        double future_yaw = future_state(6), future_r = future_state(8);

                        predicted_armor_pos = {
                            static_cast<float>(future_xc - future_r * sin(future_yaw)),
                            static_cast<float>(future_yc + future_r * cos(future_yaw)),
                            static_cast<float>(future_zc) 
                        };
                        predicted_aim_pos = predicted_armor_pos;
                        fire_flag = true;
                        
                        RCLCPP_DEBUG(node->get_logger(), "yaw: %.2f" , rest_frame_euler_angles[0] );
                        RCLCPP_DEBUG(node->get_logger(), "distance: %.2f" , aim.distance );
                        RCLCPP_DEBUG(node->get_logger(), "position: (%.2f, %.2f, %.2f)" , aim.position.x, aim.position.y, aim.position.z);
                        RCLCPP_DEBUG(node->get_logger(), "Future armor pos: (%.2f, %.2f, %.2f)",
                                    predicted_aim_pos.x, predicted_aim_pos.y, predicted_aim_pos.z);
                    }
                    
                    // ========================== EKF 逻辑 (9D模型修改) =========================== END
                    // ========================== FirePredictor ===========================
                    predictor3d -> addPoint(rest_frame_pos);
                    predictor3d -> fitFourier(predictor3d_fit_step, predictor3d_fourier_fit_order);
                    predictor3dCenterPredictions = predictor3d -> predictLinear(predictor3d_predict_step);//std::vector<cv::Point3f>(predictor3d_predict_step, cv::Point3f(predictor3d -> getAveragePosition())); //predictor3d -> predictLinear(predictor3d_predict_step); // predictFourier | predictLinear
                    predictor3dArmorPredictions = predictor3d -> predictFourier(predictor3d_predict_step); // predictFourier | predictLinear
                    predictor3dPrediction_nowIndex = 0;
                    size_t predictor3dPrediction_indexToAim = std::min(predictor3d_predict_step-1, (int)(total_delay * fps_counter->fps())); // total_delay
    //#ifdef USE_PREDICTOR3D
                    if (using_predictor_type == PredictorType::FirePredictor) {
                        predicted_armor_pos = predictor3dArmorPredictions[predictor3dPrediction_indexToAim]; // todo
                        predicted_aim_pos = predictor3dCenterPredictions[predictor3dPrediction_indexToAim]; // todo
                    }

                    // 计算弹道最近点并绘制（大紫色圈）
                    std::vector<float> cam_position = rest_frame_ -> getCamPosition();
                    cv::Point3f bullet_nearest_point = ballistic_solver_ -> calcNearestPointWithAirResistance( // todo
                        rest_frame_pos / 1000, {cam_position[0], cam_position[1], cam_position[2]}, last_aim_yaw_pitch_, bullet_velocity_) * 1000;
                    cv::Point3f pnp_bullet_nearest_point = rest_frame_ -> worldToPnpP3f(bullet_nearest_point);
                    cv::Point2f bullet_nearest_point_pixel = armor_solver_->project3DToPixel(pnp_bullet_nearest_point);
                    cv::circle(frame, bullet_nearest_point_pixel, 15, cv::Scalar(255, 0, 255), 2);
                    RCLCPP_DEBUG(node->get_logger(), "bullet_nearest_point: (%.2f, %.2f, %.2f)",
                                bullet_nearest_point.x, bullet_nearest_point.y, bullet_nearest_point.z);

                    bool armor_near_flag = cv::norm(bullet_nearest_point - rest_frame_pos) < fire_distance; // todo


                    fire_data_predictor_ -> setPeriod(predictor3d->getFourierPeriod());
                    //fire_data_predictor_ -> autoFindPeriod();
                    fire_data_predictor_ -> addPoint(armor_near_flag);
                    pred_fire_data_filter_ -> addPoint(fire_data_predictor_ -> isUpper(predictor3dPrediction_indexToAim, 0.5) || fire_data_predictor_ -> getA0() > 0.8);
                    if (using_predictor_type == PredictorType::FirePredictor) {
                        fire_flag = pred_fire_data_filter_ -> getExponentialValue() > 0.5;
                    }
                    //oscilloscope_fire_ -> addDataPoint(pred_fire_data_filter_ -> getExponentialValue());
                    // ========================== FirePredictor =========================== END
                    // ========================== RotationMotionModel ===========================
                    double RMM_update_time = (std::chrono::steady_clock::now() - node_start_time).count() / 1e9;
                    ObservedData RMM_update_data({
                        rest_frame_pos.x, rest_frame_pos.y, rest_frame_pos.z, rest_frame_euler_angles[0],
                        RMM_update_time
                    });
                    if (!rotation_motion_model_) {
                        rotation_motion_model_ = std::make_unique<RotationMotionModel>(RMM_update_data, rest_frame_, armor_class==ArmorType::Outpost, init_r);
                    } else {
                        if (best_result.is_tracked_now) {
                            rotation_motion_model_ -> update(RMM_update_data);
                        } else {
                            rotation_motion_model_ -> emptyUpdate(RMM_update_time);
                        }
                    }

                    PredictResult RMM_pred_now_data = rotation_motion_model_ -> predict(0.0);
                    cv::Point3f RMM_pred_now_center_p3f = rest_frame_ -> worldToPnpP3f({
                        static_cast<float>(RMM_pred_now_data.center_x), 
                        static_cast<float>(RMM_pred_now_data.center_y), 
                        static_cast<float>(RMM_pred_now_data.center_z)
                    });
                    cv::Point2f RMM_pred_now_center_pixel = armor_solver_->project3DToPixel(RMM_pred_now_center_p3f);
                    if (best_result.is_tracked_now) {
                        cv::circle(frame, RMM_pred_now_center_pixel, 10, cv::Scalar(0, 255, 0), 2);
                    } else {
                        cv::circle(frame, RMM_pred_now_center_pixel, 10, cv::Scalar(255, 0, 255), 2);
                    }

                    cv::Mat RMM_visualize_frame = cv::Mat::zeros(800, 800, CV_8UC3);
                    if (best_result.is_tracked_now) {
                        cv::circle(RMM_visualize_frame, cv::Point2f(400+RMM_pred_now_data.center_x/10, 400-RMM_pred_now_data.center_y/10), 8, cv::Scalar(0, 255, 0), 2);
                    } else {
                        cv::circle(RMM_visualize_frame, cv::Point2f(400+RMM_pred_now_data.center_x/10, 400-RMM_pred_now_data.center_y/10), 8, cv::Scalar(255, 0, 255), 2);
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
                        predicted_aim_pos = predicted_armor_pos;
                        float nearest_armor_yaw_bias = (nearest_armor.yaw - (rotation_motion_model_ -> getCamToCenterYaw())) * static_cast<float>(RMM_pred_aim_data.rotation_direction);
                        while (nearest_armor_yaw_bias < -M_PI) {
                            nearest_armor_yaw_bias += 2*M_PI;
                        }
                        while (nearest_armor_yaw_bias > M_PI) {
                            nearest_armor_yaw_bias -= 2*M_PI;
                        }
                        fire_flag = (nearest_armor_yaw_bias > -30.0 * M_PI / 180.0) && (nearest_armor_yaw_bias < 30.0 * M_PI / 180.0);
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
                    }
                    cv::line(RMM_visualize_frame, 
                        cv::Point2f(400, 400), 
                        cv::Point2f(400 - std::sin(total_yaw_rad_delayed_)*150, 
                                    400 - std::cos(total_yaw_rad_delayed_)*150),
                        cv::Scalar(255, 255, 0), 2);
                    cv::line(RMM_visualize_frame, 
                        cv::Point2f(400, 400), 
                        cv::Point2f(400 - std::sin(total_yaw_rad_delayed_filter_ -> getExponentialValue())*150, 
                                    400 - std::cos(total_yaw_rad_delayed_filter_ -> getExponentialValue())*150),
                        cv::Scalar(0, 255, 0), 2);
                    cv::putText(RMM_visualize_frame, 
                        "total_yaw:"+std::to_string(total_yaw_rad_delayed_), 
                        cv::Point2f(20,140), 
                        cv::FONT_HERSHEY_COMPLEX, 0.7, 
                        cv::Scalar(0, 255, 0), 1, 8, false);
                    cv::putText(RMM_visualize_frame, 
                        "total_yaw_filter:"+std::to_string(total_yaw_rad_delayed_filter_ -> getExponentialValue()), 
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
                    cv::imshow("RMM visualize", RMM_visualize_frame);
#endif
                    // ========================== RotationMotionModsel =========================== END
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
                    pitch_integration += ballistic_result.delta_pitch_rad * 0.1;
                    yaw_integration += ballistic_result.delta_yaw_rad * 0.02;

                    if (pitch_integration > 60.0 * M_PI / 180.0) {
                        pitch_integration = 60.0 * M_PI / 180.0;
                    }
                    if (pitch_integration < -60.0 * M_PI / 180.0) {
                        pitch_integration = -60.0 * M_PI / 180.0;
                    }

                    if (yaw_integration > 20.0 * M_PI / 180.0) {
                        yaw_integration = 20.0 * M_PI / 180.0;
                    }
                    if (yaw_integration < -20.0 * M_PI / 180.0) {
                        yaw_integration = -20.0 * M_PI / 180.0;
                    }
                    
                    // 发布云台控制命令
                    float command_pitch = last_pitch_rad_delayed_ + ballistic_result.delta_pitch_rad * 0.8 + pitch_integration + pitch_bias; // PI控制
                    float command_yaw = last_yaw_rad_delayed_ + ballistic_result.delta_yaw_rad * 1.0 + yaw_integration; // 缓解yaw轴输入数据掉线问题
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

                if (armor_class != ArmorType::Base) {
                    if (control_predictor_type == PredictorType::AutoSwitch) {
                        // 预测器切换
                        // EKF
                        cv::Point3f EKF_to_check(0.0, 0.0, 0.0);
                        {
                            Tracker::State future_state = EKF_tracker_->predictAhead(fps_counter -> avg_frame_time() * predictor_switcher_check_frames_);
                            double future_xc = future_state(0), future_yc = future_state(2), future_zc = future_state(4);
                            double future_yaw = future_state(6), future_r = future_state(8);
                            EKF_to_check = {
                                static_cast<float>(future_xc - future_r * sin(future_yaw)),
                                static_cast<float>(future_yc + future_r * cos(future_yaw)),
                                static_cast<float>(future_zc) 
                            };
                        }
                        // P3D
                        cv::Point3f P3D_to_check(0.0, 0.0, 0.0);
                        float P3D_period = 1.0;
                        if (predictor3dArmorPredictions.size() > 0) {
                            P3D_to_check = predictor3dArmorPredictions[std::min(predictor3dPrediction_nowIndex + predictor_switcher_check_frames_, static_cast<int>(predictor3dArmorPredictions.size())-1)];
                            P3D_period = predictor3d -> getFourierPeriod();
                        }
                        // RMM
                        cv::Point3f RMM_to_check(0.0, 0.0, 0.0);
                        float RMM_period = 1.0;
                        if (rotation_motion_model_) {
                            PredictResult RMM_pred_aim_data = rotation_motion_model_ -> predict(fps_counter -> avg_frame_time() * predictor_switcher_check_frames_);
                            std::vector<float> cam_position = rest_frame_ -> getCamPosition();
                            cv::Point2d cam_to_center_vector = {RMM_pred_aim_data.center_x - cam_position[0], RMM_pred_aim_data.center_y - cam_position[1]};
                            std::vector<double> center_v_dot_yaw(RMM_pred_aim_data.armors.size());
                            float yaw_bias = M_PI / 180.0 * 15.0;
                            yaw_bias *= static_cast<float>(RMM_pred_aim_data.rotation_direction);
                            for (int RMM_pred_aim_armor_i = 0; RMM_pred_aim_armor_i < RMM_pred_aim_data.armors.size(); RMM_pred_aim_armor_i += 1) {
                                SimpleArmor& RMM_pred_aim_armor = RMM_pred_aim_data.armors[RMM_pred_aim_armor_i];
                                cv::Point2d yaw_vector = {std::sin(RMM_pred_aim_armor.yaw + yaw_bias), -std::cos(RMM_pred_aim_armor.yaw + yaw_bias)};
                                center_v_dot_yaw[RMM_pred_aim_armor_i] = cam_to_center_vector.dot(yaw_vector);
                            }
                            int nearest_idx = std::distance(center_v_dot_yaw.begin(), std::min_element(center_v_dot_yaw.begin(), center_v_dot_yaw.end()));
                            RMM_to_check = {
                                static_cast<float>(RMM_pred_aim_data.armors[nearest_idx].x),
                                static_cast<float>(RMM_pred_aim_data.armors[nearest_idx].y),
                                static_cast<float>(RMM_pred_aim_data.armors[nearest_idx].z) 
                            };
                            // RMM_period = rotation_motion_model_ -> getJumpPeriod();
                        }

                        using_predictor_type = predictor_switcher_ -> step(true, rest_frame_pos, 
                            last_rest_frame_pos, 
                            EKF_to_check, 
                            P3D_to_check, 
                            RMM_to_check, 
                            P3D_period, 
                            RMM_period,
                            predictor3dCenterPredictions[predictor3dPrediction_nowIndex]);
                    } else {
                        using_predictor_type = control_predictor_type;
                    }
                }
            }
        }
    }

    if (!pnp_valid_flag) {
        if (EKF_tracker_->state != Tracker::LOST) {
            EKF_tracker_->predict();
        }

        if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_com_time).count() >= reset_com_time) {
            result.reset = true;
            // pitch_integration = 0.0; // 积分项重置
            yaw_integration = 0.0;
            predictor3d -> clearHistory(); 
            fire_data_predictor_ -> clearHistory();
            pred_fire_data_filter_ -> clearHistory();
            armor_distance_filter_ -> clearHistory();
            if (rotation_motion_model_) {
                RotationMotionState RMMstate = rotation_motion_model_ -> getState();
                if (RMMstate.update_frames > 90) init_r = (RMMstate.r_now, RMMstate.r_another) / 2.0;
                rotation_motion_model_.reset();
            }
            is_reset = true;
            predictor_switcher_ -> clearHistory();
            last_pixel_horizontal_center_distance = 1e10;
            has_valid_ballistic = false;
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

            if (armor_class != ArmorType::Base) {
                // ==========================FirePredictor===========================
                pred_fire_data_filter_ -> addPoint(0.0);
                size_t predictor3dPrediction_indexToAim = std::min(predictor3d_predict_step-1, predictor3dPrediction_nowIndex+(int)(last_total_delay_ * fps_counter->fps()));
                cv::Point3f predicted_armor_pos = predictor3dArmorPredictions[predictor3dPrediction_indexToAim];
                cv::Point3f predicted_aim_pos = predictor3dCenterPredictions[predictor3dPrediction_indexToAim];
                predictor3d -> addPoint(predicted_armor_pos);
                if (predictor3dPrediction_nowIndex < predictor3dArmorPredictions.size()-1) {
                    predictor3dPrediction_nowIndex += 1;
                }
                if (using_predictor_type == PredictorType::FirePredictor) {
                    fire_flag = pred_fire_data_filter_ -> getExponentialValue() > 0.5;
                }
                
                fire_data_predictor_ -> setPeriod(predictor3d->getFourierPeriod());
                //fire_data_predictor_ -> autoFindPeriod();
                fire_data_predictor_ -> addPoint(0.0);
                // ==========================FirePredictor=========================== END
                // ==========================RotationMotionModel===========================
                if (rotation_motion_model_) {
                    double RMM_update_time = (std::chrono::steady_clock::now() - node_start_time).count() / 1e9;
                    rotation_motion_model_ -> emptyUpdate(RMM_update_time);
                    
                    PredictResult RMM_pred_now_data = rotation_motion_model_ -> predict(0.0);

                    cv::Point3f RMM_pred_now_center_p3f = rest_frame_ -> worldToPnpP3f({
                        static_cast<float>(RMM_pred_now_data.center_x), 
                        static_cast<float>(RMM_pred_now_data.center_y), 
                        static_cast<float>(RMM_pred_now_data.center_z)}
                    );
                    cv::Point2f RMM_pred_now_center_pixel = armor_solver_->project3DToPixel(RMM_pred_now_center_p3f);
                    cv::circle(frame, RMM_pred_now_center_pixel, 10, cv::Scalar(255, 0, 255), 2);

                    cv::Mat RMM_visualize_frame = cv::Mat::zeros(800, 800, CV_8UC3);
                    cv::circle(RMM_visualize_frame, cv::Point2f(400+RMM_pred_now_data.center_x/10, 400-RMM_pred_now_data.center_y/10), 8, cv::Scalar(255, 0, 255), 2);
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
                        PredictResult RMM_pred_aim_data = rotation_motion_model_ -> predict(last_total_delay_);
                        std::vector<float> cam_position = rest_frame_ -> getCamPosition();
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
                        predicted_aim_pos = predicted_armor_pos;
                        float nearest_armor_yaw_bias = (nearest_armor.yaw - (rotation_motion_model_ -> getCamToCenterYaw())) * static_cast<float>(RMM_pred_aim_data.rotation_direction);
                        while (nearest_armor_yaw_bias < -M_PI) {
                            nearest_armor_yaw_bias += 2*M_PI;
                        }
                        while (nearest_armor_yaw_bias > M_PI) {
                            nearest_armor_yaw_bias -= 2*M_PI;
                        }
                        fire_flag = (nearest_armor_yaw_bias > -30.0 * M_PI / 180.0) && (nearest_armor_yaw_bias < 30.0 * M_PI / 180.0);
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
                    }
                    cv::line(RMM_visualize_frame, 
                        cv::Point2f(400, 400), 
                        cv::Point2f(400 - std::sin(total_yaw_rad_delayed_)*150, 
                                    400 - std::cos(total_yaw_rad_delayed_)*150),
                        cv::Scalar(255, 255, 0), 2);
                    cv::line(RMM_visualize_frame, 
                        cv::Point2f(400, 400), 
                        cv::Point2f(400 - std::sin(total_yaw_rad_delayed_filter_ -> getExponentialValue())*150, 
                                    400 - std::cos(total_yaw_rad_delayed_filter_ -> getExponentialValue())*150),
                        cv::Scalar(0, 255, 0), 2);
                    cv::putText(RMM_visualize_frame, 
                        "total_yaw:"+std::to_string(total_yaw_rad_delayed_), 
                        cv::Point2f(20,140), 
                        cv::FONT_HERSHEY_COMPLEX, 0.7, 
                        cv::Scalar(0, 255, 0), 1, 8, false);
                    cv::putText(RMM_visualize_frame, 
                        "total_yaw_filter:"+std::to_string(total_yaw_rad_delayed_filter_ -> getExponentialValue()), 
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
                    cv::imshow("RMM visualize", RMM_visualize_frame);
#endif
                }
                // ==========================RotationMotionModel=========================== END
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
                    yaw_integration += ballistic_result.delta_yaw_rad * 0.02;

                    if (pitch_integration > 60.0 * M_PI / 180.0) {
                        pitch_integration = 60.0 * M_PI / 180.0;
                    }
                    if (pitch_integration < -60.0 * M_PI / 180.0) {
                        pitch_integration = -60.0 * M_PI / 180.0;
                    }

                    if (yaw_integration > 20.0 * M_PI / 180.0) {
                        yaw_integration = 20.0 * M_PI / 180.0;
                    }
                    if (yaw_integration < -20.0 * M_PI / 180.0) {
                        yaw_integration = -20.0 * M_PI / 180.0;
                    }
                    
                    // 发布云台控制命令
                    float command_pitch = last_pitch_rad_delayed_ + ballistic_result.delta_pitch_rad * 0.8 + pitch_integration + pitch_bias; // PI控制
                    float command_yaw = last_yaw_rad_delayed_ + ballistic_result.delta_yaw_rad * 1.0 + yaw_integration; // 缓解yaw轴输入数据掉线问题
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
                }

                // 绘制瞄准预测点（黄色）
                cv::Point2f pred_aim_pixel = armor_solver_->project3DToPixel(predicted_aim_pos);
                cv::circle(frame, pred_aim_pixel, 8, cv::Scalar(0, 255, 255), 2);
                // 绘制装甲板预测点（天蓝色）
                cv::Point2f pred_armor_pixel = armor_solver_->project3DToPixel(predicted_armor_pos);
                cv::circle(frame, pred_armor_pixel, 8, cv::Scalar(255, 255, 0), 2);
            }
            if (result.fire_flag) {
                cv::circle(frame, last_aim_yaw_pitch_pixel_, 8, cv::Scalar(0, 0, 255), 2);
            } else {
                cv::circle(frame, last_aim_yaw_pitch_pixel_, 8, cv::Scalar(255, 255, 0), 2);
            }
        }
        oscilloscope_fire_ -> addDataPoint(result.fire_flag);

        if (armor_class != ArmorType::Base) {
            if (control_predictor_type == PredictorType::AutoSwitch) {
                // 预测器切换
                // EKF
                cv::Point3f EKF_to_check(0.0, 0.0, 0.0);
                {
                    Tracker::State future_state = EKF_tracker_->predictAhead(fps_counter -> avg_frame_time() * predictor_switcher_check_frames_);
                    double future_xc = future_state(0), future_yc = future_state(2), future_zc = future_state(4);
                    double future_yaw = future_state(6), future_r = future_state(8);
                    EKF_to_check = {
                        static_cast<float>(future_xc - future_r * sin(future_yaw)),
                        static_cast<float>(future_yc + future_r * cos(future_yaw)),
                        static_cast<float>(future_zc) 
                    };
                }
                // P3D
                cv::Point3f P3D_to_check(0.0, 0.0, 0.0);
                float P3D_period = 1.0;
                if (predictor3dArmorPredictions.size() > 0) {
                    P3D_to_check = predictor3dArmorPredictions[std::min(predictor3dPrediction_nowIndex + predictor_switcher_check_frames_, static_cast<int>(predictor3dArmorPredictions.size())-1)];
                    P3D_period = predictor3d -> getFourierPeriod();
                }
                // RMM
                cv::Point3f RMM_to_check(0.0, 0.0, 0.0);
                float RMM_period = 1.0;
                if (rotation_motion_model_) {
                    PredictResult RMM_pred_aim_data = rotation_motion_model_ -> predict(fps_counter -> avg_frame_time() * predictor_switcher_check_frames_);
                    std::vector<float> cam_position = rest_frame_ -> getCamPosition();
                    cv::Point2d cam_to_center_vector = {RMM_pred_aim_data.center_x - cam_position[0], RMM_pred_aim_data.center_y - cam_position[1]};
                    std::vector<double> center_v_dot_yaw(RMM_pred_aim_data.armors.size());
                    float yaw_bias = M_PI / 180.0 * 15.0;
                    yaw_bias *= static_cast<float>(RMM_pred_aim_data.rotation_direction);
                    for (int RMM_pred_aim_armor_i = 0; RMM_pred_aim_armor_i < RMM_pred_aim_data.armors.size(); RMM_pred_aim_armor_i += 1) {
                        SimpleArmor& RMM_pred_aim_armor = RMM_pred_aim_data.armors[RMM_pred_aim_armor_i];
                        cv::Point2d yaw_vector = {std::sin(RMM_pred_aim_armor.yaw + yaw_bias), -std::cos(RMM_pred_aim_armor.yaw + yaw_bias)};
                        center_v_dot_yaw[RMM_pred_aim_armor_i] = cam_to_center_vector.dot(yaw_vector);
                    }
                    int nearest_idx = std::distance(center_v_dot_yaw.begin(), std::min_element(center_v_dot_yaw.begin(), center_v_dot_yaw.end()));
                    RMM_to_check = {
                        static_cast<float>(RMM_pred_aim_data.armors[nearest_idx].x),
                        static_cast<float>(RMM_pred_aim_data.armors[nearest_idx].y),
                        static_cast<float>(RMM_pred_aim_data.armors[nearest_idx].z) 
                    };
                    // RMM_period = rotation_motion_model_ -> getJumpPeriod();
                }

                using_predictor_type = predictor_switcher_ -> step(false, cv::Point3f(0.0, 0.0, 0.0), 
                    last_rest_frame_pos, 
                    EKF_to_check, 
                    P3D_to_check, 
                    RMM_to_check, 
                    P3D_period, 
                    RMM_period,
                    predictor3dCenterPredictions[predictor3dPrediction_nowIndex]);
            } else {
                using_predictor_type = control_predictor_type;
            }
        }
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
