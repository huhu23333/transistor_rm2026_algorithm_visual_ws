// YOLOPoseArmorDetector.cpp
#include "2d_armor_detector/YOLOPoseArmorDetector.h"

std::vector<Armor> YOLOPoseArmorDetector::detectArmors(const cv::Mat& frame, bool block, int now_history_frame_identifier) {
    std::vector<Armor> result;

    std::vector<DetectionResult> yolo_pose_detection_results = shm_python_yolo_pose -> processImage(frame, block, now_history_frame_identifier);

    for (DetectionResult& yolo_pose_detection_result : yolo_pose_detection_results) {
        if (yolo_pose_detection_result.confidence > 0.0) {
            std::vector<float> frame_keypoints = {
                yolo_pose_detection_result.keypoints[0] * static_cast<float>(frame.cols),
                yolo_pose_detection_result.keypoints[1] * static_cast<float>(frame.rows),
                yolo_pose_detection_result.keypoints[2] * static_cast<float>(frame.cols),
                yolo_pose_detection_result.keypoints[3] * static_cast<float>(frame.rows),
                yolo_pose_detection_result.keypoints[4] * static_cast<float>(frame.cols),
                yolo_pose_detection_result.keypoints[5] * static_cast<float>(frame.rows),
                yolo_pose_detection_result.keypoints[6] * static_cast<float>(frame.cols),
                yolo_pose_detection_result.keypoints[7] * static_cast<float>(frame.rows),
            };

            RCLCPP_DEBUG(node->get_logger(), "scaled_yolo_data: %f, %f, %f, %f, %f, %f, %f, %f",
                frame_keypoints[0], frame_keypoints[1], frame_keypoints[2], frame_keypoints[3], 
                frame_keypoints[4], frame_keypoints[5], frame_keypoints[6], frame_keypoints[7]);
            cv::Vec2f leftLightBar_lengthVec((frame_keypoints[0] - frame_keypoints[2]), (frame_keypoints[1] - frame_keypoints[3]));
            cv::Vec2f rightLightBar_lengthVec((frame_keypoints[6] - frame_keypoints[4]), (frame_keypoints[7] - frame_keypoints[5]));
            cv::Point2f leftLightBar_center((frame_keypoints[0] + frame_keypoints[2]) / 2.0, (frame_keypoints[1] + frame_keypoints[3]) / 2.0);
            cv::Point2f rightLightBar_center((frame_keypoints[4] + frame_keypoints[6]) / 2.0, (frame_keypoints[5] + frame_keypoints[7]) / 2.0);
            float leftLightBar_length = cv::norm(leftLightBar_lengthVec);
            float rightLightBar_length = cv::norm(rightLightBar_lengthVec);
            cv::Size2f leftLightBar_size(leftLightBar_length * lightBarLengthScale / 8.0, leftLightBar_length * lightBarLengthScale);
            cv::Size2f rightLightBar_size(rightLightBar_length * lightBarLengthScale / 8.0, rightLightBar_length * lightBarLengthScale);
            float leftLightBar_angle = std::atan2(leftLightBar_lengthVec[1], leftLightBar_lengthVec[0]) * 180.0 / M_PI + 90.0;
            float rightLightBar_angle = std::atan2(rightLightBar_lengthVec[1], rightLightBar_lengthVec[0]) * 180.0 / M_PI + 90.0;

            cv::RotatedRect leftLightBar(leftLightBar_center, leftLightBar_size, leftLightBar_angle);
            cv::RotatedRect rightLightBar(rightLightBar_center, rightLightBar_size, rightLightBar_angle);
            result.emplace_back(leftLightBar, rightLightBar, config_file_ptr, node, yolo_pose_detection_result.history_frame_identifier);
        }
    }

    return result;
}

void YOLOPoseArmorDetector::setEnemyColor(Params::EnemyColor enemy_color) {
    if (shm_python_yolo_pose) {
        shm_python_yolo_pose -> setEnemyColor(enemy_color);
    }
}