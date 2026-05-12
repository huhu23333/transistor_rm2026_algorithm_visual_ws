// RestFrameDraw.cpp
#include "visualizer/RestFrameDraw.h"

void drawRestFrame(cv::Mat& image, std::shared_ptr<RestFrame> rest_frame, std::shared_ptr<ArmorSolver> armor_solver) {
    for (float x = -3000.0; x <= 3000.0; x += 200.0) {
        for (float y = -3000.0; y <= 3000.0; y += 200.0) {
            for (double z = -500.0f; z <= 500.0f; z += 1000.0f) {

                cv::Point3f world_point = cv::Point3f(x, y, z);
                cv::Point3f cam_point = rest_frame -> worldToPnpP3f(world_point);
                
                if (cam_point.z > 0.0f) {
                    // 投影到图像平面
                    
                    cv::Point2d img_point = armor_solver -> project3DToPixel(cam_point);

                    float max_fov_angle = armor_solver -> getMaxFOVAngle(image.cols, image.rows);
                    
                    // 检查点是否在图像范围内
                    if (img_point.x >= 0 && img_point.x < image.cols && 
                        img_point.y >= 0 && img_point.y < image.rows) {

                        if (std::atan2(std::sqrt(cam_point.x * cam_point.x + cam_point.y * cam_point.y), cam_point.z) > max_fov_angle / 2.0f) {
                            continue; // 超出最大视场角，跳过绘制
                        }

                        cv::circle(image, img_point, 3, cv::Scalar(200, 200, 200), -1);
                        // cv::putText(image, "(" + std::to_string((int)x) + "," + std::to_string((int)y) + "," + std::to_string((int)z) + ")", 
                        //             img_point + cv::Point2d(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
                    }
                }
            }
        }
    }
}
