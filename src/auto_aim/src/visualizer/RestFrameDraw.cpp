// RestFrameDraw.cpp
#include "visualizer/RestFrameDraw.h"

void drawRestFrame(cv::Mat image, std::shared_ptr<RestFrame> rest_frame, std::shared_ptr<ArmorSolver> armor_solver) {
    for (float x = -3000.0; x <= 3000.0; x += 200.0) {
        for (float y = -3000.0; y <= 3000.0; y += 200.0) {
            float z = -500.0;

            cv::Point3f world_point = cv::Point3f(x, y, z);
            cv::Point3f cam_point = rest_frame -> worldToPnpP3f(world_point);
            cv::Point2f img_point = armor_solver -> project3DToPixel(cam_point);

            cv::circle(image, img_point, 3, cv::Scalar(200, 200, 200), -1);
        }
    }
}
