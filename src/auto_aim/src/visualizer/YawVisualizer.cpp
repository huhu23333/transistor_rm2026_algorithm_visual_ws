#include "visualizer/YawVisualizer.h"

YawVisualizer::YawVisualizer() {
    yaw_oscilloscope = std::make_shared<Oscilloscope>(800, 800, "yaw_oscilloscope", 2, cv::Scalar(0, 0, 0), cv::Scalar(0, 255, 0));
    yaw_oscilloscope -> setScale(1.0);
    yaw_oscilloscope -> setOffset(0.0);
    yaw_oscilloscope -> setRollingSpeed(5);
    yaw_oscilloscope -> setLayerColor(1, cv::Scalar(0, 0, 255));
}

void YawVisualizer::update(float current_yaw, float target_yaw) {
    display.setTo(cv::Scalar(0, 0, 0));

    current_yaw = std::atan2(std::sin(current_yaw), std::cos(current_yaw));
    target_yaw = std::atan2(std::sin(target_yaw), std::cos(target_yaw));


    if (current_yaw < -M_PI/2 && last_current_yaw > M_PI/2) {
        current_yaw_circle += 1;
    } else if (current_yaw > M_PI/2 && last_current_yaw < -M_PI/2) {
        current_yaw_circle -= 1;
    }
    if (target_yaw < -M_PI/2 && last_target_yaw > M_PI/2) {
        target_yaw_circle += 1;
    } else if (target_yaw > M_PI/2 && last_target_yaw < -M_PI/2) {
        target_yaw_circle -= 1;
    }

    last_current_yaw = current_yaw;
    last_target_yaw = target_yaw;

    float total_current_yaw = current_yaw + 2 * M_PI * current_yaw_circle;
    float total_target_yaw = target_yaw + 2 * M_PI * target_yaw_circle;

    yaw_oscilloscope -> addDataPoint(total_target_yaw, 0);
    yaw_oscilloscope -> addDataPoint(total_current_yaw, 1);
    yaw_oscilloscope -> update();
    yaw_oscilloscope -> putText(cv::format("total_target_yaw: %f", total_target_yaw), 
        cv::Point(20, 50), cv::Scalar(0, 255, 0), 0.7);
    yaw_oscilloscope -> putText(cv::format("total_current_yaw: %f", total_current_yaw), 
        cv::Point(20, 70), cv::Scalar(0, 0, 255), 0.7);

    display = yaw_oscilloscope -> getDisplay();

    cv::line(display, cv::Point(400, 400), cv::Point(
        400 - std::sin(total_target_yaw) * 100,
        400 - std::cos(total_target_yaw) * 100
    ), cv::Scalar(0, 255, 0), 2);
    cv::line(display, cv::Point(400, 400), cv::Point(
        400 - std::sin(total_current_yaw) * 100,
        400 - std::cos(total_current_yaw) * 100
    ), cv::Scalar(0, 0, 255), 2);
}

void YawVisualizer::show() {
#ifdef SHOW_WINDOWS
    cv::imshow("Yaw Visualizer", display);
    cv::waitKey(1);
#endif
}
