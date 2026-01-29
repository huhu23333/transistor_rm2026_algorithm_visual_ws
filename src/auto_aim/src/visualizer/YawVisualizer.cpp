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


    float total_current_yaw = current_yaw + 2 * M_PI * current_yaw_circle;
    float total_target_yaw = target_yaw + 2 * M_PI * target_yaw_circle;



    bool now_target_yaw_raise_cross_mid = false;
    bool now_current_yaw_raise_cross_mid = false;
    current_yaw_history.push_back(current_yaw);
    target_yaw_history.push_back(target_yaw);
    if (current_yaw_history.size() > 90) {
        current_yaw_history = std::vector<float>(
            current_yaw_history.end() - 90, current_yaw_history.end());
        target_yaw_history = std::vector<float>(
            target_yaw_history.end() - 90, target_yaw_history.end());
    }
    if (current_yaw_history.size() > 30) {
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        float max_target_yaw = *std::max_element(target_yaw_history.begin(), target_yaw_history.end());
        float min_target_yaw = *std::min_element(target_yaw_history.begin(), target_yaw_history.end());
        float mid_target_yaw = (max_target_yaw + min_target_yaw) / 2.0;


        bool target_yaw_raise_cross_mid = 
            ((last_target_yaw < mid_target_yaw) ^ raise_direction) && ((target_yaw >= mid_target_yaw) ^ raise_direction);
        if (target_yaw_raise_cross_mid) {
            last_target_mid_time = now;
            now_target_yaw_raise_cross_mid = false;
        }

        bool current_yaw_raise_cross_mid = 
            ((last_current_yaw < mid_target_yaw) ^ raise_direction) && ((current_yaw >= mid_target_yaw) ^ raise_direction);
        if (current_yaw_raise_cross_mid) {
            if (
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last_target_mid_time).count() < max_delay
            ) {
                delay_history.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(now - last_target_mid_time).count());
                last_target_mid_time = last_target_mid_time - std::chrono::milliseconds(max_delay);
                now_current_yaw_raise_cross_mid = false;
            }
        }
    }
    if (delay_history.size() > 90) {
        delay_history = std::vector<float>(
            delay_history.end() - 90, delay_history.end());
    }

    yaw_oscilloscope -> addDataPoint(total_target_yaw, 0, now_target_yaw_raise_cross_mid ? 5 : 1);
    yaw_oscilloscope -> addDataPoint(total_current_yaw, 1, now_current_yaw_raise_cross_mid ? 5 : 1);
    yaw_oscilloscope -> update();
    yaw_oscilloscope -> putText(cv::format("total_target_yaw: %f", total_target_yaw), 
        cv::Point(20, 50), cv::Scalar(0, 255, 0), 0.7);
    yaw_oscilloscope -> putText(cv::format("total_current_yaw: %f", total_current_yaw), 
        cv::Point(20, 100), cv::Scalar(0, 0, 255), 0.7);
    if(delay_history.size()) {
        yaw_oscilloscope -> putText(cv::format("average delay: %f ms", std::accumulate(delay_history.begin(), delay_history.end(), 0.0) / static_cast<float>(delay_history.size())), 
            cv::Point(20, 150), cv::Scalar(255, 255, 255), 0.7);
    }

    display = yaw_oscilloscope -> getDisplay();

    cv::line(display, cv::Point(400, 400), cv::Point(
        400 - std::sin(total_target_yaw) * 100,
        400 - std::cos(total_target_yaw) * 100
    ), cv::Scalar(0, 255, 0), 2);
    cv::line(display, cv::Point(400, 400), cv::Point(
        400 - std::sin(total_current_yaw) * 100,
        400 - std::cos(total_current_yaw) * 100
    ), cv::Scalar(0, 0, 255), 2);


    last_current_yaw = current_yaw;
    last_target_yaw = target_yaw;
}

void YawVisualizer::show() {
#ifdef SHOW_WINDOWS
    cv::imshow("Yaw Visualizer", display);
    cv::waitKey(1);
#endif
}
