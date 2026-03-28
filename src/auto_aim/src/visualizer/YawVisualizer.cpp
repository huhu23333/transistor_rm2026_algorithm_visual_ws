#include "visualizer/YawVisualizer.h"

struct ScaleOffset
{
    float scale;
    float offset;
};

ScaleOffset getScaleOffsetFromMinMax(float min_value, float max_value) {
    float mid_value = (min_value + max_value) / 2.0f;
    float range = max_value - min_value;
    ScaleOffset result;
    result.scale = 2.0f / range;
    result.offset = - mid_value * result.scale;
    return result;
}

void YawVisualizer::adjustScaleOffset() {
    float min_value = std::min(
        *std::min_element(total_current_yaw_history_for_adjust_scale_offset.begin() ,total_current_yaw_history_for_adjust_scale_offset.end()),
        *std::min_element(total_target_yaw_history_for_adjust_scale_offset.begin() ,total_target_yaw_history_for_adjust_scale_offset.end())
    );
    float max_value = std::max(
        *std::max_element(total_current_yaw_history_for_adjust_scale_offset.begin() ,total_current_yaw_history_for_adjust_scale_offset.end()),
        *std::max_element(total_target_yaw_history_for_adjust_scale_offset.begin() ,total_target_yaw_history_for_adjust_scale_offset.end())
    );

    ScaleOffset scale_offset = getScaleOffsetFromMinMax(min_value, max_value);

    yaw_oscilloscope -> setScale(scale_offset.scale);
    yaw_oscilloscope -> setOffset(scale_offset.offset);
}


YawVisualizer::YawVisualizer() {
    yaw_oscilloscope = std::make_shared<Oscilloscope>(800, 800, "yaw_oscilloscope", 2, cv::Scalar(0, 0, 0), cv::Scalar(0, 255, 0));
    yaw_oscilloscope -> setScale(0.5);
    yaw_oscilloscope -> setOffset(0.0);
    yaw_oscilloscope -> setRollingSpeed(5);
    yaw_oscilloscope -> setLayerColor(1, cv::Scalar(0, 0, 255));

    total_current_yaw_history_for_adjust_scale_offset.push_back(0.0f);
    total_target_yaw_history_for_adjust_scale_offset.push_back(0.0f);
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

    total_current_yaw_history_for_adjust_scale_offset.push_back(total_current_yaw);
    total_target_yaw_history_for_adjust_scale_offset.push_back(total_target_yaw);
    if (total_current_yaw_history_for_adjust_scale_offset.size() > 160) {
        total_current_yaw_history_for_adjust_scale_offset.pop_front();
        total_target_yaw_history_for_adjust_scale_offset.pop_front();
    }
    adjustScaleOffset();


    bool now_total_target_yaw_raise_cross_mid = false;
    bool now_total_current_yaw_raise_cross_mid = false;
    total_current_yaw_history.push_back(total_current_yaw);
    total_target_yaw_history.push_back(total_target_yaw);
    if (total_current_yaw_history.size() > 90) {
        total_current_yaw_history = std::vector<float>(
            total_current_yaw_history.end() - 90, total_current_yaw_history.end());
        total_target_yaw_history = std::vector<float>(
            total_target_yaw_history.end() - 90, total_target_yaw_history.end());
    }
    if (total_current_yaw_history.size() > 30) {
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        float max_total_target_yaw = *std::max_element(total_target_yaw_history.begin(), total_target_yaw_history.end());
        float min_total_target_yaw = *std::min_element(total_target_yaw_history.begin(), total_target_yaw_history.end());
        float mid_total_target_yaw = (max_total_target_yaw + min_total_target_yaw) / 2.0;


        float last_total_target_yaw = total_target_yaw_history[total_target_yaw_history.size()-2];
        float last_total_current_yaw = total_current_yaw_history[total_current_yaw_history.size()-2];

        bool total_target_yaw_raise_cross_mid = 
            ((last_total_target_yaw < mid_total_target_yaw) ^ raise_direction) && ((total_target_yaw >= mid_total_target_yaw) ^ raise_direction);
        if (total_target_yaw_raise_cross_mid) {
            last_total_target_mid_time = now;
            now_total_target_yaw_raise_cross_mid = true;
        }

        bool total_current_yaw_raise_cross_mid = 
            ((last_total_current_yaw < mid_total_target_yaw) ^ raise_direction) && ((total_current_yaw >= mid_total_target_yaw) ^ raise_direction);
        if (total_current_yaw_raise_cross_mid) {
            if (
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last_total_target_mid_time).count() < max_delay
            ) {
                delay_history.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(now - last_total_target_mid_time).count());
                last_total_target_mid_time = last_total_target_mid_time - std::chrono::milliseconds(max_delay);
                now_total_current_yaw_raise_cross_mid = true;
            }
        }
    }
    if (delay_history.size() > 90) {
        delay_history = std::vector<float>(
            delay_history.end() - 90, delay_history.end());
    }

    yaw_oscilloscope -> addDataPoint(total_target_yaw, 0, now_total_target_yaw_raise_cross_mid ? 5 : 1);
    yaw_oscilloscope -> addDataPoint(total_current_yaw, 1, now_total_current_yaw_raise_cross_mid ? 5 : 1);
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

cv::Mat YawVisualizer::getDisplay() {
    return display;
}