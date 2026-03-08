#include "controller/RebirthTargetFilter.h"


RebirthTargetFilter::RebirthTargetFilter() {
    last_dead_times.resize(16);
    all_car_rebirth_infos.resize(32);
    
    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point init_time = current_time - std::chrono::milliseconds(static_cast<int>(invincible_time * 1e3));
    for (int i=0; i<16; i++) {
        last_dead_times[i] = init_time;
    }
}


void RebirthTargetFilter::updateInfos(uint8_t enemy_color_, std::vector<uint8_t> all_car_rebirth_infos_bits) {
    enemy_color = enemy_color_;

    for (int i=0; i<4; i++) {
        for (int j=0; j<8; j++) {
            all_car_rebirth_infos[i * 8 + j] = (((all_car_rebirth_infos_bits[i] >> j) & 0x01) == 1);
        }
    }

    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    for (int i=0; i<16; i++) {
        if (all_car_rebirth_infos[i] == false)
        {
            last_dead_times[i] = current_time;
        }
    }
}

std::string vector_bool_to_bitset_string(const std::vector<bool>& bits) {
    std::string result;
    result.reserve(bits.size());                 // 预分配空间
    for (auto it = bits.rbegin(); it != bits.rend(); ++it) {
        result += *it ? '1' : '0';
    }
    return result;
}
std::vector<bool> slice(const std::vector<bool>& v, size_t start, size_t end) {
    if (start > end || end > v.size()) throw std::out_of_range("Invalid slice indices");
    return std::vector<bool>(v.begin() + start, v.begin() + end);
}

std::vector<bool> RebirthTargetFilter::getValidTargetMask(cv::Mat &frame) {
    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    
    int index_shift = enemy_color * 8;

    std::vector<bool> result(8);

    for (int i=0; i<8; i++) {
        if (static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_dead_times[i + index_shift]).count()) / 1e3 > invincible_time) {
            result[i] = true;
        } else {
            if (all_car_rebirth_infos[i + index_shift + 16]) {
                result[i] = true;
            } else {
                result[i] = false;
            }
        }
    }

    cv::putText(frame, 
        "Mask Infos:", 
        cv::Point2f(1100,840),
        cv::FONT_HERSHEY_COMPLEX, 0.7, 
        cv::Scalar(0, 255, 0), 1, 8, false);
    cv::putText(frame, 
        vector_bool_to_bitset_string(result), 
        cv::Point2f(1100,870),
        cv::FONT_HERSHEY_COMPLEX, 0.7, 
        cv::Scalar(0, 255, 0), 1, 8, false);

    cv::putText(frame, 
        vector_bool_to_bitset_string(slice(all_car_rebirth_infos, 0, 8)), 
        cv::Point2f(1100,930),
        cv::FONT_HERSHEY_COMPLEX, 0.7, 
        cv::Scalar(0, 255, 0), 1, 8, false);
    cv::putText(frame, 
        vector_bool_to_bitset_string(slice(all_car_rebirth_infos, 8, 16)), 
        cv::Point2f(1100,960),
        cv::FONT_HERSHEY_COMPLEX, 0.7, 
        cv::Scalar(0, 255, 0), 1, 8, false);
    cv::putText(frame, 
        vector_bool_to_bitset_string(slice(all_car_rebirth_infos, 16, 24)), 
        cv::Point2f(1100,990),
        cv::FONT_HERSHEY_COMPLEX, 0.7, 
        cv::Scalar(0, 255, 0), 1, 8, false);
    cv::putText(frame, 
        vector_bool_to_bitset_string(slice(all_car_rebirth_infos, 24, 32)), 
        cv::Point2f(1100,1020),
        cv::FONT_HERSHEY_COMPLEX, 0.7, 
        cv::Scalar(0, 255, 0), 1, 8, false);

    return result;
}