#include "controller/RebirthTargetFilter.h"


RebirthTargetFilter::RebirthTargetFilter() {
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

std::vector<bool> RebirthTargetFilter::getValidTargetMask() {
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

    return result;
}