// RebirthTargetFilter.h
#ifndef REBIRTH_TARGET_FILTER_H
#define REBIRTH_TARGET_FILTER_H

#include <memory>
#include <vector>
#include <chrono>
#include "macro/AutoAimMacro.h"


class RebirthTargetFilter {
private:
    float invincible_time = 30.0;
    std::vector<std::chrono::steady_clock::time_point> last_dead_times;

    uint8_t enemy_color;
    std::vector<bool> all_car_rebirth_infos;

public:
    RebirthTargetFilter();

    void updateInfos(uint8_t enemy_color_, std::vector<uint8_t> all_car_rebirth_infos_bits);
    std::vector<bool> getValidTargetMask();
};


#endif



