#ifndef PARAMS_H
#define PARAMS_H

// 枚举类型，表示敌方颜色
struct Params {
    enum EnemyColor {
        BLUE,   // 蓝色
        RED,    // 红色
        GREEN   // 绿色 (作为默认)
    };

    // 默认的敌方颜色设置为蓝色
    EnemyColor enemy_color ;

    // 灯条检测参数
    int min_light_height;
    int light_slope_offset;
    int light_min_area;
    float max_light_wh_ratio;
    float min_light_wh_ratio;
    float light_max_tilt_angle;
    int min_light_delta_x;
    float min_light_dx_ratio;
    float max_light_dy_ratio;
    float max_light_delta_angle;
    int near_face_v;
    float max_lr_rate;
    float max_wh_ratio;
    float min_wh_ratio;
    float small_armor_wh_threshold;
    int bin_cls_thres;
    int target_max_angle;
    float goodToTotalRatio;
    int matchDistThre;
    float wh_ratio_threshold;
    float wh_ratio_max;
    int M_YAW_THRES;
    float K_YAW_THRES;
    int MAX_DETECT_CNT;
    int MAX_LOST_CNT;
};

#endif // PARAMS_H
