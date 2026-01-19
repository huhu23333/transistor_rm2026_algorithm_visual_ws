#ifndef PREDICTOR_SWITCHER_H
#define PREDICTOR_SWITCHER_H
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <deque>

#include "utils/DataProcessFuncs.h"
#include "3d_processing/RestFrame.h"
// #include "predictor/RotationJudge.h"

namespace PredictorType {
    enum PredictorType {
        None = 0,   // 直接瞄准装甲板
        RotationMotionModel,
        AutoSwitch
    };

    extern std::vector<std::string> PredictorTypeStrings;
}

class PredictorSwitcher {
public:
    PredictorSwitcher(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node) 
    : config_file_ptr(config_file_ptr), node(node){

    }

    PredictorType::PredictorType step();
    void clearHistory();
    
private:
    std::shared_ptr<YAML::Node> config_file_ptr; 
    rclcpp::Node* node;
};

#endif