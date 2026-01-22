#include "predictor/PredictorSwitcher.h"

namespace PredictorType {
    std::vector<std::string> PredictorTypeStrings = {
        "None",
        "RMM",
        "EKF",
        "AutoSwitch(should not be used)"
    };
}

void PredictorSwitcher::clearHistory() {
}


PredictorType::PredictorType PredictorSwitcher::step() {
    return PredictorType::RotationMotionModel; 
}