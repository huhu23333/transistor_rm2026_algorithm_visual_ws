#pragma once
#include "utils/PeriodFunctions.h"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/SVD>



struct ObservedData {
    double x;
    double y;
    double z;
    double yaw;
    double t;
    
    ObservedData(double x_val, double y_val, double z_val, double yaw_val, double t_val)
        : x(x_val), y(y_val), z(z_val), yaw(yaw_val), t(t_val) {}
};

struct fitCenterXYResult {
    double center_x;
    double center_y;
    double center_vx;
    double center_vy;
};

struct SimpleArmor {
    double x;
    double y;
    double z;
    double yaw;
};

struct PredictResult {
    double center_x;
    double center_y;
    double center_z;
    double r;
    double yaw;
    int rotation_direction;
    std::vector<SimpleArmor> armors;
};

struct RotationMotionState {
    double center_vx;
    double center_vy;
    double center_vz;
    double vyaw;
    double r;
    double center_x;
    double center_y;
    double center_z;
};

class RotationMotionModel {
private:

    std::vector<ObservedData> observedDataHistory;
    double center_vx;
    double center_vy;
    double center_vz;
    double vyaw;
    double r;
    double center_x;
    double center_y;
    double center_z;
    int max_history;
    int refine_multiple;
    double jump_period_frames = 1.0;
    double rotation_period;
    double current_phase;
    int n_armors;
    int rotation_direction;
    double jump_rad;
    double delta_phase;

    // Private methods
    std::vector<std::vector<double>> getParams();
    void calculateR();
    void fitRotationParameters();
    std::vector<int> findMidYaw(const std::vector<double>& yawData, double periodFrames, 
                               std::vector<double>& dataToFit, std::vector<double>& fittedData);
    int getRotationDirection(const std::vector<double>& yawData);
    double centerResiduals(const std::vector<double>& params, double armorYaw, double armorX, 
                          double armorY, double dataT);

public:

    RotationMotionModel(const ObservedData& initObservedData);
    void update(const ObservedData& observedData);
    PredictResult predict(double predictTime);
    fitCenterXYResult fitCenterXY(const std::vector<double>& armorYaw, 
                                 const std::vector<double>& armorX,
                                 const std::vector<double>& armorY,
                                 const std::vector<double>& dataT,
                                 double lastR,
                                 const std::vector<double>& weightT = {},
                                 double alpha = 1e-10,
                                 double tikhonovLambda = 1e-12);
    double getJumpPeriod();
    void emptyUpdate(double update_time);
    double getTheoreticYaw(double armor_x, double armor_y);
    RotationMotionState getState();
};