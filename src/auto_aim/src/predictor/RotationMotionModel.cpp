#include "predictor/RotationMotionModel.h"

using namespace Eigen;

RotationMotionModel::RotationMotionModel(ObservedData& initObservedData, std::shared_ptr<RestFrame> rest_frame_, bool is_outpost)
    : rest_frame_(rest_frame_), is_outpost(is_outpost) {
    initObservedData.theoreticYaw = getTheoreticYaw(initObservedData.x, initObservedData.y);
    observedDataHistory.push_back(initObservedData);
    center_vx = 0.0;
    center_vy = 0.0;
    center_vz = 0.0;
    vyaw = 0.0;
    center_x = initObservedData.x - r * sin(initObservedData.yaw);
    center_y = initObservedData.y + r * cos(initObservedData.yaw);
    center_z = initObservedData.z;
    max_history = 90;
    refine_multiple = 30;
    rotation_period = 0.0;
    current_phase = 0.0;
    if (is_outpost) {
        n_armors = 3;
        r = 276.5;
        delta_phase = 25.0 * M_PI / 180.0;
    } else {
        n_armors = 4;
        r = 250.0;
        delta_phase = 0.0 * M_PI / 180.0; // todo
    }
    rotation_direction = 1;
    jump_rad = M_PI * 2.0 / n_armors;
}

void RotationMotionModel::emptyUpdate(double update_time) {
    ObservedData last_observed_data = observedDataHistory.back();
    PredictResult pred_data_to_update = predict(update_time - last_observed_data.t);
    ObservedData update_data({
        pred_data_to_update.armors[0].x,
        pred_data_to_update.armors[0].y,
        pred_data_to_update.armors[0].z,
        pred_data_to_update.armors[0].yaw,
        update_time
    });
    update(update_data);
}


void RotationMotionModel::update(ObservedData& observedData) {
    observedData.theoreticYaw = getTheoreticYaw(observedData.x, observedData.y);
    observedDataHistory.push_back(observedData);
    if (observedDataHistory.size() > max_history) {
        observedDataHistory = std::vector<ObservedData>(
            observedDataHistory.end() - max_history, observedDataHistory.end());
    }
    
    auto params = getParams();
    std::vector<double> tData = params[0];
    std::vector<double> xData = params[1];
    std::vector<double> yData = params[2];
    std::vector<double> zData = params[3];
    std::vector<double> yawData = params[4];
    
    LinearRegressionResult zResult = linearRegression(tData, zData);
    center_z = zResult.a;
    center_vz = zResult.b;
    
    fitCenterXYResult centerResult = fitCenterXY(yawData, xData, yData, tData, r);
    center_x = centerResult.center_x;
    center_y = centerResult.center_y;
    center_vx = centerResult.center_vx;
    center_vy = centerResult.center_vy;
    

    if (!is_outpost) {
        calculateR();
    }
    fitRotationParameters();
}

std::vector<std::vector<double>> RotationMotionModel::getParams() {
    std::vector<double> tData, xData, yData, zData, yawData;
    double lastT = observedDataHistory.back().t;
    
    for (const auto& data : observedDataHistory) {
        tData.push_back(data.t - lastT);
        xData.push_back(data.x);
        yData.push_back(data.y);
        zData.push_back(data.z);
        yawData.push_back(data.yaw);
    }
    
    return {tData, xData, yData, zData, yawData};
}

void RotationMotionModel::calculateR() {
    auto params = getParams();
    std::vector<double> tData = params[0];
    std::vector<double> xData = params[1];
    std::vector<double> yData = params[2];
    
    double sum = 0.0;
    int n = tData.size();
    for (int i = 0; i < n; i++) {
        double dx = xData[i] - (center_x + center_vx * tData[i]);
        double dy = yData[i] - (center_y + center_vy * tData[i]);
        sum += sqrt(dx * dx + dy * dy);
    }
    
    r = sum / n;
    r = std::max(std::min(r, 600.0), 100.0);
}

fitCenterXYResult RotationMotionModel::fitCenterXY(const std::vector<double>& armorYaw,
                                                  const std::vector<double>& armorX,
                                                  const std::vector<double>& armorY,
                                                  const std::vector<double>& dataT,
                                                  double lastR,
                                                  const std::vector<double>& weightT,
                                                  double alpha,
                                                  double tikhonovLambda) {
    int n = armorYaw.size();
    std::vector<double> localWeightT = weightT;
    
    if (localWeightT.empty()) {
        localWeightT.resize(n, 1.0);
    }
    
    for (int i = 0; i < n; i++) {
        localWeightT[i] = std::max(localWeightT[i], 0.0);
    }
    
    std::vector<double> sqrtWeights(n);
    for (int i = 0; i < n; i++) {
        sqrtWeights[i] = sqrt(localWeightT[i]);
    }
    
    // Build axis vectors
    std::vector<double> cosYaw(n), sinYaw(n);
    std::vector<double> axisX(n), axisY(n);
    std::vector<double> offAxisX(n), offAxisY(n);
    
    for (int i = 0; i < n; i++) {
        cosYaw[i] = cos(armorYaw[i]);
        sinYaw[i] = sin(armorYaw[i]);
        axisX[i] = -sinYaw[i];
        axisY[i] = cosYaw[i];
        offAxisX[i] = axisY[i];
        offAxisY[i] = -axisX[i];
    }
    
    // Build main design matrix
    MatrixXd A_main(n, 4);
    VectorXd b_main(n);
    
    for (int i = 0; i < n; i++) {
        A_main(i, 0) = offAxisX[i];
        A_main(i, 1) = offAxisY[i];
        A_main(i, 2) = offAxisX[i] * dataT[i];
        A_main(i, 3) = offAxisY[i] * dataT[i];
        b_main(i) = offAxisX[i] * armorX[i] + offAxisY[i] * armorY[i];
    }
    
    // Build regularization design matrix
    MatrixXd A_reg(n, 4);
    VectorXd b_reg(n);
    
    for (int i = 0; i < n; i++) {
        A_reg(i, 0) = axisX[i];
        A_reg(i, 1) = axisY[i];
        A_reg(i, 2) = axisX[i] * dataT[i];
        A_reg(i, 3) = axisY[i] * dataT[i];
        b_reg(i) = axisX[i] * armorX[i] + axisY[i] * armorY[i] + lastR;
    }
    
    // Apply weights
    for (int i = 0; i < n; i++) {
        A_main.row(i) *= sqrtWeights[i];
        b_main(i) *= sqrtWeights[i];
        A_reg.row(i) *= sqrtWeights[i];
        b_reg(i) *= sqrtWeights[i];
    }
    
    // Compute condition number
    JacobiSVD<MatrixXd> svd(A_main, ComputeThinU | ComputeThinV);
    double cond_A = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
    
    // Adaptive regularization
    double lambda_reg = alpha * std::min(cond_A, 1e10);
    
    // Build augmented system
    MatrixXd A_aug(n * 2, 4);
    VectorXd b_aug(n * 2);
    
    A_aug.block(0, 0, n, 4) = A_main;
    A_aug.block(n, 0, n, 4) = std::sqrt(lambda_reg) * A_reg;
    
    b_aug.segment(0, n) = b_main;
    b_aug.segment(n, n) = std::sqrt(lambda_reg) * b_reg;
    
    // Tikhonov regularization
    MatrixXd A_tikh = MatrixXd::Zero(A_aug.rows() + 4, 4);
    VectorXd b_tikh = VectorXd::Zero(A_aug.rows() + 4);
    
    A_tikh.block(0, 0, A_aug.rows(), 4) = A_aug;
    A_tikh.block(A_aug.rows(), 0, 4, 4) = tikhonovLambda * MatrixXd::Identity(4, 4);
    
    b_tikh.segment(0, A_aug.rows()) = b_aug;
    
    // SVD solve
    JacobiSVD<MatrixXd> svd_tikh(A_tikh, ComputeThinU | ComputeThinV);
    VectorXd s_inv = svd_tikh.singularValues();
    
    double s_threshold = s_inv.maxCoeff() * std::max(A_tikh.rows(), A_tikh.cols()) * std::numeric_limits<double>::epsilon();
    for (int i = 0; i < s_inv.size(); i++) {
        s_inv(i) = (s_inv(i) > s_threshold) ? 1.0 / s_inv(i) : 0.0;
    }
    
    VectorXd x = svd_tikh.matrixV() * (s_inv.asDiagonal() * (svd_tikh.matrixU().transpose() * b_tikh));
    
    fitCenterXYResult result;
    result.center_x = x(0);
    result.center_y = x(1);
    result.center_vx = x(2);
    result.center_vy = x(3);
    
    return result;
}

void RotationMotionModel::fitRotationParameters() {
    if (observedDataHistory.size() < 10) return;
    
    auto params = getParams();
    std::vector<double> tData = params[0];
    std::vector<double> xData = params[1];
    std::vector<double> yData = params[2];
    std::vector<double> zData = params[3];
    std::vector<double> yawData = params[4];
    
    // Remove linear trends
    LinearRegressionResult xReg = linearRegression(tData, xData);
    LinearRegressionResult yReg = linearRegression(tData, yData);
    LinearRegressionResult zReg = linearRegression(tData, zData);
    LinearRegressionResult yawReg = linearRegression(tData, yawData);
    
    for (size_t i = 0; i < tData.size(); i++) {
        xData[i] -= (xReg.a + xReg.b * tData[i]);
        yData[i] -= (yReg.a + yReg.b * tData[i]);
        zData[i] -= (zReg.a + zReg.b * tData[i]);
        yawData[i] -= (yawReg.a + yawReg.b * tData[i]);
    }
    
    // Compute ACF for all components
    std::vector<double> acf_x = computeModifiedACF(xData);
    std::vector<double> acf_y = computeModifiedACF(yData);
    std::vector<double> acf_z = computeModifiedACF(zData);
    
    std::vector<double> yawScaled(yawData.size());
    for (size_t i = 0; i < yawData.size(); i++) {
        yawScaled[i] = yawData[i] * r;
    }
    std::vector<double> acf_yaw = computeModifiedACF(yawScaled);
    
    // Combine ACF
    size_t min_size = std::min({acf_x.size(), acf_y.size(), acf_z.size(), acf_yaw.size()});
    std::vector<double> combined_acf(min_size, 0.0);
    for (size_t i = 0; i < min_size; i++) {
        combined_acf[i] = acf_x[i] + acf_y[i];// + acf_z[i] + acf_yaw[i];
    }
    
    std::vector<double> refined_acf = lagStackWithDecay(combined_acf, refine_multiple);
    int max_idx = std::distance(refined_acf.begin(), 
                               std::max_element(refined_acf.begin(), refined_acf.end()));
    jump_period_frames = static_cast<double>(max_idx) / refine_multiple;
    
    if (jump_period_frames > 1 && jump_period_frames < tData.size() / 2) {
        double avg_interval = 0.0;
        for (size_t i = 1; i < tData.size(); i++) {
            avg_interval += (tData[i] - tData[i-1]);
        }
        avg_interval /= (tData.size() - 1);
        
        rotation_period = jump_period_frames * avg_interval * n_armors;
        vyaw = (rotation_period > 0) ? (2 * M_PI / rotation_period) : 0.0;
    }
    
    std::vector<double> TheoreticYawDataInCam(observedDataHistory.size());
    for (size_t i = 0; i < observedDataHistory.size(); i++) {
        TheoreticYawDataInCam[i] = (getTheoreticYawFacingArmor(observedDataHistory[i].x, observedDataHistory[i].y));
    }
    std::vector<double> dataToFit, fittedData;
    std::vector<int> midPoints = findMidYaw(TheoreticYawDataInCam, jump_period_frames, dataToFit, fittedData);
    rotation_direction = getRotationDirection(TheoreticYawDataInCam);
    vyaw *= rotation_direction;
    
    if (!midPoints.empty()) {
        int last_mid_idx = midPoints.back();
        if (last_mid_idx < tData.size()) {
            double time_since_mid = tData.back() - tData[last_mid_idx];
            double camToCenterYaw = getCamToCenterYaw();
            current_phase = fmod(time_since_mid * vyaw + delta_phase * rotation_direction + camToCenterYaw, 2 * M_PI);
        }
    }
}

double RotationMotionModel::getJumpPeriod() {
    return jump_period_frames;
}

std::vector<int> RotationMotionModel::findMidYaw(const std::vector<double>& yawData, 
                                                double periodFrames,
                                                std::vector<double>& dataToFit,
                                                std::vector<double>& fittedData) {
    double yaw_mean = 0.0;
    for (double val : yawData) {
        yaw_mean += val;
    }
    yaw_mean /= yawData.size();
    
    dataToFit.resize(yawData.size());
    for (size_t i = 0; i < yawData.size(); i++) {
        double diff = yawData[i] - yaw_mean;
        dataToFit[i] = tanh(diff * diff * 3);
    }
    
    int n = dataToFit.size();
    std::vector<double> ts(n);
    for (int i = 0; i < n; i++) {
        ts[i] = i;
    }
    
    std::vector<double> thetas(n);
    for (int i = 0; i < n; i++) {
        thetas[i] = ts[i] / periodFrames * 2 * M_PI;
    }
    
    double a0 = 0.0, a1 = 0.0, b1 = 0.0;
    for (int i = 0; i < n; i++) {
        a0 += dataToFit[i];
        a1 += dataToFit[i] * cos(thetas[i]);
        b1 += dataToFit[i] * sin(thetas[i]);
    }
    a0 /= n;
    a1 = a1 * 2 / n;
    b1 = b1 * 2 / n;
    
    double phi = atan2(b1, a1);
    double A = sqrt(a1 * a1 + b1 * b1);
    
    std::vector<double> thetas_fitted(n);
    for (int i = 0; i < n; i++) {
        thetas_fitted[i] = thetas[i] - phi;
    }
    
    fittedData.resize(n);
    for (int i = 0; i < n; i++) {
        fittedData[i] = a0 + A * cos(thetas_fitted[i]);
    }
    
    std::vector<int> mid_points;
    for (int i = 1; i < n - 1; i++) {
        if (fittedData[i] <= fittedData[i-1] && fittedData[i] <= fittedData[i+1]) {
            mid_points.push_back(i);
        }
    }
    
    return mid_points;
}

int RotationMotionModel::getRotationDirection(const std::vector<double>& yawData) {
    if (yawData.size() < 2) return 1;
    
    double d_yaw_integrate = 0.0;
    for (int first_idx = 0; first_idx < yawData.size() - 1; first_idx += 1) {
        d_yaw_integrate += 0.1 * (yawData[first_idx + 1] - yawData[first_idx]) / 
                           ((yawData[first_idx + 1] - yawData[first_idx]) * (yawData[first_idx + 1] - yawData[first_idx]) + 0.01);
    }
    
    return (d_yaw_integrate > 0) ? 1 : -1;
}

double RotationMotionModel::centerResiduals(const std::vector<double>& params, 
                                          double armorYaw, double armorX, 
                                          double armorY, double dataT) {
    double center_x = params[0];
    double center_vx = params[1];
    double center_y = params[2];
    double center_vy = params[3];
    
    double axis_vector_x = -sin(armorYaw);
    double axis_vector_y = cos(armorYaw);
    double off_axis_vector_x = axis_vector_y;
    double off_axis_vector_y = -axis_vector_x;
    
    double aromr_to_center_vector_x = center_x + dataT * center_vx - armorX;
    double aromr_to_center_vector_y = center_y + dataT * center_vy - armorY;
    
    return aromr_to_center_vector_x * off_axis_vector_x + aromr_to_center_vector_y * off_axis_vector_y;
}

PredictResult RotationMotionModel::predict(double predictTime) {
    PredictResult result;
    std::vector<double> start_yaw_distances(n_armors);
    double latest_yaw = observedDataHistory.back().theoreticYaw;
    for (int i = 0; i < n_armors; i++) {
        start_yaw_distances[i] = std::min(std::min(
            std::abs(current_phase + i * jump_rad - latest_yaw),
            std::abs(current_phase + i * jump_rad - latest_yaw + 2 * M_PI)),
            std::abs(current_phase + i * jump_rad - latest_yaw - 2 * M_PI));
    }
    auto start_yaw_distances_min_iter = std::min_element(start_yaw_distances.begin(), start_yaw_distances.end());
    int start_yaw_distances_min_index = std::distance(start_yaw_distances.begin(), start_yaw_distances_min_iter);
    double start_yaw = current_phase + start_yaw_distances_min_index * jump_rad;

    result.center_x = center_x + predictTime * center_vx;
    result.center_y = center_y + predictTime * center_vy;
    result.center_z = center_z + predictTime * center_vz;
    result.r = r;
    result.yaw = start_yaw + predictTime * vyaw;
    for (int i = 0; i < n_armors; i++) {
        double armor_yaw = result.yaw - i * rotation_direction * jump_rad;
        result.armors.push_back(SimpleArmor({
            result.center_x + r * std::sin(armor_yaw),
            result.center_y - r * std::cos(armor_yaw),
            result.center_z,
            armor_yaw
        }));
    }

    result.rotation_direction = rotation_direction;

    return result;
}

double RotationMotionModel::getCamToCenterYaw() {
    std::vector<float> cam_center = rest_frame_ -> getCamPosition();
    std::vector<double> cam_to_rotation_center_vector = {center_x - cam_center[0], center_y - cam_center[1]};
    double cam_to_rotation_center_yaw = std::atan2(-cam_to_rotation_center_vector[0], cam_to_rotation_center_vector[1]);
    return cam_to_rotation_center_yaw;
}

double RotationMotionModel::getTheoreticYaw(double armor_x, double armor_y) {
    double theoreticYawFacingArmor = getTheoreticYawFacingArmor(armor_x, armor_y);
    std::vector<float> cam_center = rest_frame_ -> getCamPosition();
    std::vector<double> cam_to_rotation_center_vector = {center_x - cam_center[0], center_y - cam_center[1]};
    double cam_to_rotation_center_yaw = getCamToCenterYaw();
    return cam_to_rotation_center_yaw + theoreticYawFacingArmor;
}

double RotationMotionModel::getTheoreticYawFacingArmor(double armor_x, double armor_y) {
    std::vector<float> cam_center = rest_frame_ -> getCamPosition();
    std::vector<double> cam_to_rotation_center_vector = {center_x - cam_center[0], center_y - cam_center[1]};
    double cam_to_rotation_center_vector_len = std::sqrt(cam_to_rotation_center_vector[0] * cam_to_rotation_center_vector[0] + cam_to_rotation_center_vector[1] * cam_to_rotation_center_vector[1]);
    std::vector<double> right_unit_v = {cam_to_rotation_center_vector[1] / cam_to_rotation_center_vector_len, - cam_to_rotation_center_vector[0] / cam_to_rotation_center_vector_len};
    std::vector<double> center_armor_v = {armor_x - center_x, armor_y - center_y};
    double right_shift = right_unit_v[0] * center_armor_v[0] + right_unit_v[1] * center_armor_v[1];
    if (right_shift > r) {
        return M_PI / 2.0;
    }
    if (right_shift < -r) {
        return - M_PI / 2.0;
    }
    return std::asin(right_shift / r);
}

RotationMotionState RotationMotionModel::getState() {
    RotationMotionState state({
        center_vx,
        center_vy,
        center_vz,
        vyaw,
        r,
        center_x,
        center_y,
        center_z
    });
    return state;
}
