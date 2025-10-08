#include "utils/DataProcessFuncs.h"


LinearRegressionResult linearRegression(const std::vector<double>& xn, const std::vector<double>& yn) {
    double x_mean = 0.0, y_mean = 0.0;
    int n = xn.size();
    
    for (int i = 0; i < n; i++) {
        x_mean += xn[i];
        y_mean += yn[i];
    }
    x_mean /= n;
    y_mean /= n;
    
    double numerator = 0.0, denominator = 0.0;
    for (int i = 0; i < n; i++) {
        numerator += (xn[i] - x_mean) * (yn[i] - y_mean);
        denominator += (xn[i] - x_mean) * (xn[i] - x_mean);
    }
    
    LinearRegressionResult result;
    result.b = numerator / denominator;
    result.a = y_mean - result.b * x_mean;
    
    return result;
}

double variance(const std::vector<double>& signal) {
    double signal_mean = 0.0;
    int n = signal.size();
    
    for (double val : signal) {
        signal_mean += val;
    }
    signal_mean /= n;
    
    double var = 0.0;
    for (double val : signal) {
        var += (val - signal_mean) * (val - signal_mean);
    }
    
    return var / n;
}

std::vector<double> linearInterpolation(const std::vector<double>& data, int refineMultiple) {
    int result_len = (data.size() - 1) * refineMultiple + 1;
    std::vector<double> result(result_len, 0.0);
    
    for (int result_i = 0; result_i < result_len; result_i++) {
        int origin_i = result_i / refineMultiple;
        int result_i_left_part = result_i - origin_i * refineMultiple;
        
        if (result_i_left_part == 0) {
            result[result_i] = data[origin_i];
        } else {
            double weight_high = static_cast<double>(result_i_left_part) / refineMultiple;
            double weight_low = 1.0 - weight_high;
            result[result_i] = weight_low * data[origin_i] + weight_high * data[origin_i + 1];
        }
    }
    
    return result;
}

double meanSquaredError(const std::vector<double>& pred_value, const std::vector<double>& true_value) {
    double result = 0.0;
    size_t value_num = std::min(pred_value.size(), true_value.size());
    if (value_num == 0) {
        return 0.0;
    }
    for (size_t value_index = 0; value_index < value_num; value_index++) {
        double value_error = pred_value[value_index] - true_value[value_index];
        result += value_error * value_error;
    }
    return result / static_cast<double>(value_num);
}
