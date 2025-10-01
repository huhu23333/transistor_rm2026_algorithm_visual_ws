#include "utils/UtilsFunc.h"

LinearRegressionResult simpleLinearRegression(const std::vector<double>& xn, const std::vector<double>& yn) {
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

std::vector<double> computeModifiedACF(const std::vector<double>& residual) {
    int n = residual.size();
    if (n == 0) return {};
    
    double residual_mean = 0.0;
    for (double val : residual) {
        residual_mean += val;
    }
    residual_mean /= n;
    
    int max_lag = static_cast<int>(n * 0.8);
    std::vector<double> modified_acf(max_lag + 1, 0.0);
    
    for (int k = 0; k <= max_lag; k++) {
        if (k == 0) {
            double sum = 0.0;
            for (double val : residual) {
                sum += (val - residual_mean) * (val - residual_mean);
            }
            modified_acf[k] = sum / n;
        } else {
            double sum = 0.0;
            for (int i = 0; i < n - k; i++) {
                sum += (residual[i] - residual_mean) * (residual[i + k] - residual_mean);
            }
            modified_acf[k] = sum / (n - k);
        }
    }
    
    return modified_acf;
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

std::vector<double> lagStackWithDecay(const std::vector<double>& signal, int refineMultiple) {
    std::vector<double> refined_signal = linearInterpolation(signal, refineMultiple);
    int result_len = refined_signal.size();
    std::vector<double> result(result_len, 0.0);
    
    for (int lag = 1; lag < result_len; lag++) {
        int lag_n = result_len / lag;
        int lag_left = result_len - lag_n * lag;
        
        std::vector<double> temp(lag, 0.0);
        for (int lag_i = 0; lag_i < lag_n; lag_i++) {
            for (int j = 0; j < lag; j++) {
                temp[j] += refined_signal[lag_i * lag + j];
            }
        }
        
        if (lag_left > 0) {
            for (int j = 0; j < lag_left; j++) {
                temp[j] += refined_signal[result_len - lag_left + j];
                temp[j] /= (lag_n + 1);
            }
            for (int j = lag_left; j < lag; j++) {
                temp[j] /= lag_n;
            }
        } else {
            for (int j = 0; j < lag; j++) {
                temp[j] /= lag_n;
            }
        }
        
        std::vector<double> temp_vec(temp.begin(), temp.end());
        result[lag] = variance(temp_vec) / lag;
    }
    
    return result;
}