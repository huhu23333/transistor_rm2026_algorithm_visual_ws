// PeriodicDataPredictor.cpp
#include "utils/PeriodicDataPredictor.h"

PeriodicDataPredictor::PeriodicDataPredictor(int max_history) 
    : max_history_(max_history) {
    if (max_history <= 0) {
        throw std::invalid_argument("max_history must be positive");
    }
}

void PeriodicDataPredictor::addPoint(double point) {
    history_.push_back(point);
    point_count_++;
    
    // 保持历史不超过最大步数
    if (history_.size() > max_history_) {
        history_.erase(history_.begin());
    }
    
    coefficients_dirty_ = true;
}

void PeriodicDataPredictor::setPeriod(int period) {
    if (period <= 0) {
        period_ = 1;
    } else {
        period_ = period;
    }
    coefficients_dirty_ = true;
}


std::vector<double> PeriodicDataPredictor::computeModifiedACF() const {
    int n = history_.size();
    if (n == 0) return {};
    
    double history_mean = std::accumulate(history_.begin(), history_.end(), 0.0) / n;
    double denominator = 0.0;
    for (double r : history_) {
        denominator += (r - history_mean) * (r - history_mean);
    }
    
    if (denominator == 0) {
        return std::vector<double>(n / 2 + 1, 0.0);
    }
    
    int max_lag = static_cast<int>(n * 0.8);
    std::vector<double> modified_acf(max_lag + 1);
    
    for (int k = 0; k <= max_lag; k++) {
        double numerator = 0.0;
        if (k == 0) {
            for (int t = 0; t < n; t++) {
                numerator += (history_[t] - history_mean) * (history_[t] - history_mean);
            }
            numerator /= n;
        } else {
            for (int t = 0; t < n - k; t++) {
                numerator += (history_[t] - history_mean) * (history_[t + k] - history_mean);
            }
            numerator /= (n - k);
        }
        modified_acf[k] = numerator / denominator;
    }
    
    return modified_acf;
}

void PeriodicDataPredictor::autoFindPeriod() {
    std::vector<double> modified_acf = computeModifiedACF();

    if (modified_acf.size() < 2) {
        period_ = 1;
        return;  // 直接返回，避免后续访问
    }
    
    int max_k = 1;
    double max_value = modified_acf[1];
    double last_modified_acf = modified_acf[1];
    
    // 寻找第一个下降点
    for (int k = 2; k < static_cast<int>(modified_acf.size() / 2); k++) {
        if (modified_acf[k] < 0) {
            max_k = k;
            max_value = modified_acf[k];
            last_modified_acf = modified_acf[k];
            break;
        }
    }
    
    bool modified_acf_updating = false;
    for (int k = max_k + 1; k < static_cast<int>(modified_acf.size()); k++) {
        if ((modified_acf[k] > max_value * 3.0) || (modified_acf_updating && modified_acf[k] > max_value * 0.8)) {
            if (modified_acf[k] > last_modified_acf) {
                modified_acf_updating = true;
            }
            if (modified_acf[k] > max_value) {
                max_value = modified_acf[k];
                max_k = k;
            }
        }
        if (modified_acf[k] < last_modified_acf * 0.8) {
            modified_acf_updating = false;
            //break;
        }
        last_modified_acf = modified_acf[k];
    }
    
    period_ = max_k;
}

int PeriodicDataPredictor::getPeriod() const {
    return period_;
}

double PeriodicDataPredictor::smooth(int time_index) const {
    if (history_.empty()) {
        return 0.0;
    }
    
    // 如果需要，重新计算傅里叶系数
    if (coefficients_dirty_) {
        computeFourierCoefficients();
    }
    
    // 计算绝对时间索引
    // 正数表示相对最后添加的数据点向后预测
    // 负数表示相对最后一个数据点向前x索引处数据的平滑
    int absolute_index = static_cast<int>(history_.size()) - 1 + time_index;
    
    // 使用傅里叶级数计算平滑值（阶数为1）
    double t = static_cast<double>(absolute_index);
    return a0_ + a1_ * std::cos(2 * M_PI * t / period_) + b1_ * std::sin(2 * M_PI * t / period_);
}

bool PeriodicDataPredictor::isRising(int time_index, double compare_threshold) const {
    // 计算导数并判断是否大于0
    return computeDerivative(time_index) / std::sqrt(a1_ * a1_ + b1_ * b1_) > compare_threshold;
}

bool PeriodicDataPredictor::isUpper(int time_index, double compare_threshold) const {
    // 计算相位是否为正半周期
    if (history_.empty() || period_ <= 0) {
        return 0.0;
    }
    
    // 如果需要，重新计算傅里叶系数
    if (coefficients_dirty_) {
        computeFourierCoefficients();
    }
    
    // 计算绝对时间索引
    int absolute_index = static_cast<int>(history_.size()) - 1 + time_index;
    double t = static_cast<double>(absolute_index);
    
    // 计算傅里叶级数的相位
    // f(t) = a0 + a1*cos(2πt/T) + b1*sin(2πt/T)
    double omega = 2 * M_PI / period_;
    return (a1_ * omega * std::cos(omega * t) + b1_ * omega * std::sin(omega * t)) / std::sqrt(a1_ * a1_ + b1_ * b1_) > compare_threshold;
}

double PeriodicDataPredictor::getA0() const {
    return a0_;
}

void PeriodicDataPredictor::clearHistory() {
    history_.clear();
    point_count_ = 0;
    coefficients_dirty_ = true;
}

int PeriodicDataPredictor::getPointCount() const {
    return point_count_;
}

void PeriodicDataPredictor::computeFourierCoefficients() const {
    if (history_.empty()) {
        a0_ = 0.0;
        a1_ = 0.0;
        b1_ = 0.0;
        coefficients_dirty_ = false;
        return;
    }
    
    if (period_ <= 0) {
        throw std::runtime_error("Period must be set before computing Fourier coefficients");
    }
    
    int n = static_cast<int>(history_.size());
    
    // 计算a0 (直流分量)
    a0_ = std::accumulate(history_.begin(), history_.end(), 0.0) / n;
    
    // 计算a1和b1 (基波分量)
    double sum_cos = 0.0;
    double sum_sin = 0.0;
    
    for (int i = 0; i < n; i++) {
        double theta = 2 * M_PI * i / period_;
        sum_cos += history_[i] * std::cos(theta);
        sum_sin += history_[i] * std::sin(theta);
    }
    
    a1_ = 2.0 * sum_cos / n;
    b1_ = 2.0 * sum_sin / n;
    
    coefficients_dirty_ = false;
}

double PeriodicDataPredictor::computeDerivative(int time_index) const {
    if (history_.empty() || period_ <= 0) {
        return 0.0;
    }
    
    // 如果需要，重新计算傅里叶系数
    if (coefficients_dirty_) {
        computeFourierCoefficients();
    }
    
    // 计算绝对时间索引
    int absolute_index = static_cast<int>(history_.size()) - 1 + time_index;
    double t = static_cast<double>(absolute_index);
    
    // 计算傅里叶级数的导数
    // f(t) = a0 + a1*cos(2πt/T) + b1*sin(2πt/T)
    // f'(t) = -a1*(2π/T)*sin(2πt/T) + b1*(2π/T)*cos(2πt/T)
    double omega = 2 * M_PI / period_;
    return -a1_ * omega * std::sin(omega * t) + b1_ * omega * std::cos(omega * t);
}