// PeriodicDataPredictor.cpp
#include "predictor/PeriodicDataPredictor.h"

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



void PeriodicDataPredictor::autoFindPeriod() {
    std::vector<double> modified_acf = computeModifiedACF(history_);

    std::vector<double> acf_stack = lagStackWithDecay(modified_acf);
    auto acf_stack_max_iter = std::max_element(acf_stack.begin(), acf_stack.end());
    period_ = std::distance(acf_stack.begin(), acf_stack_max_iter);
    
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