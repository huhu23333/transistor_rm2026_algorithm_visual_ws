#pragma once
#include <vector>
#include <cmath>
#include <algorithm>


struct LinearRegressionResult {
    double a;
    double b;
};

LinearRegressionResult linearRegression(const std::vector<double>& xn, const std::vector<double>& yn);
double variance(const std::vector<double>& signal);
std::vector<double> linearInterpolation(const std::vector<double>& data, int refineMultiple);
double meanSquaredError(const std::vector<double>& pred_value, const std::vector<double>& true_value);