#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

struct LinearRegressionResult {
    double a;
    double b;
};

LinearRegressionResult simpleLinearRegression(const std::vector<double>& xn, const std::vector<double>& yn);
std::vector<double> computeModifiedACF(const std::vector<double>& residual);
double variance(const std::vector<double>& signal);
std::vector<double> linearInterpolation(const std::vector<double>& data, int refineMultiple);
std::vector<double> lagStackWithDecay(const std::vector<double>& signal, int refineMultiple = 1);