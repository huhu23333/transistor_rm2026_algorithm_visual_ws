// RestFrameDraw.h

#include <opencv2/opencv.hpp>
#include "3d_processing/RestFrame.h"
#include "3d_processing/ArmorSolver.h"

void drawRestFrame(cv::Mat& image, std::shared_ptr<RestFrame> rest_frame, std::shared_ptr<ArmorSolver> armor_solver);

