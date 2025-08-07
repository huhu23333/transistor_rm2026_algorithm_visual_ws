#include "test_codes/UnwarpUtils.h"

cv::Mat UnwarpUtils::unwarpQuadrilateral(
    const cv::Mat& input,
    const std::vector<cv::Point2f>& corners
) {
    // 输出尺寸256*256
    int outputWidth = 256;
    int outputHeight = 256;
    // 验证输入

    if (corners.size() != 4) {
        // 返回空cv::Mat
        return cv::Mat();
    }
    
    // 验证输入图像有效性
    if (input.empty()) {
        return cv::Mat();
    }

    // 定义目标矩形的四个角点（按相同顺序）
    std::vector<cv::Point2f> dstCorners = {
        cv::Point2f(0, 0),                     // 左上
        cv::Point2f(0, static_cast<float>(outputHeight)),           // 左下
        cv::Point2f(static_cast<float>(outputWidth), static_cast<float>(outputHeight)), // 右下
        cv::Point2f(static_cast<float>(outputWidth), 0)             // 右上
    };

    try {
        // 计算透视变换矩阵
        cv::Mat transform = cv::getPerspectiveTransform(corners, dstCorners);

        // 应用透视变换
        cv::Mat output;
        cv::warpPerspective(
            input, 
            output, 
            transform, 
            cv::Size(outputWidth, outputHeight),
            cv::INTER_LINEAR,    // 线性插值
            cv::BORDER_CONSTANT   // 边界填充为常数（黑色）
        );

        return output;
    } catch (...) {
        // 发生任何异常时返回cv::Mat
        return cv::Mat();
    }
}