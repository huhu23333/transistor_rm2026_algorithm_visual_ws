#ifndef UNWARP_UTILS_H
#define UNWARP_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>

class UnwarpUtils {
public:
    /**
     * @brief 将图像中的不规则凸四边形区域展平成矩形
     * 
     * @param input 输入图像
     * @param corners 四边形区域的四个顶点坐标，按左上->左下->右下->右上的顺序排列
     * @param outputWidth 可选参数，指定输出矩形的宽度（默认自动计算）
     * @param outputHeight 可选参数，指定输出矩形的高度（默认自动计算）
     * @return cv::Mat 展平后的矩形图像（错误时返回100x100黑色图像）
     */
    static cv::Mat unwarpQuadrilateral(
        const cv::Mat& input,
        const std::vector<cv::Point2f>& corners,
        int outputWidth = -1,
        int outputHeight = -1
    );
};

#endif // UNWARP_UTILS_H