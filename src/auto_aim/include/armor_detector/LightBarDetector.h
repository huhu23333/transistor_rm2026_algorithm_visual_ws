// // LightBarDetector.h
// // 负责检测图像中的装甲板灯条
// // 包含灯条类(Light)和灯条检测器类(LightBarDetector)

// #ifndef LIGHTBARDETECTOR_H
// #define LIGHTBARDETECTOR_H

// #include <opencv2/opencv.hpp>
// #include <vector>
// #include "Params.h"

// /**
//  * @brief 灯条类，存储和处理单个灯条的信息
//  */
// class Light {
// public:
//     cv::RotatedRect el;      // 灯条的旋转矩形，包含中心点、大小和角度
//     float length;            // 灯条的长度（较长边）
//     float width;             // 灯条的宽度（较短边）
//     float angle;             // 灯条的倾斜角度，范围[-90,90]
//     cv::Point2f top;         // 灯条顶部点坐标
//     cv::Point2f bottom;      // 灯条底部点坐标

//     /**
//      * @brief 构造函数，通过旋转矩形初始化灯条
//      * @param rect 输入的旋转矩形
//      */
//     explicit Light(const cv::RotatedRect& rect);

//     /**
//      * @brief 计算灯条的所有几何参数
//      * 包括长度、宽度、角度、顶部点和底部点
//      */
//     void calculateDimensions();

//     /**
//      * @brief 获取灯条的旋转矩形
//      * @return 返回灯条的旋转矩形
//      */
//     cv::RotatedRect getRect() const { return el; }
// };

// /**
//  * @brief 灯条检测器类，负责检测和处理图像中的所有灯条
//  */
// class LightBarDetector {
// public:
//     /**
//      * @brief 构造函数
//      * @param params 检测参数结构体，包含各种阈值和配置
//      */
//     explicit LightBarDetector(const Params& params);
    
//     /**
//      * @brief 设置敌方装甲板的颜色
//      * @param color 颜色枚举值（红色或蓝色）
//      */
//     void setEnemyColor(int color);

//     /**
//      * @brief 在输入图像中检测灯条
//      * @param images 输入图像vector
//      */
//     void detectLights(const std::vector<cv::Mat>& images);

//     /**
//      * @brief 处理检测到的灯条（过滤和更新）
//      */
//     void processLights();

//     /**
//      * @brief 获取当前检测到的所有灯条
//      * @return 灯条vector的常量引用
//      */
//     const std::vector<Light>& getLights() const { return lights; }

// private:
//     std::vector<Light> lights;           // 存储检测到的所有灯条
//     Params params;                       // 检测参数
//     Params::EnemyColor enemy_color;      // 敌方颜色

//     /**
//      * @brief 根据敌方颜色提取颜色通道差值图像
//      * @param img 输入图像
//      * @return 处理后的二值图像
//      */
//     cv::Mat extractColorChannelDiff(const cv::Mat& img);

//     /**
//      * @brief 在二值图像中检测可能的灯条
//      * @param img 输入的二值图像
//      * @return 检测到的旋转矩形vector
//      */
//     std::vector<cv::RotatedRect> detectLightRects(const cv::Mat& img);

//     /**
//      * @brief 过滤不符合条件的灯条
//      */
//     void filterLights();

//     /**
//      * @brief 更新灯条状态（用于追踪）
//      */
//     void updateLights();
// };

// #endif // LIGHTBARDETECTOR

#ifndef LIGHTBARDETECTOR_H
#define LIGHTBARDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "Params.h"

/**
 * @brief 灯条类，存储和处理单个灯条的信息
 */
class Light {
public:
    cv::RotatedRect el;      // 灯条的旋转矩形，包含中心点、大小和角度
    float length;            // 灯条的长度（较长边）
    float width;             // 灯条的宽度（较短边）
    float angle;             // 灯条的倾斜角度，范围[-90,90]
    cv::Point2f top;         // 灯条顶部点坐标
    cv::Point2f bottom;      // 灯条底部点坐标

    /**
     * @brief 构造函数，通过旋转矩形初始化灯条
     * @param rect 输入的旋转矩形
     */
    explicit Light(const cv::RotatedRect& rect);

    /**
     * @brief 计算灯条的所有几何参数
     * 包括长度、宽度、角度、顶部点和底部点
     */
    void calculateDimensions();

    /**
     * @brief 获取灯条的旋转矩形
     * @return 返回灯条的旋转矩形
     */
    cv::RotatedRect getRect() const { return el; }
};

/**
 * @brief 灯条检测器类，负责检测和处理图像中的所有灯条
 */
class LightBarDetector {
public:
    /**
     * @brief 构造函数
     * @param params 检测参数结构体，包含各种阈值和配置
     */
    explicit LightBarDetector(const Params& params);
    
    /**
     * @brief 设置敌方装甲板的颜色
     * @param color 颜色枚举值（红色或蓝色）
     */
    void setEnemyColor(int color);

    /**
     * @brief 在输入图像中检测灯条
     * @param images 输入图像vector
     */
    void detectLights(const std::vector<cv::Mat>& images);

    /**
     * @brief 处理检测到的灯条（过滤和更新）
     */
    void processLights();

    /**
     * @brief 获取当前检测到的所有灯条
     * @return 灯条vector的常量引用
     */
    const std::vector<Light>& getLights() const { return lights; }

private:
    std::vector<Light> lights;           // 存储检测到的所有灯条
    Params params;                       // 检测参数
    Params::EnemyColor enemy_color;      // 敌方颜色

    /**
     * @brief 根据敌方颜色提取颜色通道差值图像
     * @param img 输入图像
     * @return 处理后的二值图像
     */
    cv::Mat extractColorChannelDiff(const cv::Mat& img);

    /**
     * @brief 在二值图像中检测可能的灯条
     * @param img 输入的二值图像
     * @return 检测到的旋转矩形vector
     */
    std::vector<cv::RotatedRect> detectLightRects(const cv::Mat& img);

    /**
     * @brief 过滤不符合条件的灯条
     */
    void filterLights();

    /**
     * @brief 更新灯条状态（用于追踪）
     */
    void updateLights();

    /**
     * @brief 计算两个灯条是否重叠
     * @param light1 灯条1
     * @param light2 灯条2
     * @return 如果两个灯条重叠，返回true
     */
    bool isOverlap(const Light& light1, const Light& light2);
};

#endif // LIGHTBARDETECTOR_H
