// LightBarDetector.cpp
// 实现灯条检测的具体功能

#include "armor_detector/LightBarDetector.h"

/************************* Light类实现 *************************/

// 默认构造函数
Light::Light() {};

Light::Light(const cv::RotatedRect& rect) 
    : el(rect), length(0), width(0), angle(0) {
    calculateDimensions();  // 构造时计算所有几何参数
}

void Light::calculateDimensions() {
    // 1. 计算角度并标准化到[-90, 90]
    angle = el.angle;

    // 2. 计算长宽（确保length始终为较长边）
    length = el.size.height;
    width = el.size.width;
    
    // 3. 计算顶部和底部点
    cv::Point2f vertices[4];
    el.points(vertices);
    
    // 将四个顶点按y坐标排序
    std::vector<cv::Point2f> vertexVec(vertices, vertices + 4);
    std::sort(vertexVec.begin(), vertexVec.end(), 
        [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });
    
    // 取最上和最下的点
    top = vertexVec[0];
    bottom = vertexVec[3];
}

void Light::correctLength(const cv::Mat& binary_img) {
    float sum_value_target = computeRotatedRectSum(el, binary_img) * 0.99;
    // 二分查找使旋转矩形内二值图像总值下降为原来0.99倍的长度
    int binarySearchFrequency = 10; // 二分查找次数
    float upper_ratio = 1.0;
    float lower_ratio = 0.0;
    float try_ratio;
    float length_original = length;
    for (int i = 0; i < binarySearchFrequency; i++) {
        try_ratio = (upper_ratio + lower_ratio) * 0.5;
        el.size.height = length_original * try_ratio;
        float sum_value = computeRotatedRectSum(el, binary_img);
        if (sum_value > sum_value_target) {
            upper_ratio = try_ratio;
        } else {
            lower_ratio = try_ratio;
        }
    }
    try_ratio = (upper_ratio + lower_ratio) * 0.5;
    length = length_original * try_ratio;
    el.size.height = length;
}

// 高效计算旋转矩形内白色像素面积的辅助函数
float Light::computeRotatedRectSum(const cv::RotatedRect& rect, const cv::Mat& binary_img) {
    // 获取旋转矩形的四个顶点
    cv::Point2f vertices[4];
    rect.points(vertices);
    
    // 计算最小外接矩形（ROI）
    cv::Rect boundRect = cv::boundingRect(std::vector<cv::Point2f>(vertices, vertices + 4));
    boundRect &= cv::Rect(0, 0, binary_img.cols, binary_img.rows);  // 确保在图像范围内
    
    // 如果ROI无效，返回0面积
    if (boundRect.width <= 0 || boundRect.height <= 0) {
        return 0;
    }
    
    // 创建局部ROI的掩码
    cv::Mat localMask = cv::Mat::zeros(boundRect.size(), CV_8UC1);
    
    // 将顶点坐标转换为ROI局部坐标
    std::vector<cv::Point> polyPoints;
    for (int i = 0; i < 4; i++) {
        polyPoints.push_back(cv::Point(
            static_cast<int>(vertices[i].x - boundRect.x),
            static_cast<int>(vertices[i].y - boundRect.y)
        ));
    }

    // 填充多边形创建掩码
    cv::fillConvexPoly(localMask, polyPoints, cv::Scalar(255));
    
    // 提取ROI区域并计算面积
    cv::Mat roi = binary_img(boundRect);
    
    // 将旋转矩形区域填充为白色（255）
    cv::fillConvexPoly(localMask, polyPoints, cv::Scalar(255));
    // 计算掩码区域的总值
    cv::Mat masked;
    roi.copyTo(masked, localMask);
    float sum_value = cv::sum(masked)[0];

    return sum_value;
}

/************************* LightBarDetector类实现 *************************/

LightBarDetector::LightBarDetector(const Params& params, std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node) // 新增传入节点，用于debug打印
    : params(params), enemy_color(params.enemy_color), node(node), config_file_ptr(config_file_ptr) {
        mean_color_diff_THRESHOLD_BLUE = (*config_file_ptr)["mean_color_diff_THRESHOLD_BLUE"].as<float>();
        mean_color_diff_THRESHOLD_RED = (*config_file_ptr)["mean_color_diff_THRESHOLD_RED"].as<float>();
        color_rect_expand_FACTOR = (*config_file_ptr)["color_rect_expand_FACTOR"].as<float>(); 
        binary_img_THRESHOLD = (*config_file_ptr)["binary_img_THRESHOLD"].as<uint8_t>(); 
        THRES_MAX_COLOR_RED = (*config_file_ptr)["THRES_MAX_COLOR_RED"].as<int>(); 
        THRES_MAX_COLOR_BLUE = (*config_file_ptr)["THRES_MAX_COLOR_BLUE"].as<int>(); 
    }

void LightBarDetector::setEnemyColor(int color) {
    enemy_color = static_cast<Params::EnemyColor>(color);
    params.enemy_color = enemy_color;
}

struct alignas(64) LightDetectThreadInfo { // 64字节对齐
    cv::RotatedRect* lightRect;
    bool is_true_light = true;
    Light light;
};

void LightBarDetector::detectLights(const std::vector<cv::Mat>& images) {
    lights.clear();  // 清除上一帧的检测结果
    
    // 处理每一帧图像
    for (const auto& img : images) {

        // 1. 提取二值化图片
        cv::Mat binary_img = binaryImg(img);
        // cv::imshow("Light Bar Debug", binary_img);

        // 2. 检测可能的灯条
        std::vector<cv::RotatedRect> detectedRects = detectLightRects(binary_img);

        // 3. 移除颜色错误的灯条，只保留目标颜色的灯条

        // 进行多线程优化
        int lightRectsNum = detectedRects.size();
        std::vector<LightDetectThreadInfo> lightDetectThreadInfos(lightRectsNum);
        for (size_t i = 0; i < lightRectsNum; ++i) {
            lightDetectThreadInfos[i].lightRect = &detectedRects[i];
        }

        cv::Mat color_diff;
        if (enemy_color != Params::BOTH) {
            // 1. 提取颜色通道差值图像
            color_diff = extractColorChannelDiff(img);
            // cv::imshow("Light Bar Debug", color_diff);
        }

        std::for_each(std::execution::par, lightDetectThreadInfos.begin(), lightDetectThreadInfos.end(), 
        [&](LightDetectThreadInfo& lightDetectThreadInfo) {

            cv::RotatedRect& rect = *lightDetectThreadInfo.lightRect;
                    
            if (enemy_color != Params::BOTH) {
                // 2. 获取扩张后的旋转矩形
                cv::RotatedRect expandedRect = rectExpand(rect, color_rect_expand_FACTOR);

                // 3. 获取矩形范围内通道差值图像的均值
                float mean_color_diff = calculateMeanInRotatedRect(color_diff, expandedRect);

                // 4. 移除小于阈值的图像
                RCLCPP_DEBUG(node->get_logger(), "mean_color_diff: %f\n", mean_color_diff);
                float mean_color_diff_THRESHOLD;
                if (params.enemy_color == Params::BLUE) {
                    mean_color_diff_THRESHOLD = mean_color_diff_THRESHOLD_BLUE;
                } else {
                    mean_color_diff_THRESHOLD = mean_color_diff_THRESHOLD_RED;
                }
                if (mean_color_diff < mean_color_diff_THRESHOLD) {
                    lightDetectThreadInfo.is_true_light = false;
                    return;
                }
            }

            // 将Light::calculateDimensions的方向纠正迁移至此
            // 1. 计算角度并标准化到[-90, 90]
            if (rect.size.width > rect.size.height) {
                rect.angle += 90;  // 确保角度始终表示长边的方向
            }
            while (rect.angle > 90) rect.angle -= 180;
            while (rect.angle < -90) rect.angle += 180;

            // 2. 计算长宽（确保length始终为较长边）
            float length = std::max(rect.size.width, rect.size.height);
            float width = std::min(rect.size.width, rect.size.height);
            rect.size.height = length;
            rect.size.width = width;

            // 4. 将检测到的旋转矩形转换为Light对象
            lightDetectThreadInfo.light = Light(rect);
            // 5. 修正在拟合旋转矩形时造成的长度误差
            lightDetectThreadInfo.light.correctLength(binary_img);
        });
        
        // 统计结果
        for (const auto& lightDetectThreadInfo : lightDetectThreadInfos) {
            if (lightDetectThreadInfo.is_true_light) {
                lights.push_back(lightDetectThreadInfo.light);
            }
        }
    }
}

cv::Mat LightBarDetector::binaryImg(const cv::Mat& img) {
    // 1. 获取灰度图
    
    // 1. 分离BGR通道
    std::vector<cv::Mat> channels;
    cv::split(img, channels);  // channels[0]=B, [1]=G, [2]=R

    // 2. 根据敌方颜色提取对应的通道
    cv::Mat gray_img;
    switch (enemy_color) {
        case Params::RED:
            // 红色装甲板：R通道
            gray_img = channels[2];
            break;
        case Params::BLUE:
            // 蓝色装甲板：B通道
            gray_img = channels[0];
            break;
        case Params::BOTH:
            // 识别两者：R通道和B通道最大值
            gray_img = cv::max(channels[0], channels[2]);
            break;
        default:
            // 默认情况：灰度图
            cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
            break;
    }

    // cv::Mat gray_img;
    // cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    
    // 1. 获取二值图
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, binary_img_THRESHOLD, 255, cv::THRESH_BINARY);

    return binary_img;
}

cv::RotatedRect LightBarDetector::rectExpand(const cv::RotatedRect& rect, float factor) {
    return cv::RotatedRect(
        rect.center, 
        cv::Size2f(rect.size.width * factor, 
                  rect.size.height * factor),
        rect.angle
    );
}

float LightBarDetector::calculateMeanInRotatedRect(const cv::Mat& grayImage, const cv::RotatedRect& rect) {
    // 1. 创建与图像同尺寸的掩码（全黑）
    cv::Mat mask = cv::Mat::zeros(grayImage.size(), CV_8UC1);
    
    // 2. 获取旋转矩形的四个顶点（浮点坐标）
    cv::Point2f vertices2f[4];
    rect.points(vertices2f);
    
    // 3. 将浮点顶点转换为整数顶点
    std::vector<cv::Point> vertices;
    for (int i = 0; i < 4; i++) {
        vertices.push_back(cv::Point(static_cast<int>(vertices2f[i].x), 
                                   static_cast<int>(vertices2f[i].y)));
    }
    
    // 4. 将旋转矩形区域填充为白色（255）
    cv::fillConvexPoly(mask, vertices, cv::Scalar(255));
    
    // 5. 计算掩码区域的均值
    cv::Scalar meanValue = cv::mean(grayImage, mask);
    
    return meanValue[0];  // 灰度图像只有一个通道
}

cv::Mat LightBarDetector::extractColorChannelDiff(const cv::Mat& img) {
    // 1. 分离BGR通道
    std::vector<cv::Mat> channels;
    cv::split(img, channels);  // channels[0]=B, [1]=G, [2]=R

    // 2. 根据敌方颜色提取对应的通道差值
    cv::Mat color_diff;
    switch (enemy_color) {
        case Params::RED:
            // 红色装甲板：R通道减B通道
            cv::subtract(channels[2], channels[0], color_diff);
            cv::threshold(color_diff, color_diff, THRES_MAX_COLOR_RED, 255, cv::THRESH_BINARY);
            break;
        case Params::BLUE:
            // 蓝色装甲板：B通道减R通道
            cv::subtract(channels[0], channels[2], color_diff);
            cv::threshold(color_diff, color_diff, THRES_MAX_COLOR_BLUE, 255, cv::THRESH_BINARY);
            break;
        case Params::BOTH:
            // 识别两者：上述两者最大值
            {
            cv::Mat color_diff_1, color_diff_2;
            cv::subtract(channels[2], channels[0], color_diff_1);
            cv::subtract(channels[0], channels[2], color_diff_2);
            color_diff = cv::max(color_diff_1, color_diff_2);
            }
            break;
        default:
            // 默认情况：G通道减R通道
            cv::subtract(channels[1], channels[0], color_diff);
            break;
    }
    return color_diff;
}

std::vector<cv::RotatedRect> LightBarDetector::detectLightRects(const cv::Mat& img) {
    std::vector<cv::RotatedRect> rects;
    
    // 1. 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 2. 遍历所有轮廓
    for (const auto& contour : contours) {
        // 检查轮廓点数是否足够拟合椭圆
        if (contour.size() < 5) continue;
        
        // 检查轮廓面积
        float area = cv::contourArea(contour);
        if (area < params.light_min_area) continue;

        // 3. 拟合旋转矩形
        cv::RotatedRect rect = cv::fitEllipse(contour);

        // 4. 标准化宽高和角度
        float width = std::min(rect.size.width, rect.size.height);
        float length = std::max(rect.size.width, rect.size.height);
        float angle = rect.angle;
        
        if (rect.size.width > rect.size.height) {
            std::swap(width, length);
            angle += 90;
        }

        // 5. 标准化角度到[-90, 90]
        while (angle > 90) angle -= 180;
        while (angle < -90) angle += 180;

        // 6. 检查几何约束条件
        float ratio = length / width;
        if (ratio >= params.min_light_wh_ratio &&
            ratio <= params.max_light_wh_ratio &&
            length >= params.min_light_height &&
            std::abs(angle) <= params.light_max_tilt_angle) {
            
            // 创建标准化后的旋转矩形
            cv::RotatedRect newRect(rect.center, cv::Size2f(width, length), angle);
            rects.push_back(newRect);
        }
    }

    return rects;
}

void LightBarDetector::processLights() {
    filterLights();    // 过滤不合格的灯条
    updateLights();    // 更新灯条状态（用于追踪）
}

void LightBarDetector::filterLights() {
    // 移除不满足条件的灯条
    lights.erase(std::remove_if(lights.begin(), lights.end(),
        [this](const Light& light) {
            return light.length < params.min_light_height || 
                   light.length / light.width > params.max_light_wh_ratio ||
                   light.length / light.width < params.min_light_wh_ratio;
        }), lights.end());

    // 移除重叠的灯条，只保留较大的灯条
    for (size_t i = 0; i < lights.size(); ++i) {
        for (size_t j = i + 1; j < lights.size(); ++j) {
            if (isOverlap(lights[i], lights[j])) {
                // 保留较大的灯条
                if (lights[i].length * lights[i].width < lights[j].length * lights[j].width) {
                    lights.erase(lights.begin() + i);
                    --i;
                    break;
                } else {
                    lights.erase(lights.begin() + j);
                    --j;
                }
            }
        }
    }
}

bool LightBarDetector::isOverlap(const Light& light1, const Light& light2) {
    // 获取两个灯条的旋转矩形交集区域
    cv::Rect rect1 = light1.el.boundingRect();
    cv::Rect rect2 = light2.el.boundingRect();
    
    // 计算两个矩形的交集
    cv::Rect intersection = rect1 & rect2;
    
    // 如果交集面积大于一定阈值，则认为是重叠的
    return intersection.area() > (rect1.area() * 0.5) || intersection.area() > (rect2.area() * 0.5);
}

void LightBarDetector::updateLights() {
    // 预留用于实现灯条追踪功能
    // TODO: 实现灯条追踪逻辑
}
