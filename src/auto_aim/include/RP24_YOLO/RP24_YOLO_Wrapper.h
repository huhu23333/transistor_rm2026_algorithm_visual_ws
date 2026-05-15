#include "RP24_YOLO/OpenvinoInfer.h"
#include "memory"
#include "2d_armor_detector/Armor.h"
#include <opencv2/opencv.hpp>

class RP24YOLOWrapper {
public:
    RP24YOLOWrapper(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node, string model_path, string device);
    vector<Armor> detectArmors(cv::Mat& frame, string detect_color);

private:
    std::shared_ptr<OpenvinoInfer> infer;
    std::shared_ptr<YAML::Node> config_file_ptr;
    rclcpp::Node* node;
    float lightBarLengthScale = 0.8;
};
