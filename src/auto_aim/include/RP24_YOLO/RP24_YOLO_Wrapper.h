#include "RP24_YOLO/OpenvinoInfer.h"
#include "memory"
#include "2d_armor_detector/Armor.h"
#include "2d_armor_detector/ArmorTracker.h"
#include <opencv2/opencv.hpp>
#include "macro/AutoAimMacro.h"

class RP24YOLOWrapper {
public:
    RP24YOLOWrapper(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node, string model_path, string device);
    vector<Armor> detectArmors(cv::Mat& frame, string detect_color, vector<int>* rp24_classes = nullptr);
    vector<ArmorResult> detectArmorsWithClassifyAndTrack(cv::Mat& frame, string detect_color, 
        const cv::Point2f& ground_stable_point, vector<Armor>* armors_out = nullptr);

private:
    std::shared_ptr<OpenvinoInfer> infer;
    std::shared_ptr<YAML::Node> config_file_ptr;
    rclcpp::Node* node;
    float lightBarLengthScale = 0.82;

    int class_map[9] = {5, 0, 1, 2, 3, 4, 6, 7, 7};
    bool big_map[9] = {false, true, false, false, false, false, false, false, true};
    std::shared_ptr<ArmorTracker> armor_tracker;
};
