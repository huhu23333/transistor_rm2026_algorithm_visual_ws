#include "2d_armor_detector/Armor.h"

#include "3d_processing/ArmorSolver.h"
#include "3d_processing/BallisticSolver.h"
#include "3d_processing/RestFrame.h"

#include "utils/FrameRateCounter.h"
#include "utils/SimpleDataFilter.h"

#include "visualizer/DataVisualizer.h"

#include <rclcpp/rclcpp.hpp>
#include "EKF/Tracker.h"

namespace PredictorType {
    enum PredictorType {
        None = 0,   // 直接瞄准装甲板
        RotationMotionModel,
        AutoSwitch
    };

    extern std::vector<std::string> PredictorTypeStrings;
}  //TODO:预测器大一統，将其它类型去除，即不存在预测器类型

struct PredictorResult {
    bool reset = true;
    float command_pitch = 0.0;
    float command_yaw = 0.0;
    float command_delta_pitch = 0.0;
    float command_delta_yaw = 0.0;
    bool fire_flag = false;
    PredictorType::PredictorType predictor_type = PredictorType::None; //TODO:预测器大一統，将其它类型去除，即不存在预测器类型
    ArmorType::ArmorType armor_type = ArmorType::Hero;
    float pixel_horizontal_center_distance = 1e10;
};

class PredictorEKF {
public:
    PredictorEKF(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node, 
        std::chrono::time_point<std::chrono::steady_clock> node_start_time, 
        std::shared_ptr<ArmorSolver> armor_solver,
        std::shared_ptr<BallisticSolver> ballistic_solver,
        std::shared_ptr<RestFrame> rest_frame, std::shared_ptr<FrameRateCounter> fps_counter,
        ArmorType::ArmorType armor_class,
        std::shared_ptr<Tracker> tracker
        ) : config_file_ptr_(config_file_ptr), node_(node), node_start_time_(node_start_time), 
            armor_solver_(armor_solver_), ballistic_solver_(ballistic_solver_),
            rest_frame_(rest_frame_), fps_counter_(fps_counter), armor_class_(armor_class), tracker_(tracker)
    {
        // 初始化参数
        bullet_velocity_ = (*config_file_ptr)["bullet_velocity_"].as<float>();
        reset_predictor_time_ = (*config_file_ptr)["reset_predictor_time"].as<float>(); 

        // 相机参数
        const YAML::Node& camera_matrix_Node = (*config_file_ptr)["camera_matrix"];
        yaw_rad_to_x_pixel_ratio_ = camera_matrix_Node[0][0].as<float>(); 
        pitch_rad_to_y_pixel_ratio_ = camera_matrix_Node[1][1].as<float>(); 

        last_com_time_ = std::chrono::steady_clock::now();

        oscilloscope_common_ = std::make_shared<Oscilloscope>(640, 480, "Common Debug Oscilloscope", 2);
        oscilloscope_common_ -> setScale(2.0);
        oscilloscope_common_ -> setOffset(-1.0);

        armor_distance_filter_ = std::make_shared<SimpleDataFilter>(1);
        armor_distance_filter_ -> setExponentialAlpha((*config_file_ptr)["armor_distance_smooth_factor"].as<float>());
    }

    PredictorResult step(std::vector<ArmorResult>& classifyResults, cv::Mat& frame, PredictorType::PredictorType control_predictor_type);
    void update_serial_info(float bullet_velocity, float last_pitch_rad_delayed, float last_yaw_rad_delayed, float total_yaw_rad_delayed);

    bool is_reset = false;

private:
    std::shared_ptr<YAML::Node> config_file_ptr_; 
    rclcpp::Node* node_;
    std::chrono::time_point<std::chrono::steady_clock> node_start_time_;
    std::shared_ptr<ArmorSolver> armor_solver_;
    std::shared_ptr<BallisticSolver> ballistic_solver_;
    std::shared_ptr<RestFrame> rest_frame_;
    std::shared_ptr<FrameRateCounter> fps_counter_;

    ArmorType::ArmorType armor_class_;

    float bullet_velocity_;

    float yaw_rad_to_x_pixel_ratio_;
    float pitch_rad_to_y_pixel_ratio_;

    float reset_predictor_time_;

    std::shared_ptr<Tracker> tracker_;

    std::chrono::time_point<std::chrono::steady_clock> last_com_time_;

    std::shared_ptr<Oscilloscope> oscilloscope_common_;

    std::shared_ptr<SimpleDataFilter> armor_distance_filter_;
};