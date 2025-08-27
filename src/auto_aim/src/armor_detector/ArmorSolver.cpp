// ArmorSolver.cpp
#include "armor_detector/ArmorSolver.h"
#include <Eigen/Geometry> // For Quaternion and rotation matrix math

void ArmorSolver::initCameraMatrix(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node) {
    const YAML::Node& camera_matrix_Node = (*config_file_ptr)["camera_matrix"];
    /* RCLCPP_DEBUG(node->get_logger(), "camera_matrix_config: \n[%f, %f, %f,\n %f, %f, %f,\n %f, %f, %f]\n", 
        camera_matrix_Node[0][0].as<double>(), camera_matrix_Node[0][1].as<double>(), camera_matrix_Node[0][2].as<double>(), 
        camera_matrix_Node[1][0].as<double>(), camera_matrix_Node[1][1].as<double>(), camera_matrix_Node[1][2].as<double>(), 
        camera_matrix_Node[2][0].as<double>(), camera_matrix_Node[2][1].as<double>(), camera_matrix_Node[2][2].as<double>()); */
    // 相机内参矩阵
    camera_matrix = (cv::Mat_<double>(3, 3) << 
        camera_matrix_Node[0][0].as<double>(), camera_matrix_Node[0][1].as<double>(), camera_matrix_Node[0][2].as<double>(), 
        camera_matrix_Node[1][0].as<double>(), camera_matrix_Node[1][1].as<double>(), camera_matrix_Node[1][2].as<double>(), 
        camera_matrix_Node[2][0].as<double>(), camera_matrix_Node[2][1].as<double>(), camera_matrix_Node[2][2].as<double>());
    
    const YAML::Node& dist_coeffs_Node = (*config_file_ptr)["dist_coeffs"];
    /* RCLCPP_DEBUG(node->get_logger(), "dist_coeffs_config: %f, %f, %f, %f, %f\n", 
        dist_coeffs_Node[0].as<double>(), dist_coeffs_Node[1].as<double>(), dist_coeffs_Node[2].as<double>(), 
        dist_coeffs_Node[3].as<double>(), dist_coeffs_Node[4].as<double>()); */
    // 畸变系数
    dist_coeffs = (cv::Mat_<double>(1, 5) << 
        dist_coeffs_Node[0].as<double>(), dist_coeffs_Node[1].as<double>(), dist_coeffs_Node[2].as<double>(), 
        dist_coeffs_Node[3].as<double>(), dist_coeffs_Node[4].as<double>());
}

void ArmorSolver::initArmorPoints() {
    // 使用小装甲板尺寸初始化（因为在初始化阶段我们还不知道具体是哪种装甲板）
    const float HALF_WIDTH = ArmorConstants::SMALL_ARMOR_WIDTH / 2.0f;   // 67.5mm
    const float HALF_HEIGHT = ArmorConstants::SMALL_ARMOR_HEIGHT / 2.0f; // 62.5mm
    
    armor_points_3d = {
        cv::Point3f(-HALF_WIDTH, -HALF_HEIGHT, 0.0f),  // 左上
        cv::Point3f(HALF_WIDTH, -HALF_HEIGHT, 0.0f),   // 右上
        cv::Point3f(HALF_WIDTH, HALF_HEIGHT, 0.0f),    // 右下
        cv::Point3f(-HALF_WIDTH, HALF_HEIGHT, 0.0f)    // 左下
    };
}

cv::Point2f ArmorSolver::project3DToPixel(const cv::Point3f& world_point) const {
    // 确保相机参数已初始化
    if (camera_matrix.empty() || dist_coeffs.empty()) {
        throw std::runtime_error("Camera parameters not initialized!");
    }

    // 将3D点转换为OpenCV输入格式
    std::vector<cv::Point3f> object_points = {world_point};
    std::vector<cv::Point2f> image_points;

    // 使用solvePnP投影
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);  // 假设无旋转
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);  // 假设无平移
    
    // 直接使用projectPoints进行投影
    cv::projectPoints(object_points, rvec, tvec, 
                     camera_matrix, dist_coeffs, 
                     image_points);

    return image_points[0];
}

// 修改solveArmor函数实现
AimResult ArmorSolver::solveArmor(const ArmorResult& armor_result) const {
    AimResult result;
    result.valid = false;
    const Armor armor = armor_result.armor;
    int number = armor_result.number;
    try {
        bool is_large_armor = armor_result.is_large;
        
        float half_width = is_large_armor ? 
            ArmorConstants::LARGE_ARMOR_WIDTH / 2.0f :
            ArmorConstants::SMALL_ARMOR_WIDTH / 2.0f;
            
        float half_height = is_large_armor ? 
            ArmorConstants::LARGE_ARMOR_HEIGHT / 2.0f :
            ArmorConstants::SMALL_ARMOR_HEIGHT / 2.0f;
            
        std::vector<cv::Point3f> armor_points_3d = {
            cv::Point3f(-half_width, -half_height, 0.0f),
            cv::Point3f(-half_width, half_height, 0.0f),
            cv::Point3f(half_width, half_height, 0.0f),
            cv::Point3f(half_width, -half_height, 0.0f)
        };

        cv::Mat rvec, tvec;
        bool solve_success = cv::solvePnP(armor_points_3d, armor.corners, 
                                        camera_matrix, dist_coeffs, 
                                        rvec, tvec, false, cv::SOLVEPNP_IPPE);

        if (solve_success) {
            result.valid = true;
            result.position = cv::Point3f(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
            result.rvec = rvec.clone(); // <<--- 填充rvec
            result.yaw = getYawFromRvec(rvec); // <<--- 计算并填充yaw
        } else {
            std::cerr << "PnP solve failed!" << std::endl;
            return result;
        }
        
        // 设置位置信息（相机坐标系下的三维位置）
        result.position = cv::Point3f(tvec.at<double>(0),
                                    tvec.at<double>(1),
                                    tvec.at<double>(2));
        
        // 计算距离
        result.distance = cv::norm(result.position);
        
        // 标记解算成功
        result.valid = true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in solveArmor: " << e.what() << std::endl;
    }
    
    return result;
}

double ArmorSolver::getYawFromRvec(const cv::Mat& rvec) {
    if (rvec.empty()) return 0.0;
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat); // 从旋转向量得到旋转矩阵

    // 根据OpenCV相机坐标系从旋转矩阵直接计算Yaw角
    // Yaw是绕Y轴的旋转，一个稳健的计算方法如下：
    // yaw = atan2(-R(2,0), sqrt(R(0,0)^2 + R(1,0)^2))
    double yaw = std::atan2(-rmat.at<double>(2, 0),
                           std::sqrt(std::pow(rmat.at<double>(0, 0), 2) +
                                     std::pow(rmat.at<double>(1, 0), 2)));
    return yaw;
}