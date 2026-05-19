#include "RP24_YOLO/RP24_YOLO_Wrapper.h"

std::pair<string, string> convertOnnxToIR(const string& onnx_path) {
    ov::Core core;
    cout << "[INFO] Loading ONNX model: " << onnx_path << endl;
    auto model = core.read_model(onnx_path);

    string base_path = onnx_path;
    // 去掉 .onnx 后缀
    size_t dot_pos = base_path.rfind(".onnx");
    if (dot_pos != string::npos) {
        base_path = base_path.substr(0, dot_pos);
    }

    string xml_path = base_path + ".xml";
    string bin_path = base_path + ".bin";

    cout << "[INFO] Serializing model to IR format..." << endl;
    ov::serialize(model, xml_path, bin_path);
    cout << "[INFO] IR files generated: " << xml_path << ", " << bin_path << endl;

    return {xml_path, bin_path};
}

RP24YOLOWrapper::RP24YOLOWrapper(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node, string model_path, string device) 
    : config_file_ptr(config_file_ptr), node(node) {

    // -------------------- Step 1: 模型转换（ONNX -> IR） --------------------
    // 检查模型文件是否存在
    if (FILE* f = fopen(model_path.c_str(), "r")) {
        fclose(f);
    } else {
        cerr << "[ERROR] Model file not found: " << model_path << endl;
        throw runtime_error("Model file not found");
    }

    // 检查是否已经有同名的 .xml 文件（避免重复转换）
    string base_path = model_path;
    size_t dot_pos = base_path.rfind(".onnx");
    if (dot_pos != string::npos) base_path = base_path.substr(0, dot_pos);
    string xml_path_str = base_path + ".xml";
    string bin_path_str = base_path + ".bin";

    // 检查 .xml 和 .bin 是否都已存在
    bool need_convert = true;
    FILE* f_xml = fopen(xml_path_str.c_str(), "r");
    FILE* f_bin = fopen(bin_path_str.c_str(), "r");
    if (f_xml && f_bin) {
        need_convert = false;
        cout << "[INFO] IR files already exist, skipping conversion." << endl;
    }
    if (f_xml) fclose(f_xml);
    if (f_bin) fclose(f_bin);

    if (need_convert) {
        auto [xml_path, bin_path] = convertOnnxToIR(model_path);
        xml_path_str = xml_path;
        bin_path_str = bin_path;
        cout << "[INFO] Model converted to IR format successfully!" << endl;
    }

    // -------------------- Step 2: 初始化推理器 --------------------
    // 使用 OpenvinoInfer 的第一个构造函数
    // （参考 OpenvinoInfer.cpp 中的实现：BGR输入 -> RGB -> 归一化 -> NCHW）
    infer = std::make_shared<OpenvinoInfer>(xml_path_str, bin_path_str, device);
    cout << "[INFO] Inference model loaded successfully!" << endl;

    armor_tracker = std::make_shared<ArmorTracker>(config_file_ptr, node);
}

vector<Armor> RP24YOLOWrapper::detectArmors(cv::Mat& frame, string detect_color, vector<int>* rp24_classes) {
    // 1. 缩放到 640x640 进行推理（模型输入要求 640x640）
    cv::Mat infer_frame;
    cv::resize(frame, infer_frame, cv::Size(640, 640));
    int detect_color_int = (detect_color == "BLUE") ? 0 : ((detect_color == "RED") ? 1 : -1);
    infer -> infer(infer_frame, detect_color_int);

    // 2. 将检测结果的坐标从 640x640 映射回原图尺寸
    float scale_x = (float)frame.cols / 640.0f;
    float scale_y = (float)frame.rows / 640.0f;
    int img_w = frame.cols, img_h = frame.rows;
    vector<Object> objects = infer -> tmp_objects;
    for (auto& obj : objects) {
        obj.rect.x      = (int)(obj.rect.x * scale_x);
        obj.rect.y      = (int)(obj.rect.y * scale_y);
        obj.rect.width  = (int)(obj.rect.width * scale_x);
        obj.rect.height = (int)(obj.rect.height * scale_y);
        for (int i = 0; i < 8; i += 2) {
            obj.landmarks[i]   *= scale_x;
            obj.landmarks[i+1] *= scale_y;
        }
        obj.length *= scale_x;
        obj.width  *= scale_y;

        // // 钳位 rect 到图像范围内，防止绘制时越界
        // obj.rect.x = std::max(0.0f, obj.rect.x);
        // obj.rect.y = std::max(0.0f, obj.rect.y);
        // obj.rect.width  = std::min(obj.rect.width,  img_w - obj.rect.x);
        // obj.rect.height = std::min(obj.rect.height, img_h - obj.rect.y);
    }

    vector<Armor> armors;

    for (Object& object : objects) {
        std::vector<float> frame_keypoints(object.landmarks, object.landmarks + 8);

        RCLCPP_DEBUG(node->get_logger(), "scaled_yolo_data: %f, %f, %f, %f, %f, %f, %f, %f",
            frame_keypoints[0], frame_keypoints[1], frame_keypoints[2], frame_keypoints[3], 
            frame_keypoints[4], frame_keypoints[5], frame_keypoints[6], frame_keypoints[7]);
        cv::Vec2f leftLightBar_lengthVec((frame_keypoints[0] - frame_keypoints[2]), (frame_keypoints[1] - frame_keypoints[3]));
        cv::Vec2f rightLightBar_lengthVec((frame_keypoints[6] - frame_keypoints[4]), (frame_keypoints[7] - frame_keypoints[5]));
        cv::Point2f leftLightBar_center((frame_keypoints[0] + frame_keypoints[2]) / 2.0, (frame_keypoints[1] + frame_keypoints[3]) / 2.0);
        cv::Point2f rightLightBar_center((frame_keypoints[4] + frame_keypoints[6]) / 2.0, (frame_keypoints[5] + frame_keypoints[7]) / 2.0);
        float leftLightBar_length = cv::norm(leftLightBar_lengthVec);
        float rightLightBar_length = cv::norm(rightLightBar_lengthVec);
        cv::Size2f leftLightBar_size(leftLightBar_length * lightBarLengthScale / 8.0, leftLightBar_length * lightBarLengthScale);
        cv::Size2f rightLightBar_size(rightLightBar_length * lightBarLengthScale / 8.0, rightLightBar_length * lightBarLengthScale);
        float leftLightBar_angle = std::atan2(leftLightBar_lengthVec[1], leftLightBar_lengthVec[0]) * 180.0 / M_PI + 90.0;
        float rightLightBar_angle = std::atan2(rightLightBar_lengthVec[1], rightLightBar_lengthVec[0]) * 180.0 / M_PI + 90.0;

        cv::RotatedRect leftLightBar(leftLightBar_center, leftLightBar_size, leftLightBar_angle);
        cv::RotatedRect rightLightBar(rightLightBar_center, rightLightBar_size, rightLightBar_angle);
        armors.emplace_back(leftLightBar, rightLightBar, config_file_ptr, node);

        if (rp24_classes != nullptr) {
            rp24_classes->push_back(object.label);
        }
    }

    return armors;
}

vector<ArmorResult> RP24YOLOWrapper::detectArmorsWithClassifyAndTrack(cv::Mat& frame, string detect_color, 
        const cv::Point2f& ground_stable_point, vector<Armor>* armors_out) {

    vector<int> rp24_classes;
    vector<Armor> armors = detectArmors(frame, detect_color, &rp24_classes);

    armor_tracker -> preProcess(ground_stable_point);
    for (size_t i = 0; i < armors.size(); i++) {
        Armor& armor = armors[i];
        int number = class_map[rp24_classes[i]];
        bool is_large = big_map[rp24_classes[i]];
#ifdef FIX_ARMOR_CLASS
        number = FIX_ARMOR_CLASS;
#endif
        bool not_slant = true;
        float confidence = armor.confidence;

        armor_tracker -> addArmor(armor, number, is_large, not_slant, confidence);
    }

    return armor_tracker -> afterProcess();
}