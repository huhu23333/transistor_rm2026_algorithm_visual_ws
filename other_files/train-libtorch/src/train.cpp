#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "model_rm2026.h" // 包含提供的模型头文件
#include <torch/data/dataloader.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

// 配置参数
const std::string IMAGE_FOLDER = "./images";
const std::string INDEX_FILE = "index.json";
const std::string TAGS_FOLDER = "user_tags";
const int NUM_EPOCHS = 128;
const int BATCH_SIZE = 32;
const float LEARNING_RATE = 3e-4;
const int NUM_CLASSES = 8;

const int IMG_HEIGHT = 48;
const int IMG_WIDTH = 64;

// 数据结构定义
struct LabelData {
    std::string is_possible;
    std::string has_armor;
    std::string color;
    std::string size;
    std::string not_slant;
    int type;
};

struct DataPair {
    cv::Mat image;
    LabelData label;
};

// 数据集类
class CustomDataset : public torch::data::Dataset<CustomDataset> {
public:
    CustomDataset(std::vector<DataPair> data, bool apply_augmentation = true)
        : data_(std::move(data)), apply_augmentation_(apply_augmentation) {
        // 初始化随机数生成器
        gen_.seed(time(nullptr));
    }
    
    // 获取数据项数量
    torch::data::Example<> get(size_t index) override {
        DataPair item = data_[index];
        cv::Mat image = item.image.clone();
        
        if (apply_augmentation_) {
            // 应用数据增强
            if (uniform_(gen_) < 0.5) {
                rotate180(image);
            }
            randomTranslate(image);
            randomScale(image);
            randomRotate(image);
            addGaussianNoise(image);
            adjustBrightness(image);
            adjustContrast(image);
            colorShift(image);
        }
        
        // 转换图像为Tensor
        torch::Tensor tensor = imageToTensor(image);
        
        // 创建标签Tensor
        std::vector<torch::Tensor> label_tensors;
        label_tensors.push_back(torch::tensor(item.label.has_armor == "yes" ? 1.0f : 0.0f));
        label_tensors.push_back(torch::tensor(item.label.size == "large" ? 1.0f : 0.0f));
        label_tensors.push_back(torch::tensor(item.label.not_slant == "yes" ? 1.0f : 0.0f));
        label_tensors.push_back(torch::tensor(static_cast<float>(item.label.type)));
        
        return {tensor, torch::stack(label_tensors)};
    }
    
    // 返回数据集大小
    torch::optional<size_t> size() const override {
        return data_.size();
    }

private:
    // 数据增强辅助函数
    void rotate180(cv::Mat& img) {
        cv::rotate(img, img, cv::ROTATE_180);
    }
    
    void randomTranslate(cv::Mat& img) {
        if (uniform_(gen_) < 0.9) {
            int max_translate = 5;
            int tx = std::uniform_int_distribution<int>(-max_translate, max_translate)(gen_);
            int ty = std::uniform_int_distribution<int>(-max_translate, max_translate)(gen_);
            
            cv::Mat M = (cv::Mat_<float>(2,3) << 1, 0, tx, 0, 1, ty);
            cv::warpAffine(img, img, M, img.size(), cv::BORDER_REFLECT);
        }
    }
    
    void randomScale(cv::Mat& img) {
        if (uniform_(gen_) < 0.9) {
            float scale_factor = std::uniform_real_distribution<float>(0.9, 1.1)(gen_);
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(), scale_factor, scale_factor);
            
            if (scale_factor < 1.0) {
                int pad_x = (img.cols - resized.cols) / 2;
                int pad_y = (img.rows - resized.rows) / 2;
                cv::copyMakeBorder(resized, img, pad_y, pad_y, pad_x, pad_x, 
                                  cv::BORDER_REFLECT);
                img = img(cv::Rect(0, 0, img.cols, img.rows));
            } else if (scale_factor > 1.0) {
                int start_x = (resized.cols - img.cols) / 2;
                int start_y = (resized.rows - img.rows) / 2;
                img = resized(cv::Rect(start_x, start_y, img.cols, img.rows));
            }
        }
    }
    
    void randomRotate(cv::Mat& img) {
        if (uniform_(gen_) < 0.7) {
            float angle = std::uniform_real_distribution<float>(-5.0, 5.0)(gen_);
            cv::Point2f center(img.cols/2.0, img.rows/2.0);
            cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
            cv::warpAffine(img, img, M, img.size(), cv::BORDER_REFLECT);
        }
    }
    
    void addGaussianNoise(cv::Mat& img) {
        if (uniform_(gen_) < 0.9) {
            float noise_level = std::uniform_real_distribution<float>(1.0, 50.0)(gen_);
            cv::Mat noise(img.size(), img.type());
            cv::randn(noise, 0, noise_level);
            img = img + noise;
        }
    }
    
    void adjustBrightness(cv::Mat& img) {
        if (uniform_(gen_) < 0.9) {
            float delta = std::uniform_real_distribution<float>(-80.0, 80.0)(gen_);
            cv::Mat hsv;
            cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
            
            std::vector<cv::Mat> channels;
            cv::split(hsv, channels);
            channels[2] = channels[2] + delta;
            cv::merge(channels, hsv);
            
            cv::cvtColor(hsv, img, cv::COLOR_HSV2BGR);
        }
    }
    
    void adjustContrast(cv::Mat& img) {
        if (uniform_(gen_) < 0.9) {
            float alpha = std::uniform_real_distribution<float>(0.3, 3.0)(gen_);
            img.convertTo(img, -1, alpha, 0);
        }
    }
    
    void colorShift(cv::Mat& img) {
        if (uniform_(gen_) < 0.9) {
            std::vector<cv::Mat> channels;
            cv::split(img, channels);
            
            for (int i = 0; i < 3; i++) {
                int shift = std::uniform_int_distribution<int>(-80, 80)(gen_);
                channels[i] = channels[i] + shift;
            }
            
            cv::merge(channels, img);
        }
    }
    
    // 图像转Tensor
    torch::Tensor imageToTensor(const cv::Mat& image) {
        cv::Mat float_image;
        image.convertTo(float_image, CV_32FC3);
        
        // 归一化 [-1, 1]
        float_image = (float_image / 127.5) - 1.0;

        cv::resize(float_image, float_image, cv::Size(IMG_WIDTH, IMG_HEIGHT));
        
        // 转换为Tensor (H x W x C -> C x H x W)
        torch::Tensor tensor = torch::from_blob(
            float_image.data, 
            {float_image.rows, float_image.cols, 3}, 
            torch::kFloat32
        ).permute({2, 0, 1});
        
        return tensor.clone(); // 确保数据独立
    }
    
    std::vector<DataPair> data_;
    bool apply_augmentation_;
    std::mt19937 gen_;
    std::uniform_real_distribution<float> uniform_{0.0, 1.0};
};

// 损失函数
std::vector<torch::Tensor> losses_function(
    const std::vector<torch::Tensor>& results, 
    const torch::Tensor& labels) {

    torch::Tensor result_has_armor = results[0].squeeze(1);
    torch::Tensor result_size = results[1].squeeze(1);
    torch::Tensor result_not_slant = results[3].squeeze(1);
    torch::Tensor result_classify = results[4];

    // std::cout << result_has_armor.sizes() << std::endl;
    
    torch::Tensor target_has_armor = labels.index({torch::indexing::Slice(), 0});
    torch::Tensor target_size = labels.index({torch::indexing::Slice(), 1});
    torch::Tensor target_not_slant = labels.index({torch::indexing::Slice(), 2});
    torch::Tensor target_classify = labels.index({torch::indexing::Slice(), 3}).to(torch::kLong);
    
    torch::Tensor mask_armor = (target_has_armor > 0.5).to(torch::kFloat32);
    float averager = mask_armor.sum().item<float>() + 1e-6;
    
    // printf("--------------debug line: %d \n", __LINE__);
    // 计算各项损失
    torch::Tensor loss_has_armor = torch::binary_cross_entropy_with_logits(
        result_has_armor, target_has_armor
    );
    
    // printf("--------------debug line: %d \n", __LINE__);
    torch::Tensor loss_size = (
        torch::binary_cross_entropy_with_logits(
            result_size, target_size, {}, {}, torch::Reduction::None
        ) * mask_armor
    ).sum() / averager;
    
    // printf("--------------debug line: %d \n", __LINE__);
    torch::Tensor loss_not_slant = (
        torch::binary_cross_entropy_with_logits(
            result_not_slant, target_not_slant, {}, {}, torch::Reduction::None
        ) * mask_armor
    ).sum() / averager;
    
    // printf("--------------debug line: %d \n", __LINE__);
    torch::Tensor loss_classify = (
        torch::nll_loss(
            torch::log_softmax(result_classify, 1),
            target_classify, {}, torch::Reduction::None
        ) * mask_armor
    ).sum() / averager;
    
    return {loss_has_armor, loss_size, loss_not_slant, loss_classify};
}

// 主训练函数
template <typename TrainLoader, typename ValLoader>
void train_model(std::shared_ptr<TransistorRM2026Net> model,
                TrainLoader& train_loader,
                ValLoader& val_loader,
                int train_loader_size) {
    
    // 设置设备
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;
    model->to(device);
    
    // 优化器
    torch::optim::Adam optimizer(
        model->parameters(), 
        torch::optim::AdamOptions(LEARNING_RATE).weight_decay(0.001)
    );
    
    // 学习率调度器
    /* torch::optim::ReduceLROnPlateauScheduler scheduler(
        optimizer,
        torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::max,
        0.7, // factor 
        15, // patience 
        1e-4, // threshold 
        torch::optim::ReduceLROnPlateauScheduler::ThresholdMode::rel, // ThresholdMode 
        0, // cooldown 
        std::vector<float>({1e-6}), // min_lr 
        1e-8, // eps 
        false // verbose 
    ); */
    
    float best_val_acc = 0.0;
    
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        // 训练阶段
        model->train();
        float epoch_train_loss = 0.0;
        int batch_count = 0;
        
        for (auto& batch : train_loader) {
            auto data = batch.data.to(device);
            auto labels = batch.target.to(device);
            
            optimizer.zero_grad();
            auto results = model->forward(data);
            
            auto losses = losses_function(results, labels);
            torch::Tensor total_loss = losses[0] + losses[1] + losses[2] + losses[3];
            
            total_loss.backward();
            optimizer.step();
            
            epoch_train_loss += total_loss.item<float>();
            batch_count++;
            
            std::cout << "Epoch [" << (epoch+1) << "/" << NUM_EPOCHS 
                      << "], Batch [" << batch_count << "/" << train_loader_size
                      << "], Loss: " << total_loss.item<float>() << std::endl;
        }
        
        // 验证阶段
        model->eval();
        float epoch_val_loss = 0.0;
        float val_acc_avg = 0.0;
        int val_batch_count = 0;

        float val_acc_has_armor = 0.0;
        float val_acc_size = 0.0;
        float val_acc_not_slant = 0.0;
        float val_acc_classify = 0.0;
        
        for (auto& batch : val_loader) {
            auto data = batch.data.to(device);
            auto labels = batch.target.to(device);
            
            auto results = model->forward(data);
            auto losses = losses_function(results, labels);
            torch::Tensor total_loss = losses[0] + losses[1] + losses[2] + losses[3];
            
            epoch_val_loss += total_loss.item<float>();
            val_batch_count++;
            
            torch::Tensor acc_tensor_has_armor = torch::sigmoid(results[0].squeeze(1)).round().eq(labels.index({torch::indexing::Slice(), 0})).to(torch::kFloat32).mean();
            val_acc_has_armor += acc_tensor_has_armor.item<float>();
            torch::Tensor acc_tensor_size = torch::sigmoid(results[1].squeeze(1)).round().eq(labels.index({torch::indexing::Slice(), 1})).to(torch::kFloat32).mean();
            val_acc_size += acc_tensor_size.item<float>();
            torch::Tensor acc_tensor_not_slant = torch::sigmoid(results[3].squeeze(1)).round().eq(labels.index({torch::indexing::Slice(), 2})).to(torch::kFloat32).mean();
            val_acc_not_slant += acc_tensor_not_slant.item<float>();
            torch::Tensor acc_tensor_classify = (results[4].argmax(1).eq(labels.index({torch::indexing::Slice(), 3}).to(torch::kLong))).to(torch::kFloat32).mean();
            val_acc_classify += acc_tensor_classify.item<float>();
        }
        
        // 计算平均损失和准确率
        epoch_train_loss /= batch_count;
        epoch_val_loss /= val_batch_count;
        val_acc_has_armor /= val_batch_count;
        val_acc_size /= val_batch_count;
        val_acc_not_slant /= val_batch_count;
        val_acc_classify /= val_batch_count;
        val_acc_avg = 0.25 * (val_acc_has_armor + val_acc_size + val_acc_not_slant + val_acc_classify);
        
        // 更新学习率
        //scheduler.step(val_acc_avg);
        
        // 保存最佳模型
        if (val_acc_avg > best_val_acc) {
            best_val_acc = val_acc_avg;
            torch::save(model, "best_model.pt");
            std::cout << "Saved best model with val_acc_avg: " << best_val_acc << std::endl;
        }
        
        std::cout << "Epoch [" << (epoch+1) << "/" << NUM_EPOCHS << "] | "
                  << "Train Loss: " << epoch_train_loss << " | "
                  << "LR: " << optimizer.param_groups()[0].options().get_lr()<< " | "
                  << "Val Loss: " << epoch_val_loss << " | "
                  << "Val Acc (val_acc_avg): " << val_acc_avg  << std::endl;
        std::cout << "val_acc_has_armor: " << val_acc_has_armor << " | "
                  << "val_acc_size: " << val_acc_size << " | "
                  << "val_acc_not_slant: " << val_acc_not_slant << " | "
                  << "val_acc_classify: " << val_acc_classify << std::endl;
    }
    
    // 保存最终模型
    torch::save(model, "final_model.pt");
}

// 加载数据集
std::vector<DataPair> load_dataset() {
    // 加载索引文件
    std::ifstream index_file(INDEX_FILE);
    json index_data = json::parse(index_file);
    
    // 加载用户标签
    std::map<std::string, std::map<std::string, LabelData>> user_tags_data;
    for (const auto& entry : fs::directory_iterator(TAGS_FOLDER)) {
        if (entry.path().extension() == ".jsonl") {
            std::string user = entry.path().stem().string();
            std::ifstream tag_file(entry.path());
            std::string line;
            
            while (std::getline(tag_file, line)) {
                auto data_line = json::parse(line);
                std::string filename = data_line["filename"];
                auto tags = data_line["tags"];
                
                LabelData label;
                label.is_possible = tags["is_possible"];
                
                if (label.is_possible == "yes") {
                    label.has_armor = tags["has_armor"];
                    
                    if (label.has_armor == "yes") {
                        label.color = tags["color"];
                        label.size = tags["size"];
                        label.not_slant = tags["not_slant"];
                        label.type = std::stoi(tags["type"].get<std::string>()) - 1;
                    } else {
                        label.color = "None";
                        label.size = "None";
                        label.not_slant = "None";
                        label.type = 0;
                    }
                } else {
                    label.has_armor = "None";
                    label.color = "None";
                    label.size = "None";
                    label.not_slant = "None";
                    label.type = 0;
                }
                
                user_tags_data[user][filename] = label;
            }
        }
    }
    
    // 创建数据集
    std::vector<DataPair> dataset;
    auto tagged_images = index_data["tagged_images"];
    
    for (auto& [image_name, users] : tagged_images.items()) {
        if (users.empty()) {
            std::cerr << "bad image " << image_name << ": no user" << std::endl;
            continue;
        }
        
        std::string user = users.begin().key();
        if (user_tags_data.find(user) == user_tags_data.end()) {
            std::cerr << "bad image " << image_name << ": user not exist" << std::endl;
            continue;
        }
        
        if (user_tags_data[user].find(image_name) == user_tags_data[user].end()) {
            std::cerr << "bad image " << image_name << ": data not exist" << std::endl;
            continue;
        }
        
        auto& label = user_tags_data[user][image_name];
        if (label.is_possible != "yes") continue;
        
        cv::Mat image = cv::imread(fs::path(IMAGE_FOLDER) / image_name);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_name << std::endl;
            continue;
        }
        
        dataset.push_back({image, label});
    }
    
    std::cout << "Loaded " << dataset.size() << " valid samples" << std::endl;
    return dataset;
}

int main() {
    // 加载数据集
    auto dataset = load_dataset();
    
    // 划分训练集和验证集 (9:1)
    size_t train_size = dataset.size() * 0.9;
    size_t val_size = dataset.size() - train_size;
    
    std::vector<DataPair> train_data(dataset.begin(), dataset.begin() + train_size);
    std::vector<DataPair> val_data(dataset.begin() + train_size, dataset.end());
    
    // 创建数据集和数据加载器
    auto train_dataset = CustomDataset(train_data, true)
        .map(torch::data::transforms::Stack<>());
    
    auto val_dataset = CustomDataset(val_data, false)
        .map(torch::data::transforms::Stack<>());
    
    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE)
    );
    
    auto val_loader = torch::data::make_data_loader(
        std::move(val_dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE)
    );
    
    // 创建模型
    std::shared_ptr<TransistorRM2026Net> model = std::make_shared<TransistorRM2026Net>(NUM_CLASSES);

    int train_loader_size = train_size / BATCH_SIZE + 1;
    
    // 训练模型
    train_model(model, *train_loader, *val_loader, train_loader_size);
    
    return 0;
}