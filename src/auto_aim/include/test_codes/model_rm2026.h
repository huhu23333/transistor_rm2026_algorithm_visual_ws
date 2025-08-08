// model_rm2026.h
#pragma once
#include <torch/torch.h>

class TransistorRM2026Net : public torch::nn::Module {
public:
    TransistorRM2026Net(int64_t num_classes = 5) {

        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)));
        conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)));
        conv6 = register_module("conv6", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
        conv7 = register_module("conv7", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
        
        conv1x1 = register_module("conv1x1", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 1)));
        
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(128));
        bn4 = register_module("bn4", torch::nn::BatchNorm2d(128));
        bn5 = register_module("bn5", torch::nn::BatchNorm2d(256));
        bn6 = register_module("bn6", torch::nn::BatchNorm2d(256));
        bn7 = register_module("bn7", torch::nn::BatchNorm2d(256));
        
        bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(512));

        dropout2d_1 = register_module("dropout2d_1", torch::nn::Dropout2d(0.4));
        dropout2d_2 = register_module("dropout2d_2", torch::nn::Dropout2d(0.3));

        gap = register_module("gap", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        dropout = register_module("dropout", torch::nn::Dropout(0.5));
        fc = register_module("fc", torch::nn::Linear(512, 512));

        head1 = register_module("head1", torch::nn::Linear(512, 1));
        head2 = register_module("head2", torch::nn::Linear(512, 1));
        head3 = register_module("head3", torch::nn::Linear(512, 1));
        head4 = register_module("head4", torch::nn::Linear(512, num_classes));
    }

    std::vector<torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(bn1(conv1(x)));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(bn2(conv2(x)));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(bn3(conv3(x)));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(bn4(conv4(x)));
        x = dropout2d_1(x);
        x = torch::relu(bn5(conv5(x)));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(bn6(conv6(x)));
        x = dropout2d_2(x);
        x = torch::relu(bn7(conv7(x)));
        x = torch::relu(bn1x1(conv1x1(x)));
        x = gap(x).squeeze(-1).squeeze(-1);
        x = torch::relu(fc(x));
        x = dropout(x);
        
        return {
            head1(x),
            head2(x),
            head3(x),
            head4(x)
        };
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr},
                      conv5{nullptr}, conv6{nullptr}, conv7{nullptr}, 
                      conv1x1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr},
                           bn5{nullptr}, bn6{nullptr}, bn7{nullptr}, 
                           bn1x1{nullptr};
    torch::nn::AdaptiveAvgPool2d gap{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Dropout2d dropout2d_1{nullptr}, dropout2d_2{nullptr};
    torch::nn::Linear fc{nullptr}, head1{nullptr}, head2{nullptr}, head3{nullptr}, head4{nullptr};
};