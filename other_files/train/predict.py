import torch
import cv2
import numpy as np
import torch.nn as nn
import os

class TransistorRM2026Net(nn.Module):
    def __init__(self, num_classes=8):
        super(TransistorRM2026Net, self).__init__()

        self.activate = nn.ReLU()
        self.pooling = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.conv1x1 = nn.Conv2d(256, 512, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(256)
        
        self.bn1x1 = nn.BatchNorm2d(512)

        self.dropout2d_1 = nn.Dropout2d(0.4)
        self.dropout2d_2 = nn.Dropout2d(0.3)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 512)

        self.head1 = nn.Linear(512, 1)
        self.head2 = nn.Linear(512, 1)
        self.head3 = nn.Linear(512, 1)
        self.head4 = nn.Linear(512, 1)
        self.head5 = nn.Linear(512, num_classes)
        
    def forward(self, x):                                   #64*48*3
        x = self.activate(self.bn1(self.conv1(x)))          #64*48*32
        x = self.pooling(x)                                 #32*24*32
        x = self.activate(self.bn2(self.conv2(x)))          #32*24*64
        x = self.pooling(x)                                 #16*12*64
        x = self.activate(self.bn3(self.conv3(x)))          #16*12*128
        x = self.pooling(x)                                 #8*6*128
        x = self.activate(self.bn4(self.conv4(x)))          #8*6*128
        x = self.dropout2d_1(x)                             #8*6*128
        x = self.activate(self.bn5(self.conv5(x)))          #8*6*256
        x = self.pooling(x)                                 #4*3*256
        x = self.activate(self.bn6(self.conv6(x)))          #4*3*256
        x = self.dropout2d_2(x)                             #4*3*256
        x = self.activate(self.bn7(self.conv7(x)))          #4*3*256
        x = self.activate(self.bn1x1(self.conv1x1(x)))      #4*3*512
        x = self.gap(x).squeeze(-1).squeeze(-1)             #512
        x = self.activate(self.fc(x))                       #512
        x = self.dropout(x)                                 #512
        result1 = self.head1(x)                             #1
        result2 = self.head2(x)                             #1
        result3 = self.head3(x)                             #1
        result4 = self.head4(x)                             #1
        result5 = self.head5(x)                             #num_classes
        return (result1, result2, result3, result4, result5)

def preprocess_image(image_path):
    """预处理图像，与训练时相同"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为张量并归一化
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    image_tensor = (image_tensor / 127.5) - 1.0
    return image_tensor.unsqueeze(0)  # 添加batch维度

def print_raw_predictions(raw_outputs):
    """打印模型的原始输出"""
    print("\n模型原始输出:")
    print(f"1. 装甲存在预测 (原始logits): {raw_outputs[0].item():.4f}")
    print(f"2. 装甲大小预测 (原始logits): {raw_outputs[1].item():.4f}")
    print(f"3. 未使用的输出: {raw_outputs[2].item():.4f}")
    print(f"4. 是否倾斜预测 (原始logits): {raw_outputs[3].item():.4f}")
    
    # 打印分类输出
    print("5. 装甲类型分类 (原始logits):")
    for i, val in enumerate(raw_outputs[4].squeeze().tolist()):
        print(f"   类型 {i+1}: {val:.4f}")

def interpret_predictions(raw_outputs):
    """解释模型预测结果"""
    # 装甲存在预测
    has_armor_prob = torch.sigmoid(raw_outputs[0]).item()
    has_armor = "是" if has_armor_prob > 0.5 else "否"
    
    # 装甲大小预测
    size_prob = torch.sigmoid(raw_outputs[1]).item()
    size = "大" if size_prob > 0.5 else "小"
    
    # 是否倾斜预测
    not_slant_prob = torch.sigmoid(raw_outputs[3]).item()
    not_slant = "是" if not_slant_prob > 0.5 else "否"
    
    # 装甲类型分类
    class_probs = torch.softmax(raw_outputs[4], dim=1).squeeze()
    predicted_class = torch.argmax(class_probs).item() + 1
    max_prob = class_probs[predicted_class-1].item()
    
    print("\n预测结果解释:")
    print(f"装甲存在: {has_armor} (概率: {has_armor_prob:.4f})")
    if has_armor == "是":
        print(f"装甲大小: {size} (概率: {size_prob:.4f})")
        print(f"是否倾斜: {not_slant} (概率: {not_slant_prob:.4f})")
        print(f"装甲类型: {predicted_class} (概率: {max_prob:.4f})")
        print("类型概率分布:")
        for i, prob in enumerate(class_probs.tolist()):
            print(f"  类型 {i+1}: {prob:.4f}")

def main():
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransistorRM2026Net()
    
    # 加载训练好的模型权重
    model_path = "best_model.pth"  # 或 "final_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"已加载模型权重: {model_path}")
    else:
        print(f"错误: 模型文件 {model_path} 不存在")
        return
    
    while True:
        # 获取用户输入的图像路径
        image_path = input("\n请输入图像路径(或输入'q'退出): ").strip()
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print(f"错误: 文件 {image_path} 不存在")
            continue
        
        try:
        #if True:
            # 预处理图像
            input_tensor = preprocess_image(image_path).to(device)
            
            # 进行预测
            with torch.no_grad():
                raw_outputs = model(input_tensor)
            
            # 打印原始输出
            print_raw_predictions([out.squeeze() for out in raw_outputs])
            
            # 解释预测结果
            #interpret_predictions([out.squeeze() for out in raw_outputs])
            
            # 可选：显示图像
            """ show_image = input("是否显示图像? (y/n): ").lower()
            if show_image == 'y':
                img = cv2.imread(image_path)
                cv2.imshow("输入图像", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows() """
                
        except Exception as e:
            print(f"处理图像时出错: {e}")

if __name__ == "__main__":
    main()