import torch
import cv2
import numpy as np
import torch.nn as nn
import os

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
    model = torch.jit.load("model_rm2026.pt", map_location=device)
    
    """ # 加载训练好的模型权重
    model_path = "best_model.pth"  # 或 "final_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"已加载模型权重: {model_path}")
    else:
        print(f"错误: 模型文件 {model_path} 不存在")
        return """
    
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