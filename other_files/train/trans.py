import torch
import torch.nn as nn
import cv2
# import onnxruntime
# from openvino.tools import mo
# from openvino.runtime import serialize
import openvino as ov

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

def main():
    model = TransistorRM2026Net()
    model_path = "model_rm2026.pt"
    model.load_state_dict(torch.load(model_path))
    image_path = input("\n请输入图像路径(或输入'q'退出): ").strip()
    image_data = preprocess_image(image_path)
    model.eval()

    with torch.no_grad():
        # torch.onnx.export(model, image_data, 'model_rm2026.onnx',
        #     do_constant_folding=True,
        #     input_names=['input'],
        #     output_names=['output'],
        #     dynamic_axes={
        #         'input': {0: 'batch_size'},
        #         'output': {0: 'batch_size'}
        #     })
        ov_model = ov.convert_model(model, example_input=image_data)
        ov.save_model(ov_model, 'model_rm2026_openvino/model.xml')


if __name__ == "__main__":
    main()