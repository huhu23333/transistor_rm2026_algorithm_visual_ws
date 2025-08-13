import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import json
import jsonlines
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision import transforms as T

IMAGE_FOLDER = "./images"  # 图片文件夹路径
INDEX_FILE = "index.json"  # 索引文件
TAGS_FOLDER = "user_tags"  # 用户标签文件夹
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

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

def load_dataset():
    with open(INDEX_FILE, 'r') as f:
        index_data = json.load(f)
    
    user_tags_dir = os.path.join(current_dir, TAGS_FOLDER)
    user_tags_data = {}
    for user_tags_name in os.listdir(user_tags_dir):
        user_tags_data[user_tags_name.replace(".jsonl", "")] = {}
        with jsonlines.open(os.path.join(user_tags_dir, user_tags_name), mode="r") as reader:
            for data_line in reader:
                tags = data_line["tags"]
                data = {"is_possible": tags["is_possible"]}
                if data["is_possible"] == "yes":
                    data["has_armor"] = tags["has_armor"]
                    if data["has_armor"] == "yes":
                        data["color"] = tags["color"]
                        data["size"] = tags["size"]
                        data["not_slant"] = tags["not_slant"]
                        data["type"] = int(tags["type"])-1
                    else:
                        data["color"] = "None"
                        data["size"] = "None"
                        data["not_slant"] = "None"
                        data["type"] = 0
                else:
                    data["has_armor"] = "None"
                    data["color"] = "None"
                    data["size"] = "None"
                    data["not_slant"] = "None"
                    data["type"] = 0

                user_tags_data[user_tags_name.replace(".jsonl", "")][data_line["filename"]] = data

    dataset = []
    for image_name, users in index_data["tagged_images"].items():
        users = list(users.keys())
        if len(users) == 0:
            print(f"bad image {image_name}: no user")
            continue
        user = users[0]
        if user not in user_tags_data:
            print(f"bad image {image_name}: user not exist")
            continue
        if image_name not in user_tags_data[user]:
            print(f"bad image {image_name}: data not exist")
            continue
        data_pair = {}
        image_dir = os.path.join(current_dir, IMAGE_FOLDER)
        data_pair["image"] = cv2.imread(os.path.join(image_dir, image_name))
        data_pair["label"] = user_tags_data[user][image_name]
        if data_pair["label"]["is_possible"] == "yes":
            dataset.append(data_pair)
    
    return dataset

class CustomDataset(Dataset):
    def __init__(self, data_list, apply_augmentation=True):
        self.data_list = data_list
        self.apply_augmentation = apply_augmentation
        
        if self.apply_augmentation:
            # 创建非空间增强组合
            self.non_spatial_aug = T.Compose([
                T.Lambda(self._add_gaussian_noise),
                T.Lambda(self._adjust_brightness),
                T.Lambda(self._adjust_contrast),
                T.Lambda(self._color_shift)
            ])
            # 创建空间增强组合
            self.spatial_aug = T.Compose([
                T.Lambda(self._random_translate),
                T.Lambda(self._random_scale),
                T.Lambda(self._random_rotate)
            ])
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = item['image'].copy()
        label = item['label']
        
        if self.apply_augmentation:
            # 应用180度旋转
            if random.random() < 0.5:
                image = self._rotate_180(image)
            # 应用空间增强
            image = self.spatial_aug(image)
            # 应用非空间增强
            image = self.non_spatial_aug(image)
            
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = (image_tensor / 127.5) - 1.0
        
        return {'image': image_tensor, 'label': label}
    
    def _rotate_180(self, img):
        return cv2.rotate(img, cv2.ROTATE_180)
    
    # 新增：随机平移（不超过5像素）
    def _random_translate(self, img):
        if random.random() < 0.9:  # 90%的概率应用平移
            max_translate = 5  # 最大平移距离5像素
            rows, cols = img.shape[:2]
            
            # 随机生成x和y方向的平移量
            tx = random.randint(-max_translate, max_translate)
            ty = random.randint(-max_translate, max_translate)
            
            # 创建平移矩阵
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            
            # 应用平移变换（使用反射边界模式避免黑边）
            img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        return img
    
    # 新增：随机缩放（不超过10%）
    def _random_scale(self, img):
        if random.random() < 0.9:  # 90%的概率应用缩放
            scale_factor = random.uniform(0.9, 1.1)  # 缩放范围90%-110%
            rows, cols = img.shape[:2]
            
            # 计算缩放后的尺寸
            new_cols = int(cols * scale_factor)
            new_rows = int(rows * scale_factor)
            
            # 缩放图像
            img = cv2.resize(img, (new_cols, new_rows))
            
            # 如果缩放后尺寸不同，裁剪或填充到原始尺寸
            if scale_factor < 1.0:  # 缩小了，需要填充
                pad_x = (cols - new_cols) // 2
                pad_y = (rows - new_rows) // 2
                img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, 
                                        cv2.BORDER_REFLECT)
                img = img[:rows, :cols]  # 确保尺寸正确
            elif scale_factor > 1.0:  # 放大了，需要裁剪
                start_x = (new_cols - cols) // 2
                start_y = (new_rows - rows) // 2
                img = img[start_y:start_y+rows, start_x:start_x+cols]
            if img.shape[0] != rows or img.shape[1] != cols:
                img = cv2.resize(img, (cols, rows))
        return img
    
    # 新增：随机旋转（不超过5°）
    def _random_rotate(self, img):
        if random.random() < 0.7:  # 70%的概率应用旋转
            max_angle = 5  # 最大旋转角度5°
            angle = random.uniform(-max_angle, max_angle)
            rows, cols = img.shape[:2]
            
            # 创建旋转矩阵（围绕图像中心）
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)
            
            # 应用旋转变换（使用反射边界模式避免黑边）
            img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        return img
    
    # 以下保持原有的非空间增强函数不变
    def _add_gaussian_noise(self, img):
        if random.random() < 0.9: # 0.3
            noise_level = random.uniform(1.0, 50.0) 
            noise = np.random.normal(
                scale=noise_level, 
                size=img.shape
            ).astype(np.int8)
            noise = np.abs(noise).astype(np.uint8)
            return cv2.add(img, noise)
        return img
    
    def _adjust_brightness(self, img):
        if random.random() < 0.9: # 0.5
            delta = random.uniform(-80, 80) # (-30, 30)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.int16)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta, 0, 255)
            hsv = hsv.astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img
    
    def _adjust_contrast(self, img):
        if random.random() < 0.9: # 0.4
            alpha = random.uniform(0.3, 3.0) # (0.8, 1.2)
            return cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        return img
    
    def _color_shift(self, img):
        if random.random() < 0.9:
            shifts = np.random.randint(-80, 80, size=3) # -10 10
            b, g, r = cv2.split(img)
            b = np.clip(b.astype(np.int16) + shifts[0], 0, 255).astype(np.uint8)
            g = np.clip(g.astype(np.int16) + shifts[1], 0, 255).astype(np.uint8)
            r = np.clip(r.astype(np.int16) + shifts[2], 0, 255).astype(np.uint8)
            return cv2.merge((b, g, r))
        return img

def losses_function(results, labels):
    labels_has_armor = labels["has_armor"]
    labels_size = labels["size"]
    labels_not_slant = labels["not_slant"]
    labels_classify = labels["type"]

    results_has_armor = results[0]
    results_size = results[1]
    results_not_slant = results[3]
    results_classify = results[4]
    
    batch_size = len(results_has_armor)
    classes = results_classify.shape[-1]

    target_has_armor = torch.tensor([[1.0] if labels_has_armor[i]=="yes" else [0.0] for i in range(batch_size)])
    target_size = torch.tensor([[1.0] if labels_size[i]=="large" else [0.0] for i in range(batch_size)])
    target_not_slant = torch.tensor([[1.0] if labels_not_slant[i]=="yes" else [0.0] for i in range(batch_size)])
    target_classify = torch.tensor([labels_classify[i] for i in range(batch_size)], dtype=torch.long)

    mask_armor = torch.tensor([1.0 if labels_has_armor[i]=="yes" else 0.0 for i in range(batch_size)])
    averager = mask_armor.sum() + 1e-6

    loss_has_armor = nn.functional.binary_cross_entropy_with_logits(results_has_armor, target_has_armor, reduction='mean')
    loss_size = (nn.functional.binary_cross_entropy_with_logits(results_size, target_size, reduction='none').flatten() * mask_armor).sum() / averager
    loss_not_slant = (nn.functional.binary_cross_entropy_with_logits(results_not_slant, target_not_slant, reduction='none').flatten() * mask_armor).sum() / averager
    loss_classify = (nn.functional.cross_entropy(results_classify, target_classify, reduction='none') * mask_armor).sum() / averager
    
    return [loss_has_armor, loss_size, loss_not_slant, loss_classify]

def loss_function(results, labels):
    weights = {
        "has_armor": 1.0,
        "size": 1.0,
        "not_slant": 1.0,
        "classify": 1.0
    }
    losses = losses_function(results, labels)
    loss = 0
    loss += losses[0] * weights["has_armor"]
    loss += losses[1] * weights["size"]
    loss += losses[2] * weights["not_slant"]
    loss += losses[3] * weights["classify"]
    return loss

def calculate_metrics(results, labels):
    """计算各项任务的准确率"""
    metrics = {}
    
    # 装甲存在检测准确率
    pred_has_armor = torch.sigmoid(results[0]) > 0.5
    true_has_armor = torch.tensor([1 if l == "yes" else 0 for l in labels["has_armor"]])
    metrics["acc_has_armor"] = (pred_has_armor.flatten() == true_has_armor).float().mean().item()
    
    # 装甲大小检测准确率（只考虑存在装甲的样本）
    armor_indices = [i for i, l in enumerate(labels["has_armor"]) if l == "yes"]
    if armor_indices:
        pred_size = torch.sigmoid(results[1][armor_indices]) > 0.5
        true_size = torch.tensor([1 if labels["size"][i] == "large" else 0 for i in armor_indices])
        metrics["acc_size"] = (pred_size.flatten() == true_size).float().mean().item()
    else:
        metrics["acc_size"] = 0.0
    
    # 装甲倾斜检测准确率（只考虑存在装甲的样本）
    if armor_indices:
        pred_not_slant = torch.sigmoid(results[3][armor_indices]) > 0.5
        true_not_slant = torch.tensor([1 if labels["not_slant"][i] == "yes" else 0 for i in armor_indices])
        metrics["acc_not_slant"] = (pred_not_slant.flatten() == true_not_slant).float().mean().item()
    else:
        metrics["acc_not_slant"] = 0.0
    
    # 装甲类型分类准确率（只考虑存在装甲的样本）
    if armor_indices:
        pred_classify = torch.argmax(results[4][armor_indices], dim=1)
        true_classify = torch.tensor([labels["type"][i] for i in armor_indices])
        metrics["acc_classify"] = (pred_classify == true_classify).float().mean().item()
    else:
        metrics["acc_classify"] = 0.0
    
    return metrics

def train_model(model, train_loader, val_loader, num_epochs=128, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 使用SGD优化器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr, 
        #momentum=0.9, 
        weight_decay=0.001,
        #nesterov=True
    )
    
    # 监控验证集平均准确率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.7, 
        patience=15, 
        threshold=1e-4, 
        min_lr=1e-5
    )
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_has_armor_loss': [], 'val_has_armor_loss': [],
        'train_size_loss': [], 'val_size_loss': [],
        'train_not_slant_loss': [], 'val_not_slant_loss': [],
        'train_classify_loss': [], 'val_classify_loss': [],
        'train_has_armor_acc': [], 'val_has_armor_acc': [],
        'train_size_acc': [], 'val_size_acc': [],
        'train_not_slant_acc': [], 'val_not_slant_acc': [],
        'train_classify_acc': [], 'val_classify_acc': [],
        'val_acc_avg': []  # 新增验证集平均准确率
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_metrics = {
            'total_loss': 0.0, 'has_armor_loss': 0.0, 'size_loss': 0.0, 
            'not_slant_loss': 0.0, 'classify_loss': 0.0,
            'has_armor_acc': 0.0, 'size_acc': 0.0, 
            'not_slant_acc': 0.0, 'classify_acc': 0.0
        }
    
        for i, batch in enumerate(train_loader):
            iter_start_time = time.time()
            
            images = batch['image'].to(device)
            labels = batch['label']
            
            optimizer.zero_grad()
            results = model(images)
            
            # 计算损失
            loss = loss_function(results, labels)
            losses = losses_function(results, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算指标
            metrics = calculate_metrics(results, labels)
            
            # 更新训练统计
            train_metrics['total_loss'] += loss.item()
            train_metrics['has_armor_loss'] += losses[0].item()
            train_metrics['size_loss'] += losses[1].item()
            train_metrics['not_slant_loss'] += losses[2].item()
            train_metrics['classify_loss'] += losses[3].item()
            
            train_metrics['has_armor_acc'] += metrics['acc_has_armor']
            train_metrics['size_acc'] += metrics['acc_size']
            train_metrics['not_slant_acc'] += metrics['acc_not_slant']
            train_metrics['classify_acc'] += metrics['acc_classify']
            
            # 计算迭代时间
            iter_time = time.time() - iter_start_time
            
            # 每个batch打印一次
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], '
                    f'Loss: {loss.item():.4f}, '
                    f'Iter Time: {iter_time:.4f}s')
        
        # 计算训练平均指标
        num_batches = len(train_loader)
        history['train_loss'].append(train_metrics['total_loss'] / num_batches)
        history['train_has_armor_loss'].append(train_metrics['has_armor_loss'] / num_batches)
        history['train_size_loss'].append(train_metrics['size_loss'] / num_batches)
        history['train_not_slant_loss'].append(train_metrics['not_slant_loss'] / num_batches)
        history['train_classify_loss'].append(train_metrics['classify_loss'] / num_batches)
        
        history['train_has_armor_acc'].append(train_metrics['has_armor_acc'] / num_batches)
        history['train_size_acc'].append(train_metrics['size_acc'] / num_batches)
        history['train_not_slant_acc'].append(train_metrics['not_slant_acc'] / num_batches)
        history['train_classify_acc'].append(train_metrics['classify_acc'] / num_batches)
        
        # 验证阶段
        model.eval()
        val_metrics = {
            'total_loss': 0.0, 'has_armor_loss': 0.0, 'size_loss': 0.0, 
            'not_slant_loss': 0.0, 'classify_loss': 0.0,
            'has_armor_acc': 0.0, 'size_acc': 0.0, 
            'not_slant_acc': 0.0, 'classify_acc': 0.0,
            'all_preds': [], 'all_targets': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label']
                
                results = model(images)
                
                # 计算损失
                loss = loss_function(results, labels)
                losses = losses_function(results, labels)
                
                # 计算指标
                metrics = calculate_metrics(results, labels)
                
                # 更新验证统计
                val_metrics['total_loss'] += loss.item()
                val_metrics['has_armor_loss'] += losses[0].item()
                val_metrics['size_loss'] += losses[1].item()
                val_metrics['not_slant_loss'] += losses[2].item()
                val_metrics['classify_loss'] += losses[3].item()
                
                val_metrics['has_armor_acc'] += metrics['acc_has_armor']
                val_metrics['size_acc'] += metrics['acc_size']
                val_metrics['not_slant_acc'] += metrics['acc_not_slant']
                val_metrics['classify_acc'] += metrics['acc_classify']
                
                # 收集分类预测用于混淆矩阵
                armor_indices = [i for i, l in enumerate(labels["has_armor"]) if l == "yes"]
                if armor_indices:
                    pred_classify = torch.argmax(results[4][armor_indices], dim=1).cpu().numpy()
                    true_classify = [labels["type"][i] for i in armor_indices]
                    val_metrics['all_preds'].extend(pred_classify)
                    val_metrics['all_targets'].extend(true_classify)
        
        # 计算验证平均指标
        num_batches = len(val_loader)
        history['val_loss'].append(val_metrics['total_loss'] / num_batches)
        history['val_has_armor_loss'].append(val_metrics['has_armor_loss'] / num_batches)
        history['val_size_loss'].append(val_metrics['size_loss'] / num_batches)
        history['val_not_slant_loss'].append(val_metrics['not_slant_loss'] / num_batches)
        history['val_classify_loss'].append(val_metrics['classify_loss'] / num_batches)
        
        history['val_has_armor_acc'].append(val_metrics['has_armor_acc'] / num_batches)
        history['val_size_acc'].append(val_metrics['size_acc'] / num_batches)
        history['val_not_slant_acc'].append(val_metrics['not_slant_acc'] / num_batches)
        history['val_classify_acc'].append(val_metrics['classify_acc'] / num_batches)
        
        # 计算验证集平均准确率（四个任务的平均）
        val_acc_avg = (history['val_has_armor_acc'][-1] + 
                       history['val_size_acc'][-1] + 
                       history['val_not_slant_acc'][-1] + 
                       history['val_classify_acc'][-1]) / 4
        history['val_acc_avg'].append(val_acc_avg)
        
        # 更新学习率（基于验证集平均准确率）
        scheduler.step(val_acc_avg)
        
        # 保存最佳模型（基于验证集平均准确率）
        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with val_acc_avg: {best_val_acc:.4f}")
        
        # 打印epoch结果
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"Train Acc - Has Armor: {history['train_has_armor_acc'][-1]:.4f}, "
              f"Size: {history['train_size_acc'][-1]:.4f}, "
              f"Not Slant: {history['train_not_slant_acc'][-1]:.4f}, "
              f"Classify: {history['train_classify_acc'][-1]:.4f}")
        print(f"Val Acc - Has Armor: {history['val_has_armor_acc'][-1]:.4f}, "
              f"Size: {history['val_size_acc'][-1]:.4f}, "
              f"Not Slant: {history['val_not_slant_acc'][-1]:.4f}, "
              f"Classify: {history['val_classify_acc'][-1]:.4f}, "
              f"Avg: {val_acc_avg:.4f}")  # 新增平均准确率显示
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.7f}\n")
    
    # 绘制训练曲线
    plot_training_history(history)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(val_metrics['all_targets'], val_metrics['all_preds'])
    
    return model, history

def plot_training_history(history):
    plt.figure(figsize=(15, 12))
    
    # 损失曲线
    plt.subplot(3, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 各项损失曲线
    plt.subplot(3, 1, 2)
    plt.plot(history['train_has_armor_loss'], label='Train Has Armor Loss')
    plt.plot(history['val_has_armor_loss'], label='Val Has Armor Loss')
    plt.plot(history['train_size_loss'], label='Train Size Loss')
    plt.plot(history['val_size_loss'], label='Val Size Loss')
    plt.plot(history['train_not_slant_loss'], label='Train Not Slant Loss')
    plt.plot(history['val_not_slant_loss'], label='Val Not Slant Loss')
    plt.plot(history['train_classify_loss'], label='Train Classify Loss')
    plt.plot(history['val_classify_loss'], label='Val Classify Loss')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(3, 1, 3)
    plt.plot(history['train_has_armor_acc'], label='Train Has Armor Acc')
    plt.plot(history['val_has_armor_acc'], label='Val Has Armor Acc')
    plt.plot(history['train_size_acc'], label='Train Size Acc')
    plt.plot(history['val_size_acc'], label='Val Size Acc')
    plt.plot(history['train_not_slant_acc'], label='Train Not Slant Acc')
    plt.plot(history['val_not_slant_acc'], label='Val Not Slant Acc')
    plt.plot(history['train_classify_acc'], label='Train Classify Acc')
    plt.plot(history['val_classify_acc'], label='Val Classify Acc')
    plt.plot(history['val_acc_avg'], 'k--', label='Val Avg Acc', linewidth=2)  # 新增平均准确率曲线
    plt.title('Accuracy Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_training_history(history):
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    # 各项损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_has_armor_loss'], label='Train Has Armor Loss')
    plt.plot(history['val_has_armor_loss'], label='Val Has Armor Loss')
    plt.plot(history['train_size_loss'], label='Train Size Loss')
    plt.plot(history['val_size_loss'], label='Val Size Loss')
    plt.plot(history['train_not_slant_loss'], label='Train Not Slant Loss')
    plt.plot(history['val_not_slant_loss'], label='Val Not Slant Loss')
    plt.plot(history['train_classify_loss'], label='Train Classify Loss')
    plt.plot(history['val_classify_loss'], label='Val Classify Loss')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['train_has_armor_acc'], label='Train Has Armor Acc')
    plt.plot(history['val_has_armor_acc'], label='Val Has Armor Acc')
    plt.plot(history['train_size_acc'], label='Train Size Acc')
    plt.plot(history['val_size_acc'], label='Val Size Acc')
    plt.plot(history['train_not_slant_acc'], label='Train Not Slant Acc')
    plt.plot(history['val_not_slant_acc'], label='Val Not Slant Acc')
    plt.plot(history['train_classify_acc'], label='Train Classify Acc')
    plt.plot(history['val_classify_acc'], label='Val Classify Acc')
    plt.title('Accuracy Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(true_labels, pred_labels):
    if not true_labels or not pred_labels:
        print("No data for confusion matrix")
        return
    
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Armor Classification')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

# 在代码的最后部分，if __name__ == "__main__": 之前添加以下函数

def visualize_label_distribution(dataset):
    """可视化数据集中各种标签的分布"""
    # 收集所有标签
    has_armor_counts = {"yes": 0, "no": 0}
    size_counts = {"large": 0, "small": 0}
    not_slant_counts = {"yes": 0, "no": 0}
    type_counts = {i: 0 for i in range(8)}  # 8种装甲类型
    
    for item in dataset:
        label = item['label']
        has_armor = label["has_armor"]
        has_armor_counts[has_armor] += 1
        
        if has_armor == "yes":
            size = label["size"]
            size_counts[size] += 1
            
            not_slant = label["not_slant"]
            not_slant_counts[not_slant] += 1
            
            armor_type = label["type"]
            type_counts[armor_type] += 1
    
    # 设置图表
    plt.figure(figsize=(16, 12))
    
    # 1. 装甲存在情况分布
    plt.subplot(2, 2, 1)
    plt.bar(has_armor_counts.keys(), has_armor_counts.values(), color=['blue', 'orange'])
    plt.title('Armor Presence Distribution')
    plt.xlabel('Has Armor')
    plt.ylabel('Count')
    for i, v in enumerate(has_armor_counts.values()):
        plt.text(i, v + 5, str(v), ha='center')
    
    # 2. 装甲大小分布
    plt.subplot(2, 2, 2)
    plt.bar(size_counts.keys(), size_counts.values(), color=['green', 'red'])
    plt.title('Armor Size Distribution')
    plt.xlabel('Size')
    plt.ylabel('Count')
    for i, v in enumerate(size_counts.values()):
        plt.text(i, v + 5, str(v), ha='center')
    
    # 3. 是否倾斜分布
    plt.subplot(2, 2, 3)
    plt.bar(not_slant_counts.keys(), not_slant_counts.values(), color=['purple', 'brown'])
    plt.title('Armor Orientation Distribution')
    plt.xlabel('Not Slant')
    plt.ylabel('Count')
    for i, v in enumerate(not_slant_counts.values()):
        plt.text(i, v + 5, str(v), ha='center')
    
    # 4. 装甲类型分布
    plt.subplot(2, 2, 4)
    plt.bar([f'Type {i+1}' for i in range(8)], type_counts.values(), color='teal')
    plt.title('Armor Type Distribution')
    plt.xlabel('Armor Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    for i, v in enumerate(type_counts.values()):
        plt.text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    plt.show()
    plt.close()

def visualize_batch(batch, ncols=4, title=None, save_path=None):
    """
    可视化一个批次的图像和标签
    
    Args:
        batch (dict): 数据加载器返回的批次数据
        ncols (int): 每行显示的图像数量
        title (str): 图像标题
        save_path (str): 保存路径，若为None则显示图像
    """
    images = batch['image']
    labels = batch['label']
    
    # 将Tensor转换为OpenCV格式
    images = images.numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    # 反归一化 [-1,1] -> [0,255]
    images = ((images + 1.0) * 127.5).astype(np.uint8)
    
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    
    plt.figure(figsize=(ncols * 3, nrows * 3))
    if title:
        plt.suptitle(title, fontsize=16)
    
    for i in range(n):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        #plt.title(f'Label: {labels}')
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved batch visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()

# 在 __main__ 部分调用这个函数
if __name__ == "__main__":
    # 加载数据集
    dataset = load_dataset()
    
    # 新增：可视化标签分布
    visualize_label_distribution(dataset)
    
    # 划分训练集和验证集 (9:1)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据集对象
    train_dataset = CustomDataset([dataset[i] for i in train_dataset.indices], apply_augmentation=True)
    val_dataset = CustomDataset([dataset[i] for i in val_dataset.indices], apply_augmentation=False)
    
    # 创建数据加载器（batch_size=32）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # 修改为32
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,  # 修改为32
        shuffle=False
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    for batch in train_loader:
        visualize_batch(batch, ncols=6)
        break
    
    # 初始化模型
    model = TransistorRM2026Net(num_classes=8)
    
    # 训练模型（epochs=256）
    trained_model, history = train_model(model, train_loader, val_loader, num_epochs=128, lr=3e-4)
    
    # 保存最终模型
    torch.save(trained_model.state_dict(), 'final_model.pth')