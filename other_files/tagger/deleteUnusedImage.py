import os
import json

# 配置参数（与打标程序保持一致）
IMAGE_FOLDER = "./images"  # 图片文件夹路径
INDEX_FILE = "index.json"  # 索引文件

def get_tagged_images():
    """从索引文件中获取所有已标记的图片"""
    try:
        with open(INDEX_FILE, 'r') as f:
            index_data = json.load(f)
            return list(index_data.get("tagged_images", {}).keys())
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def delete_untagged_images():
    """删除所有未打标签的图片"""
    # 获取所有已标记的图片文件名
    tagged_images = get_tagged_images()
    
    # 获取图片文件夹中的所有文件
    all_images = [f for f in os.listdir(IMAGE_FOLDER) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # 找出未标记的图片
    untagged_images = [img for img in all_images if img not in tagged_images]
    
    # 删除未标记的图片
    deleted_count = 0
    for img in untagged_images:
        img_path = os.path.join(IMAGE_FOLDER, img)
        try:
            os.remove(img_path)
            print(f"已删除未标记图片: {img}")
            deleted_count += 1
        except OSError as e:
            print(f"删除失败: {img} - {str(e)}")
    
    print(f"\n操作完成! 共删除 {deleted_count} 张未标记图片")
    print(f"剩余图片数量: {len(all_images) - deleted_count}")
    print(f"已标记图片数量: {len(tagged_images)}")

if __name__ == "__main__":
    # 安全确认
    confirm = input("确定要删除所有未打标签的图片吗? (y/n): ")
    
    if confirm.lower() == 'y':
        print("开始删除未标记图片...")
        delete_untagged_images()
    else:
        print("操作已取消")