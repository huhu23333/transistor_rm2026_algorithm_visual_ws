import os
from PIL import Image
import imghdr

def remove_corrupted_jpg_files(folder_path):
    """
    删除指定文件夹中所有损坏的JPEG文件
    """
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return

    removed_count = 0
    total_files = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        total_files += 1
        file_path = os.path.join(folder_path, filename)
        
        # 检查文件扩展名（不区分大小写）
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            continue
        
        try:
            # 方法1：使用imghdr检查文件类型
            file_type = imghdr.what(file_path)
            if file_type not in ('jpeg', 'jpg'):
                print(f"损坏的JPEG文件（格式不符）: {filename}")
                os.remove(file_path)
                removed_count += 1
                continue
            
            # 方法2：使用PIL验证JPEG完整性
            with Image.open(file_path) as img:
                img.verify()  # 验证文件完整性
                
            # 方法3：尝试重新打开并转换图像以确保完全可读
            with Image.open(file_path) as img:
                img.convert('RGB')  # 尝试转换为RGB模式
                
        except (IOError, SyntaxError, OSError) as e:
            print(f"损坏的JPEG文件: {filename} - 错误: {str(e)}")
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as remove_error:
                print(f"无法删除文件 {filename}: {str(remove_error)}")
        except Exception as e:
            print(f"处理文件 {filename} 时发生未知错误: {str(e)}")

    print(f"\n处理完成！")
    print(f"扫描文件总数: {total_files}")
    print(f"删除的损坏JPEG文件数: {removed_count}")

if __name__ == "__main__":
    # 设置图片文件夹路径
    images_folder = "./images"
    
    # 执行清理操作
    remove_corrupted_jpg_files(images_folder)
