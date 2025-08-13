import os
import datetime

# 获取当前脚本的绝对路径和所在目录
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# 设置目标文件夹路径
temp_dir = os.path.join(current_dir, 'temp')

# 检查目标文件夹是否存在
if not os.path.exists(temp_dir):
    print(f"错误：目录 '{temp_dir}' 不存在")
    exit(1)

# 生成时间戳前缀（格式示例：20250812_153045_）
timestamp_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")

# 遍历目标文件夹中的所有文件
for filename in os.listdir(temp_dir):
    file_path = os.path.join(temp_dir, filename)
    
    # 跳过子目录，只处理文件
    if os.path.isfile(file_path):
        # 构造新文件名
        new_filename = timestamp_prefix + filename
        new_file_path = os.path.join(temp_dir, new_filename)
        
        try:
            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"重命名成功: {filename} -> {new_filename}")
        except Exception as e:
            print(f"重命名失败: {filename} - {str(e)}")