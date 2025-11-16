import json
import jsonlines
import os
from pathlib import Path

# 配置参数
INDEX_FILE = "index.json"  # 索引文件
TAGS_FOLDER = "user_tags"  # 用户标签文件夹

def preview_and_delete_tags():
    """
    预览并删除所有满足条件的标签条目
    删除条件: is_possible="yes" AND has_armor="yes" AND not_slant="no"
    """
    
    print("=" * 60)
    print("标签删除工具")
    print("=" * 60)
    print("删除条件: is_possible='yes' AND has_armor='yes' AND not_slant='no'")
    print("=" * 60)
    
    # 检查用户标签文件夹是否存在
    if not os.path.exists(TAGS_FOLDER):
        print(f"错误: 用户标签文件夹 '{TAGS_FOLDER}' 不存在!")
        return
    
    # 获取所有用户标签文件
    user_tag_files = [f for f in os.listdir(TAGS_FOLDER) if f.endswith('.jsonl')]
    if not user_tag_files:
        print("没有找到用户标签文件!")
        return
    
    print(f"找到 {len(user_tag_files)} 个用户标签文件")
    
    # 收集所有符合条件的记录
    records_to_delete = []
    total_records = 0
    
    for user_file in user_tag_files:
        user_id = user_file.replace('.jsonl', '')
        user_file_path = os.path.join(TAGS_FOLDER, user_file)
        
        try:
            with jsonlines.open(user_file_path) as reader:
                records = list(reader)
                total_records += len(records)
                
                for i, record in enumerate(records):
                    tags = record.get('tags', {})
                    
                    # 检查删除条件
                    if (tags.get('is_possible') == 'yes' and 
                        tags.get('has_armor') == 'yes' and 
                        tags.get('not_slant') == 'no'):
                        
                        records_to_delete.append({
                            'user_id': user_id,
                            'filename': record.get('filename'),
                            'timestamp': record.get('timestamp'),
                            'record_index': i,
                            'file_path': user_file_path
                        })
                        
        except Exception as e:
            print(f"读取文件 {user_file} 时出错: {e}")
    
    print(f"\n扫描完成!")
    print(f"总记录数: {total_records}")
    print(f"符合条件的待删除记录: {len(records_to_delete)}")
    
    if not records_to_delete:
        print("没有找到符合条件的记录，无需删除。")
        return
    
    # 显示预览
    print(f"\n预览前10条待删除记录:")
    print("-" * 60)
    for i, record in enumerate(records_to_delete[:10]):
        print(f"{i+1}. 用户: {record['user_id']}")
        print(f"   文件: {record['filename']}")
        print(f"   时间: {record['timestamp']}")
        print()
    
    if len(records_to_delete) > 10:
        print(f"... 还有 {len(records_to_delete) - 10} 条记录")
    
    # 确认删除
    print("-" * 60)
    confirm = input("确认删除以上所有符合条件的记录吗? (y/N): ").strip().lower()
    
    if confirm != 'y':
        print("操作已取消。")
        return
    
    print("\n开始删除操作...")
    
    # 执行删除操作
    deleted_count = 0
    
    # 按用户文件分组处理
    user_files_to_process = {}
    for record in records_to_delete:
        if record['file_path'] not in user_files_to_process:
            user_files_to_process[record['file_path']] = []
        user_files_to_process[record['file_path']].append(record)
    
    # 处理每个用户文件
    for file_path, records in user_files_to_process.items():
        try:
            # 读取所有记录
            with jsonlines.open(file_path) as reader:
                all_records = list(reader)
            
            # 标记要删除的记录索引（从大到小排序，避免删除时索引变化）
            indices_to_delete = sorted([r['record_index'] for r in records], reverse=True)
            
            # 删除记录
            for index in indices_to_delete:
                if 0 <= index < len(all_records):
                    deleted_record = all_records.pop(index)
                    deleted_count += 1
                    print(f"已删除: {deleted_record.get('filename')} (用户: {records[0]['user_id']})")
            
            # 重新写入文件
            if all_records:
                with jsonlines.open(file_path, mode='w') as writer:
                    for record in all_records:
                        writer.write(record)
            else:
                # 如果文件为空，删除文件
                os.remove(file_path)
                print(f"删除空文件: {file_path}")
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    # 更新索引文件
    print("\n更新索引文件...")
    try:
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, 'r') as f:
                index_data = json.load(f)
            
            # 对于每个被删除的记录，从索引中移除对应的用户标记
            for record in records_to_delete:
                filename = record['filename']
                user_id = record['user_id']
                
                if filename in index_data.get('tagged_images', {}):
                    if user_id in index_data['tagged_images'][filename]:
                        del index_data['tagged_images'][filename][user_id]
                        print(f"从索引中移除: {filename} - 用户 {user_id}")
                    
                    # 如果该图片没有用户标记了，删除整个条目
                    if not index_data['tagged_images'][filename]:
                        del index_data['tagged_images'][filename]
                        print(f"从索引中移除空条目: {filename}")
            
            # 保存更新后的索引文件
            with open(INDEX_FILE, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            print("索引文件更新完成!")
        else:
            print("警告: 索引文件不存在，跳过索引更新")
            
    except Exception as e:
        print(f"更新索引文件时出错: {e}")
    
    print(f"\n删除操作完成!")
    print(f"总共删除了 {deleted_count} 条记录")

if __name__ == "__main__":
    preview_and_delete_tags()