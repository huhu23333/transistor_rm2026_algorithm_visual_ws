import json
import jsonlines
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_all_tags(tags_folder="user_tags"):
    """加载所有用户的标签数据"""
    all_records = []
    tags_path = Path(tags_folder)
    
    if not tags_path.exists():
        print(f"错误：标签文件夹 '{tags_folder}' 不存在")
        return []
    
    for user_file in tags_path.glob("*.jsonl"):
        try:
            with jsonlines.open(user_file) as reader:
                for record in reader:
                    all_records.append(record)
        except Exception as e:
            print(f"读取文件 {user_file} 时出错: {e}")
    
    return all_records

def analyze_tags(records):
    """分析标签数据并返回统计结果"""
    if not records:
        print("没有找到标签数据")
        return None
    
    # 转换为DataFrame以便分析
    df_data = []
    for record in records:
        tags = record['tags']
        row = {
            'filename': record['filename'],
            'is_possible': tags['is_possible'],
            'has_armor': tags.get('has_armor'),
            'color': tags.get('color'),
            'size': tags.get('size'),
            'not_slant': tags.get('not_slant'),
            'type': tags.get('type')
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # 统计各类别数量
    stats = {}
    
    # 1. 是否为正常图像
    stats['is_possible'] = df['is_possible'].value_counts().to_dict()
    
    # 2. 是否包含装甲板（仅统计正常图像）
    normal_images = df[df['is_possible'] == 'yes']
    stats['has_armor'] = normal_images['has_armor'].value_counts().to_dict()
    
    # 3. 装甲板相关属性（仅统计包含装甲板的正常图像）
    armored_images = normal_images[normal_images['has_armor'] == 'yes']
    
    if not armored_images.empty:
        stats['color'] = armored_images['color'].value_counts().to_dict()
        stats['size'] = armored_images['size'].value_counts().to_dict()
        stats['not_slant'] = armored_images['not_slant'].value_counts().to_dict()
        stats['type'] = armored_images['type'].value_counts().to_dict()
    
    # 总体统计
    stats['total_images'] = len(df)
    stats['normal_images'] = len(normal_images)
    stats['armored_images'] = len(armored_images)
    
    return stats, df

def create_visualizations(stats, output_dir="visualizations"):
    """创建可视化图表"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 设置颜色
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # 1. 是否为正常图像
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 图表1: 图像类型分布
    if 'is_possible' in stats:
        labels = ['正常图像', '后期图像']
        values = [stats['is_possible'].get('yes', 0), stats['is_possible'].get('no', 0)]
        
        axes[0].bar(labels, values, color=colors[:2])
        axes[0].set_title('图像类型分布', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('数量')
        for i, v in enumerate(values):
            axes[0].text(i, v + max(values)*0.01, str(v), ha='center', fontweight='bold')
    
    # 图表2: 装甲板存在情况
    if 'has_armor' in stats:
        labels = ['包含装甲板', '不包含装甲板']
        values = [stats['has_armor'].get('yes', 0), stats['has_armor'].get('no', 0)]
        
        axes[1].bar(labels, values, color=colors[2:4])
        axes[1].set_title('装甲板存在情况\n(仅正常图像)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('数量')
        for i, v in enumerate(values):
            axes[1].text(i, v + max(values)*0.01, str(v), ha='center', fontweight='bold')
    
    # 图表3: 装甲板颜色分布
    if 'color' in stats:
        color_labels = {'blue': '蓝色', 'red': '红色'}
        labels = [color_labels.get(k, k) for k in stats['color'].keys()]
        values = list(stats['color'].values())
        
        axes[2].bar(labels, values, color=['blue', 'red'])
        axes[2].set_title('装甲板颜色分布', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('数量')
        for i, v in enumerate(values):
            axes[2].text(i, v + max(values)*0.01, str(v), ha='center', fontweight='bold')
    
    # 图表4: 装甲板大小分布
    if 'size' in stats:
        size_labels = {'small': '小', 'large': '大'}
        labels = [size_labels.get(k, k) for k in stats['size'].keys()]
        values = list(stats['size'].values())
        
        axes[3].bar(labels, values, color=colors[4:6])
        axes[3].set_title('装甲板大小分布', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('数量')
        for i, v in enumerate(values):
            axes[3].text(i, v + max(values)*0.01, str(v), ha='center', fontweight='bold')
    
    # 图表5: 是否正对装甲板
    if 'not_slant' in stats:
        slant_labels = {'yes': '是', 'no': '否'}
        labels = [slant_labels.get(k, k) for k in stats['not_slant'].keys()]
        values = list(stats['not_slant'].values())
        
        axes[4].bar(labels, values, color=colors[2:4])
        axes[4].set_title('是否正对装甲板', fontsize=14, fontweight='bold')
        axes[4].set_ylabel('数量')
        for i, v in enumerate(values):
            axes[4].text(i, v + max(values)*0.01, str(v), ha='center', fontweight='bold')
    
    # 图表6: 装甲板类型分布
    if 'type' in stats:
        type_labels = {
            '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
            '6': '哨兵', '7': '前哨站', '8': '基地'
        }
        # 按类型排序
        sorted_items = sorted(stats['type'].items(), key=lambda x: x[0])
        labels = [type_labels.get(k, k) for k, v in sorted_items]
        values = [v for k, v in sorted_items]
        
        axes[5].bar(labels, values, color=plt.cm.Set3(np.linspace(0, 1, len(labels))))
        axes[5].set_title('装甲板类型分布', fontsize=14, fontweight='bold')
        axes[5].set_ylabel('数量')
        axes[5].tick_params(axis='x', rotation=45)
        for i, v in enumerate(values):
            axes[5].text(i, v + max(values)*0.01, str(v), ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tagging_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建汇总统计图
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['总图像数', '正常图像', '包含装甲板']
    values = [stats['total_images'], stats['normal_images'], stats['armored_images']]
    
    bars = ax.bar(categories, values, color=['#34495e', '#3498db', '#2ecc71'])
    ax.set_title('标签数据总体统计', fontsize=16, fontweight='bold')
    ax.set_ylabel('数量')
    
    # 在柱子上添加数值
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats

def generate_report(stats, output_dir="visualizations"):
    """生成统计报告"""
    report_path = f"{output_dir}/tagging_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("图像打标数据统计报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总标记图像数量: {stats['total_images']}\n")
        f.write(f"正常图像数量: {stats['normal_images']} ({stats['normal_images']/stats['total_images']*100:.1f}%)\n")
        f.write(f"包含装甲板的图像数量: {stats['armored_images']} ({stats['armored_images']/stats['normal_images']*100:.1f}%)\n\n")
        
        if 'is_possible' in stats:
            f.write("图像类型分布:\n")
            for key, count in stats['is_possible'].items():
                label = "正常图像" if key == 'yes' else "后期图像"
                f.write(f"  {label}: {count}\n")
            f.write("\n")
        
        if 'has_armor' in stats:
            f.write("装甲板存在情况:\n")
            for key, count in stats['has_armor'].items():
                label = "包含装甲板" if key == 'yes' else "不包含装甲板"
                f.write(f"  {label}: {count}\n")
            f.write("\n")
        
        if 'color' in stats:
            f.write("装甲板颜色分布:\n")
            for key, count in stats['color'].items():
                label = "蓝色" if key == 'blue' else "红色"
                f.write(f"  {label}: {count}\n")
            f.write("\n")
        
        if 'size' in stats:
            f.write("装甲板大小分布:\n")
            for key, count in stats['size'].items():
                label = "小" if key == 'small' else "大"
                f.write(f"  {label}: {count}\n")
            f.write("\n")
        
        if 'not_slant' in stats:
            f.write("是否正对装甲板:\n")
            for key, count in stats['not_slant'].items():
                label = "是" if key == 'yes' else "否"
                f.write(f"  {label}: {count}\n")
            f.write("\n")
        
        if 'type' in stats:
            type_labels = {
                '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
                '6': '哨兵', '7': '前哨站', '8': '基地'
            }
            f.write("装甲板类型分布:\n")
            for key, count in sorted(stats['type'].items()):
                label = type_labels.get(key, key)
                f.write(f"  {label}: {count}\n")
    
    print(f"统计报告已保存至: {report_path}")

def main():
    """主函数"""
    print("开始分析打标数据...")
    
    # 加载数据
    records = load_all_tags()
    if not records:
        print("未找到任何标签数据，请确保打标器已经生成了一些标签。")
        return
    
    print(f"成功加载 {len(records)} 条标签记录")
    
    # 分析数据
    stats, df = analyze_tags(records)
    if stats is None:
        return
    
    # 创建可视化
    print("生成可视化图表...")
    stats = create_visualizations(stats)
    
    # 生成报告
    print("生成统计报告...")
    generate_report(stats)
    
    print("分析完成！图表和报告已保存到 'visualizations' 文件夹")

if __name__ == "__main__":
    main()
