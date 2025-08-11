import gradio as gr
import os
import json
import jsonlines
import time
import random
from datetime import datetime
from pathlib import Path
import socket
import uuid

# 配置参数
IMAGE_FOLDER = "./images"  # 图片文件夹路径
INDEX_FILE = "index.json"  # 索引文件
TAGS_FOLDER = "user_tags"  # 用户标签文件夹

def get_user_id(request: gr.Request):
    """获取用户唯一标识符（优先使用IP，如果不可用则生成UUID）"""
    try:
        if request:
            client_ip = request.client.host
            return f"user_{client_ip.replace('.', '_')}"
    except:
        pass
    return f"user_{str(uuid.uuid4())[:8]}"

def get_user_tags_file(user_id):
    """获取用户标签文件路径"""
    Path(TAGS_FOLDER).mkdir(parents=True, exist_ok=True)
    return os.path.join(TAGS_FOLDER, f"{user_id}.jsonl")

def initialize_tagging_system():
    """初始化标签系统，检查并创建必要的文件"""
    # 确保索引文件存在
    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'w') as f:
            json.dump({"tagged_images": {}}, f, indent=2)
    
    # 确保用户标签文件夹存在
    Path(TAGS_FOLDER).mkdir(parents=True, exist_ok=True)

def get_all_images():
    """获取所有图片文件"""
    return [f for f in os.listdir(IMAGE_FOLDER) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

def get_untagged_images(user_id):
    """获取当前用户未标记的图片"""
    all_images = set(get_all_images())
    
    # 加载索引文件
    with open(INDEX_FILE, 'r') as f:
        index_data = json.load(f)
    
    # 获取所有用户已标记的图片
    all_user_tagged = set()
    for img, users in index_data["tagged_images"].items():
        #if user_id in users:
        all_user_tagged.add(img)
    
    # 返回未标记图片
    return list(all_images - all_user_tagged)

def load_random_untagged_image(user_id):
    """加载随机未标记图片"""
    untagged = get_untagged_images(user_id)
    if not untagged:
        return None, "所有图片已标记完成！"
    filename = random.choice(untagged)
    return os.path.join(IMAGE_FOLDER, filename), ""

def save_tag(filename, tag_data, user_id):
    """保存标签到文件"""
    if not filename:
        return "未加载图片！"
    
    img_name = os.path.basename(filename)
    
    # 准备标签数据
    tag_record = {
        "filename": img_name,
        "timestamp": datetime.now().isoformat(),
        "tags": {
            "is_possible": tag_data["is_possible"],
            "has_armor": tag_data["has_armor"] if tag_data["is_possible"] == "yes" else None,
            "color": tag_data["color"] if tag_data["is_possible"] == "yes" and tag_data["has_armor"] == "yes" else None,
            "size": tag_data["size"] if tag_data["is_possible"] == "yes" and tag_data["has_armor"] == "yes" else None,
            "not_slant": tag_data["not_slant"] if tag_data["is_possible"] == "yes" and tag_data["has_armor"] == "yes" else None,
            "type": tag_data["type"] if tag_data["is_possible"] == "yes" and tag_data["has_armor"] == "yes" else None
        }
    }
    
    # 更新索引文件
    with open(INDEX_FILE, 'r') as f:
        index_data = json.load(f)
    
    if img_name not in index_data["tagged_images"]:
        index_data["tagged_images"][img_name] = {}
    
    # 添加用户标记记录
    index_data["tagged_images"][img_name][user_id] = datetime.now().isoformat()
    
    with open(INDEX_FILE, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # 更新用户标签文件
    user_tags_file = get_user_tags_file(user_id)
    with jsonlines.open(user_tags_file, mode='a') as writer:
        writer.write(tag_record)
    
    return "标签已保存！"

def undo_last_tag(user_id):
    """撤回用户最后一次标记"""
    user_tags_file = get_user_tags_file(user_id)
    
    # 检查用户标签文件是否存在
    if not os.path.exists(user_tags_file):
        return None, None, "没有可撤回的标记！"
    
    # 读取所有标签记录
    records = []
    try:
        with jsonlines.open(user_tags_file) as reader:
            for record in reader:
                records.append(record)
    except:
        records = []
    
    if not records:
        return None, None, "没有可撤回的标记！"
    
    # 获取最后一条记录
    last_record = records[-1]
    img_name = last_record["filename"]
    
    # 更新索引文件
    with open(INDEX_FILE, 'r') as f:
        index_data = json.load(f)
    
    if img_name in index_data["tagged_images"]:
        # 删除该用户的标记
        if user_id in index_data["tagged_images"][img_name]:
            del index_data["tagged_images"][img_name][user_id]
            
            # 如果该图片没有其他用户标记，删除整个条目
            if not index_data["tagged_images"][img_name]:
                del index_data["tagged_images"][img_name]
    
    with open(INDEX_FILE, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # 更新用户标签文件 - 移除最后一条记录
    if len(records) > 1:
        with jsonlines.open(user_tags_file, mode='w') as writer:
            for record in records[:-1]:
                writer.write(record)
    else:
        # 如果只有一条记录，直接删除文件
        os.remove(user_tags_file)
    
    # 返回撤回的图片路径
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    return img_path, img_path, f"已撤回对 {img_name} 的标记"

# 初始化标签系统
Path(IMAGE_FOLDER).mkdir(parents=True, exist_ok=True)
initialize_tagging_system()

# 自定义CSS样式，使图片能够自适应放大
custom_css = """
.zoomable-image {
    height: 70vh !important;
    min-height: 500px;
    max-height: 800px;
    width: 100%;
    overflow: auto;
    border: 1px solid #ccc;
    border-radius: 8px;
}
.zoomable-image img {
    max-width: none !important;
    max-height: none !important;
    object-fit: contain;
}
@media (max-width: 768px) {
    .zoomable-image {
        height: 50vh !important;
        min-height: 300px;
    }
}
"""

# 创建Gradio界面
with gr.Blocks(title="图像打标工具", css=custom_css) as app:
    current_image = gr.State(value=None)
    user_id = gr.State(value="")
    
    with gr.Row():
        # 图片区域占更大比例
        with gr.Column(scale=3):
            # 添加container=True参数使图片可放大
            image_display = gr.Image(
                label="当前图像", 
                interactive=False, 
                container=True,
                elem_classes="zoomable-image"
            )
        
        # 标签区域
        with gr.Column(scale=2):
            # 用户ID显示
            user_display = gr.Textbox(label="用户ID", interactive=False)
            
            # 加载按钮和输出框
            with gr.Row():
                load_btn = gr.Button("随机加载未标记图片")
                undo_btn = gr.Button("撤回上一次标记")
            output_box = gr.Textbox(label="状态信息", interactive=False)
            
            # 标签选项
            is_possible = gr.Radio(
                [("是（正常图像）", "yes"), ("否（后期图像）", "no")], 
                label="是否为正常图像（不是水印、图标等后期图像）",
                value="yes"
            )
            
            has_armor = gr.Radio(
                [("是", "yes"), ("否", "no")], 
                label="是否包含装甲板（两光条对应区域（注意不是整个图片）内有装甲板数字/图标，且装甲板两光条均显示）",
                value="yes",
                interactive=True
            )
            
            color = gr.Radio(
                [("蓝色", "blue"), ("红色", "red")], 
                label="装甲板颜色",
                value="blue",
                interactive=True
            )
            
            size = gr.Radio(
                [("小", "small"), ("大", "large")], 
                label="装甲板大小",
                value="small",
                interactive=True
            )
            
            not_slant = gr.Radio(
                [("是", "yes"), ("否", "no")], 
                label="是否正对装甲板（装甲板两光条及数字/图标是否均直立且两光条位置正确对应）",
                value="yes",
                interactive=True
            )
            
            armor_type = gr.Radio(
                [
                    ("1", "1"),
                    ("2", "2"),
                    ("3", "3"),
                    ("4", "4"),
                    ("5", "5"),
                    ("哨兵", "6"),
                    ("前哨站", "7"),
                    ("基地", "8"),
                ], 
                label="装甲板类型",
                value="1",
                interactive=True
            )
            
            next_btn = gr.Button("保存并加载下一张")

    # 初始化用户ID
    def init_user(request: gr.Request):
        user = get_user_id(request)
        return user, user
    
    app.load(init_user, inputs=None, outputs=[user_id, user_display])

    # 交互逻辑：控制选项的可用性
    def toggle_options_is_possible(is_possible_val):
        # 如果是无效图像，禁用所有装甲板相关选项
        if is_possible_val == "no":
            return [
                gr.update(interactive=False, value=None),
                gr.update(interactive=False, value=None),
                gr.update(interactive=False, value=None),
                gr.update(interactive=False, value=None),
                gr.update(interactive=False, value=None)
            ]
        # 否则启用所有选项
        else:
            return [
                gr.update(interactive=True, value="yes"),
                gr.update(interactive=True, value="blue"),
                gr.update(interactive=True, value="small"),
                gr.update(interactive=True, value="yes"),
                gr.update(interactive=True, value="1")
            ]
    def toggle_options_has_armor(has_armor_val):
        # 如果不包含装甲板，禁用装甲板属性选项
        if has_armor_val == "no" or has_armor_val == None:
            return [
                gr.update(interactive=False, value=None),
                gr.update(interactive=False, value=None),
                gr.update(interactive=False, value=None),
                gr.update(interactive=False, value=None)
            ]
        else:
        # 否则启用所有选项
            return [
                gr.update(interactive=True, value="blue"),
                gr.update(interactive=True, value="small"),
                gr.update(interactive=True, value="yes"),
                gr.update(interactive=True, value="1")
            ]
    
    is_possible.change(
        toggle_options_is_possible, 
        inputs=[is_possible],
        outputs=[has_armor, color, size, not_slant, armor_type]
    )
    
    has_armor.change(
        toggle_options_has_armor,
        inputs=[has_armor],
        outputs=[color, size, not_slant, armor_type]
    )

    # 加载按钮功能
    def load_random_image(user_id):
        img_path, msg = load_random_untagged_image(user_id)
        if img_path:
            return img_path, img_path, msg
        return None, current_image.value, msg
    
    load_btn.click(
        load_random_image,
        inputs=[user_id],
        outputs=[image_display, current_image, output_box]
    )

    # 撤回按钮功能
    undo_btn.click(
        undo_last_tag,
        inputs=[user_id],
        outputs=[image_display, current_image, output_box]
    )

    # 下一张按钮功能
    def save_and_next(is_possible, has_armor, color, size, not_slant, armor_type, current_img, user_id):
        if not current_img:
            return None, current_img, "请先加载图片！"
        
        # 保存当前标签
        tag_data = {
            "is_possible": is_possible,
            "has_armor": has_armor,
            "color": color,
            "size": size,
            "not_slant": not_slant,
            "type": armor_type
        }
        
        save_msg = save_tag(current_img, tag_data, user_id)
        
        # 加载新图片
        img_path, new_msg = load_random_untagged_image(user_id)
        if img_path:
            return img_path, img_path, save_msg + " " + new_msg
        return None, None, save_msg + " 所有图片已标记完成！"
    
    next_btn.click(
        save_and_next,
        inputs=[is_possible, has_armor, color, size, not_slant, armor_type, current_image, user_id],
        outputs=[image_display, current_image, output_box]
    )

# 启动应用
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)