import os
from pathlib import Path

def simple_convert(input_dir, output_dir):
    """
    简化版本的YOLO标签转换
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    height_ratio = 125/55
    coords_ratio = 1.5
    ldr = (height_ratio - 1)/2
    cdr = (coords_ratio - 1)/2
    
    for label_file in input_path.glob("*.txt"):
        output_file = output_path / label_file.name
        
        with open(label_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 9:
                    continue
                
                origin_class = int(parts[0])
                if origin_class <= 8:
                    color_class = 0 # 蓝色
                elif origin_class <= 17:
                    color_class = 1 # 红色
                else:
                    color_class = 2 # 无色
                
                # 提取坐标
                origin_coords = list(map(float, parts[1:]))
                points = [(origin_coords[i], origin_coords[i+1]) for i in range(0, 8, 2)]

                armor_points = [((points[0][0]-points[1][0])*ldr+points[0][0], (points[0][1]-points[1][1])*ldr+points[0][1]), 
                                ((points[1][0]-points[0][0])*ldr+points[1][0], (points[1][1]-points[0][1])*ldr+points[1][1]), 
                                ((points[2][0]-points[3][0])*ldr+points[2][0], (points[2][1]-points[3][1])*ldr+points[2][1]), 
                                ((points[3][0]-points[2][0])*ldr+points[3][0], (points[3][1]-points[2][1])*ldr+points[3][1])]
                
                bigger_coords_points = [((armor_points[0][0]-armor_points[3][0])*cdr+armor_points[0][0], (armor_points[0][1]-armor_points[3][1])*cdr+armor_points[0][1]), 
                                        ((armor_points[1][0]-armor_points[2][0])*cdr+armor_points[1][0], (armor_points[1][1]-armor_points[2][1])*cdr+armor_points[1][1]),
                                        ((armor_points[2][0]-armor_points[1][0])*cdr+armor_points[2][0], (armor_points[2][1]-armor_points[1][1])*cdr+armor_points[2][1]),  
                                        ((armor_points[3][0]-armor_points[0][0])*cdr+armor_points[3][0], (armor_points[3][1]-armor_points[0][1])*cdr+armor_points[3][1])]
                
                # 计算包围框
                x_coords = [p[0] for p in bigger_coords_points]
                y_coords = [p[1] for p in bigger_coords_points]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                # 构建新行
                new_line = f"{color_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                for x, y in points:
                    new_line += f" {x:.6f} {y:.6f}"
                new_line += "\n"
                
                f_out.write(new_line)
    
    print("转换完成！")

def convert_format():
    dir_path = os.path.dirname(__file__)
    dataset_path = os.path.join(dir_path, "已标注数据集/已标注数据集")
    if dataset_path[-1] in ["\\", "/"]:
        dataset_path = dataset_path[:-1]
    os.system(f"rm {dataset_path}/train/label/*")
    os.system(f"mv {dir_path}/label/train/* {dataset_path}/train/label/")
    os.system(f"rm {dataset_path}/test/label/*")
    os.system(f"mv {dir_path}/label/test/* {dataset_path}/test/label/")
    os.system(f"rm -r {dir_path}/label")

    os.system(f"mkdir {dataset_path}/images")
    os.system(f"mkdir {dataset_path}/labels")
    os.system(f"mkdir {dataset_path}/images/train")
    os.system(f"mkdir {dataset_path}/images/test")
    os.system(f"mkdir {dataset_path}/labels/train")
    os.system(f"mkdir {dataset_path}/labels/test")
    os.system(f"mv {dataset_path}/train/image/* {dataset_path}/images/train/")
    os.system(f"mv {dataset_path}/train/label/* {dataset_path}/labels/train/")
    os.system(f"mv {dataset_path}/test/image/* {dataset_path}/images/test/")
    os.system(f"mv {dataset_path}/test/label/* {dataset_path}/labels/test/")
    os.system(f"rm -r {dataset_path}/train")
    os.system(f"rm -r {dataset_path}/test")


# 使用方法
if __name__ == "__main__":
    input_folder = "已标注数据集/已标注数据集/train/label/"  # 修改为您的输入路径
    output_folder = "label/train"  # 修改为您的输出路径
    simple_convert(input_folder, output_folder)
    input_folder = "已标注数据集/已标注数据集/test/label/"  # 修改为您的输入路径
    output_folder = "label/test"  # 修改为您的输出路径
    simple_convert(input_folder, output_folder)

    convert_format()
    
