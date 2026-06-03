import os
import shutil
import csv
import re

def build_label_dict(source_csv):
    """
    从原始 CSV 构建一个字典，key 为 'file' 列的值，value 为对应的标签。
    假设 CSV 有两列：file,label
    """
    label_dict = {}
    with open(source_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_key = row['file']  # 例如 "203_1"
            label_value = row['label']  # 例如 "5"
            label_dict[file_key] = label_value
    return label_dict

def get_face_number(filename):
    """
    从形如 "xxx_faceN.jpg" 的文件名中提取数字 N 作为排序键。
    如果匹配失败，返回一个较大的数字保证排在后面。
    """
    match = re.search(r'_face(\d+)\.jpg$', filename)
    if match:
        return int(match.group(1))
    else:
        return 999999

def pick_arithmetic_sequence(file_list, k=15):
    """
    从 file_list（已排序）中等差选取 k 个文件。
    若文件数 <= k，则全部返回；否则按等差采样。
    """
    N = len(file_list)
    if N <= k:
        return file_list
    step = (N - 1) / (k - 1)
    chosen = []
    for i in range(k):
        idx = int(round(i * step))
        chosen.append(file_list[idx])
    return chosen

def process_images_and_heatmaps(source_dir, target_image_dir, target_heatmap_dir, prefix, label_dict):
    """
    处理图片和热力图文件，重命名并分类存储，同时返回对应的 [image_path, heatmap_path, label] 列表。
    
    参数：
      source_dir: 原始文件夹路径（如 Freeform 或 Northwind）。
      target_image_dir: 存储所有图片的目标路径。
      target_heatmap_dir: 存储所有热力图的目标路径。
      prefix: 用于区分 Freeform 和 Northwind 的前缀，比如 "Freeform" 或 "Northwind"。
      label_dict: 由 build_label_dict() 返回的字典，用于根据文件名查找标签。
    返回：
      rows: 一个列表，每个元素都是 [new_image_path, new_heatmap_path, label]
    """
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_heatmap_dir, exist_ok=True)
    
    rows = []
    # 遍历 source_dir 下的所有子文件夹
    for folder in sorted(os.listdir(source_dir)):
        folder_path = os.path.join(source_dir, folder)
        # 跳过无关或非目录
        if folder in ["audio", "audio_npy"] or not os.path.isdir(folder_path):
            continue

        # 收集该子文件夹下所有 .jpg 文件
        jpg_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".jpg")]
        # 按照 face 数字排序（正确的数值排序）
        jpg_files.sort(key=get_face_number)
        # 按等差方式选取 15 张
        chosen_jpg_files = pick_arithmetic_sequence(jpg_files, k=15)

        for file_name in chosen_jpg_files:
            file_path = os.path.join(folder_path, file_name)
            new_image_name = f"{prefix}_{file_name}"
            new_image_path = os.path.join(target_image_dir, new_image_name)
            shutil.copy(file_path, new_image_path)

            # 对应 heatmap 文件在子文件夹的 heatmaps 目录下
            heatmap_folder = os.path.join(folder_path, "heatmaps")
            heatmap_name = file_name.replace(".jpg", "_heatmap.npy")
            heatmap_path = os.path.join(heatmap_folder, heatmap_name)
            
            new_heatmap_path = ""
            if os.path.exists(heatmap_path):
                new_heatmap_name = f"{prefix}_{heatmap_name}"
                new_heatmap_path = os.path.join(target_heatmap_dir, new_heatmap_name)
                shutil.copy(heatmap_path, new_heatmap_path)
            else:
                print(f"Warning: Heatmap not found for image {file_path}")

            # 从文件名中提取关键字（例如 "203_1"）
            match = re.search(r'(\d+_\d+)', file_name)
            found_label = None
            if match:
                key = match.group(1)
                if key in label_dict:
                    found_label = label_dict[key]
            if found_label is None:
                found_label = "0"  # 默认标签
            rows.append([new_image_path, new_heatmap_path, found_label])
    
    return rows

def main():
    # 定义目标路径（不变）
    target_image_dir = "/home/b532root/data/b532zxy/AVEC2014/pretrain/images"
    target_heatmap_dir = "/home/b532root/data/b532zxy/AVEC2014/pretrain/heatmaps"

    # 定义需要处理的 6 个文件夹及对应信息
    # 注意：这里的 output_csv 文件名后缀将以各自的目录名称附加（例如 test_Northwind.csv）
    tasks = [
        {
            "source_dir": "/home/b532root/data/b532zxy/AVEC2014/face/test/Northwind",
            "source_csv": "/home/b532root/data/b532zxy/AVEC2014/face/test/test_label.csv",
            "prefix": "Northwind",
            "output_csv": "/home/b532root/data/b532zxy/AVEC2014/pretrain/test.csv"
        },
        {
            "source_dir": "/home/b532root/data/b532zxy/AVEC2014/face/test/Freeform",
            "source_csv": "/home/b532root/data/b532zxy/AVEC2014/face/test/test_label.csv",
            "prefix": "Freeform",
            "output_csv": "/home/b532root/data/b532zxy/AVEC2014/pretrain/test.csv"
        },
        {
            "source_dir": "/home/b532root/data/b532zxy/AVEC2014/face/train/Northwind",
            "source_csv": "/home/b532root/data/b532zxy/AVEC2014/face/train/train_label.csv",
            "prefix": "Northwind",
            "output_csv": "/home/b532root/data/b532zxy/AVEC2014/pretrain/train.csv"
        },
        {
            "source_dir": "/home/b532root/data/b532zxy/AVEC2014/face/train/Freeform",
            "source_csv": "/home/b532root/data/b532zxy/AVEC2014/face/train/train_label.csv",
            "prefix": "Freeform",
            "output_csv": "/home/b532root/data/b532zxy/AVEC2014/pretrain/train.csv"
        },
        {
            "source_dir": "/home/b532root/data/b532zxy/AVEC2014/face/dev/Northwind",
            "source_csv": "/home/b532root/data/b532zxy/AVEC2014/face/dev/dev_label.csv",
            "prefix": "Northwind",
            "output_csv": "/home/b532root/data/b532zxy/AVEC2014/pretrain/dev.csv"
        },
        {
            "source_dir": "/home/b532root/data/b532zxy/AVEC2014/face/dev/Freeform",
            "source_csv": "/home/b532root/data/b532zxy/AVEC2014/face/dev/dev_label.csv",
            "prefix": "Freeform",
            "output_csv": "/home/b532root/data/b532zxy/AVEC2014/pretrain/dev.csv"
        }
    ]

    for task in tasks:
        source_dir = task["source_dir"]
        source_csv = task["source_csv"]
        prefix = task["prefix"]
        output_csv = task["output_csv"]

        print(f"Processing folder: {source_dir}")
        label_dict = build_label_dict(source_csv)
        rows = process_images_and_heatmaps(source_dir, target_image_dir, target_heatmap_dir, prefix, label_dict)
        
        # 如果 CSV 文件已经存在，则以追加模式打开，否则新建并写入表头
        if os.path.exists(output_csv):
            mode = 'a'
            write_header = False
        else:
            mode = 'w'
            write_header = True

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["image_path", "heatmap_path", "label"])
            writer.writerows(rows)
        print(f"Processing completed for {source_dir}!")
        print(f"New CSV appended at: {output_csv}\n")

if __name__ == "__main__":
    main()
