import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 数据路径
image_dir = "/home/b532root/data/b532zxy/AVEC2014/face/pretrain/images"
heatmap_dir = "/home/b532root/data/b532zxy/AVEC2014/face/pretrain/heatmaps"
label_csv_path = "/home/b532root/data/b532zxy/AVEC2014/label.csv"

# 输出CSV路径
output_train_csv = "/home/b532root/data/b532zxy/AVEC2014/face/pretrain/train.csv"
output_test_csv = "/home/b532root/data/b532zxy/AVEC2014/face/pretrain/test.csv"

# 读取标签文件
label_df = pd.read_csv(label_csv_path)

# 获取所有图片文件路径
image_files = sorted(os.listdir(image_dir))
heatmap_files = sorted(os.listdir(heatmap_dir))

# 检查匹配关系，并生成路径对
data = []
for image_file in image_files:
    heatmap_file = image_file.replace(".jpg", "_heatmap.npy")
    if heatmap_file in heatmap_files:
        image_path = os.path.join(image_dir, image_file)
        heatmap_path = os.path.join(heatmap_dir, heatmap_file)

        # 从文件名中提取 ID（如 "231_1"）
        file_id = "_".join(image_file.split("_")[1:3])  # 提取第二和第三部分

        # 查找对应的 label
        label_row = label_df[label_df['file'] == file_id]
        if not label_row.empty:
            label = label_row['label'].values[0]
        else:
            print(f"Warning: Label for {file_id} not found.")
            label = None

        data.append((image_path, heatmap_path, label))
    else:
        print(f"Warning: Heatmap for {image_file} not found.")

# 创建 DataFrame
df = pd.DataFrame(data, columns=["image_path", "heatmap_path", "label"])

# 丢弃没有标签的数据
df = df.dropna()

# 按 5:1 分割训练集和测试集
train_df, test_df = train_test_split(df, test_size=1/6, random_state=42)

# 保存为 CSV
train_df.to_csv(output_train_csv, index=False)
test_df.to_csv(output_test_csv, index=False)
