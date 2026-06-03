import os
import pandas as pd
from sklearn.model_selection import train_test_split

image_dir = "/home/b532root/data/b532zxy/AVEC2014/face/pretrain/images"
heatmap_dir = "/home/b532root/data/b532zxy/AVEC2014/face/pretrain/heatmaps"
label_csv_path = "/home/b532root/data/b532zxy/AVEC2014/label.csv"

output_train_csv = "/home/b532root/data/b532zxy/AVEC2014/face/pretrain/train.csv"
output_test_csv = "/home/b532root/data/b532zxy/AVEC2014/face/pretrain/test.csv"

label_df = pd.read_csv(label_csv_path)

image_files = sorted(os.listdir(image_dir))
heatmap_files = sorted(os.listdir(heatmap_dir))

data = []
for image_file in image_files:
    heatmap_file = image_file.replace(".jpg", "_heatmap.npy")
    if heatmap_file in heatmap_files:
        image_path = os.path.join(image_dir, image_file)
        heatmap_path = os.path.join(heatmap_dir, heatmap_file)

        file_id = "_".join(image_file.split("_")[1:3])

        label_row = label_df[label_df['file'] == file_id]
        if not label_row.empty:
            label = label_row['label'].values[0]
        else:
            print(f"Warning: Label for {file_id} not found.")
            label = None

        data.append((image_path, heatmap_path, label))
    else:
        print(f"Warning: Heatmap for {image_file} not found.")


df = pd.DataFrame(data, columns=["image_path", "heatmap_path", "label"])
df = df.dropna()
train_df, test_df = train_test_split(df, test_size=1/6, random_state=42)
train_df.to_csv(output_train_csv, index=False)
test_df.to_csv(output_test_csv, index=False)
