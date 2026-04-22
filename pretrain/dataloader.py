import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import os
import torchvision.transforms as transforms

class ImageHeatmapDataset(Dataset):
    def __init__(self, csv_file, identity_csv, mode):
        """
        Args:
            csv_file (str): 包含 image_path, heatmap_path, label 三列的 CSV 文件路径。
            identity_csv (str): 包含 file 和 identity 两列的 CSV 文件路径。
            mode (str): 'train', 'eval' 或 'test'
        """
        # 读取第一个 CSV
        self.data = pd.read_csv(csv_file)
        
        # 从 image_path 中提取 "203_1" 这种格式的字符串
        # 假设命名规则是 Northwind_203_1_faceX.jpg
        def extract_file_key(path_str):
            filename = os.path.basename(path_str)             # "Northwind_203_1_face1.jpg"
            name_wo_ext = os.path.splitext(filename)[0]       # "Northwind_203_1_face1"
            parts = name_wo_ext.split("_")                    # ["Northwind", "203", "1", "face1"]
            # 这里假设你要把第2和第3个元素拼起来作为 file
            # 如果实际命名不一样，需要做相应修改
            if len(parts) < 3:
                raise ValueError(f"文件名 {filename} 不符合预期的命名规则，无法提取到 'x_y' 格式。")
            return parts[1] + "_" + parts[2]                  # "203_1"

        self.data["file"] = self.data["image_path"].apply(extract_file_key)
        
        # 读取身份映射 CSV（包含 file, identity 两列）
        identity_mapping = pd.read_csv(identity_csv)
        
        # 合并两个 CSV，确保每个样本都有对应的 identity
        self.data = pd.merge(self.data, identity_mapping, on="file", how="left")
        if self.data["identity"].isnull().any():
            # 筛选出 identity 为空的行
            missing = self.data[self.data["identity"].isnull()]
            # 这里可以选择输出行索引或具体的行数据
            raise ValueError(f"存在未匹配到 identity 的样本，请检查文件命名或 identity CSV。错误数据行索引: {missing.index.tolist()}，样本详情:\n{missing}")
        self.mode = mode

        # 定义图像预处理（训练和验证/测试）
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4513, 0.3201, 0.3194],
                                 std=[0.2726, 0.2528, 0.2840]),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop((256, 256), scale=(0.9, 1.0)),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4513, 0.3201, 0.3194],
                                 std=[0.2726, 0.2528, 0.2840]),
        ])

        self.transform = train_transform if self.mode == 'train' else eval_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 从合并后的 DataFrame 中读取各项信息
        image_path = self.data.iloc[idx]["image_path"]
        heatmap_path = self.data.iloc[idx]["heatmap_path"]
        depression_label = self.data.iloc[idx]["label"]  # 抑郁评分
        identity = self.data.iloc[idx]["identity"]       # 身份编号

        # 加载图像
        image = self.transform(Image.open(image_path).convert("RGB"))
        # 加载热力图
        heatmap = np.load(heatmap_path)
        heatmap = torch.tensor(heatmap, dtype=torch.float32)

        return image, heatmap, torch.tensor(depression_label, dtype=torch.float), torch.tensor(identity, dtype=torch.long)

# Dataloader 创建函数
def create_dataloaders(train_csv, val_csv, test_csv, identity_csv, batch_size, num_workers=4):
    train_loader, val_loader, test_loader = None, None, None
    if train_csv:
        train_dataset = ImageHeatmapDataset(csv_file=train_csv, identity_csv=identity_csv, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if val_csv:
        val_dataset = ImageHeatmapDataset(csv_file=val_csv, identity_csv=identity_csv, mode='eval')
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if test_csv:
        test_dataset = ImageHeatmapDataset(csv_file=test_csv, identity_csv=identity_csv, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# Example usage
if __name__ == "__main__":
    train_csv_path = "/home/b532root/data/b532zxy/AVEC15/pretrain/train.csv"
    val_csv_path   = "/home/b532root/data/b532zxy/AVEC15/pretrain/dev.csv"
    test_csv_path  = "/home/b532root/data/b532zxy/AVEC15/pretrain/test.csv"
    identity_csv_path = "/home/b532root/data/b532zxy/AVEC15/file_identity.csv"  # 存储 file, identity 两列

    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=train_csv_path,
        val_csv=val_csv_path,
        test_csv=test_csv_path,
        identity_csv=identity_csv_path,
        batch_size=120
    )

    # 测试 DataLoader 输出
    for images, heatmaps, dep_labels, identities in train_loader:
        print("Image batch shape:", images.size())
        print("Heatmap batch shape:", heatmaps.size())
        print("Depression labels shape:", dep_labels.size())
        print("Identities:", identities)
        break
