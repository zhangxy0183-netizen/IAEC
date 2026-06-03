import os
import numpy as np
import pandas as pd

# ==== 配置区 ====
PROCESSED_ROOT = "/home/b532root/data/b532zxy/AVEC2017"
LABEL_PATHS = {
    'train': os.path.join(PROCESSED_ROOT, "train_label.csv"),
    'dev':   os.path.join(PROCESSED_ROOT, "dev_label.csv"),
    'test':  os.path.join(PROCESSED_ROOT, "test_label.csv")
}
# 伪图像文件名格式： {pid}_image.npy
# 形状：(15,3,256,256)

def compute_channel_stats(processed_root, train_csv):
    """
    从训练集伪图像中计算每个通道的全局均值和标准差
    """
    df = pd.read_csv(train_csv)
    chan_sum = np.zeros(3, dtype=np.float64)
    chan_sumsq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for pid in df['Participant_ID'].astype(str):
        img_path = os.path.join(processed_root, f"{pid}_P", f"{pid}_image.npy")
        if not os.path.exists(img_path):
            continue
        arr = np.load(img_path).astype(np.float32)  # shape (15,3,256,256)
        # 折叠到 (3, N) 方便计算
        C = arr.shape[1]
        flat = arr.transpose(1,0,2,3).reshape(C, -1)  # (3, 15*256*256)
        chan_sum += flat.sum(axis=1)
        chan_sumsq += (flat**2).sum(axis=1)
        total_pixels += flat.shape[1]

    mean = chan_sum / total_pixels
    var = chan_sumsq / total_pixels - mean**2
    std = np.sqrt(np.maximum(var, 0.0))
    # 防止 std 为 0
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def normalize_images(processed_root, mean, std, label_csv):
    """
    对 split 下的伪图像进行 (x - mean) / std 标准化，并覆盖保存
    """
    df = pd.read_csv(label_csv)
    for pid in df['Participant_ID'].astype(str):
        img_path = os.path.join(processed_root, f"{pid}_P", f"{pid}_image.npy")
        if not os.path.exists(img_path):
            print(f"[跳过] 未找到 {img_path}")
            continue
        arr = np.load(img_path).astype(np.float32)
        # 标准化
        arr = (arr - mean[None,:,None,None]) / std[None,:,None,None]
        # 清除 NaN/Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.save(img_path, arr)
        print(f"[归一化] {pid}: 完成")

if __name__ == "__main__":
    # 1. 计算训练集统计量
    train_csv = LABEL_PATHS['train']
    mean, std = compute_channel_stats(PROCESSED_ROOT, train_csv)
    print("Channel mean:", mean)
    print("Channel std :", std)

    # 2. 分别归一化 train/dev/test
    for split, csv in LABEL_PATHS.items():
        print(f"=== 归一化 {split} 集 ===")
        normalize_images(PROCESSED_ROOT, mean, std, csv)
