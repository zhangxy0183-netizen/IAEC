import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_features_from_dir(feature_dir, file_pattern="*.npy"):
    """
    从目录加载所有的 npy 特征文件，并将其合并成一个大的矩阵。
    """
    feature_files = glob.glob(os.path.join(feature_dir, file_pattern))
    all_features = []
    for file in feature_files:
        features = np.load(file)  # shape: (15, 128)
        all_features.append(features)
    if all_features:
        return np.concatenate(all_features, axis=0)
    else:
        return None

def compute_mean_std(features, feature_name):
    """
    计算并打印特征的均值和标准差，检查是否归一化良好。
    """
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    print(f"{feature_name} 的各维度均值：", means[:10])  # 打印前 10 个维度的均值
    print(f"{feature_name} 的各维度标准差：", stds[:10])  # 打印前 10 个维度的标准差

def main_check():
    """
    加载并统计训练集、验证集和测试集的音频特征分布。
    """
    dirs = {
        "训练集_Freeform": "/home/b532root/data/b532zxy/AVEC15/train/Freeform/audio/frame_15",
        "训练集_Northwind": "/home/b532root/data/b532zxy/AVEC15/train/Northwind/audio/frame_15",
        "验证集_Freeform": "/home/b532root/data/b532zxy/AVEC15/dev/Freeform/audio/frame_15",
        "验证集_Northwind": "/home/b532root/data/b532zxy/AVEC15/dev/Northwind/audio/frame_15",
        "测试集_Freeform": "/home/b532root/data/b532zxy/AVEC15/test/Freeform/audio/frame_15",
        "测试集_Northwind": "/home/b532root/data/b532zxy/AVEC15/test/Northwind/audio/frame_15",
    }
    
    for name, path in dirs.items():
        features = load_features_from_dir(path)
        if features is None:
            print(f"目录 {path} 没有文件")
            continue
        print(f"\n{name} 加载的特征形状：{features.shape}")
        compute_mean_std(features, name)

if __name__ == "__main__":
    main_check()
