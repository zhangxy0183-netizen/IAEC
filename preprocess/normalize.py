import os
import re
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# -------------------------------
# 1. 视频帧统计函数（针对指定样本列表）
# -------------------------------
def compute_video_stats_for_samples(av_path, modality, sample_names, num_frames=50, select_frames=15):
    """
    计算指定 modality（'Freeform' 或 'Northwind'）中给定样本的视频帧像素统计量（均值和标准差）。
    
    每个样本中：
      - 从该样本文件夹下读取所有图片（扩展名 .jpg/.jpeg/.png/.bmp），
      - 按文件名中 "face" 后的数字排序后取前 num_frames 帧，
      - 等差采样选取 select_frames 帧（此处均为 15 帧），
      - 使用确定性预处理：Resize 到 (256,256) 并 ToTensor（像素归一化到 [0,1]）。
    """
    video_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # 像素值归一化到 [0,1]
    ])
    face_pattern = re.compile(r'face(\d+)')
    all_images = []

    modality_dir = os.path.join(av_path, modality)
    for sample in sample_names:
        sample_dir = os.path.join(modality_dir, sample)
        if not os.path.isdir(sample_dir):
            continue

        # 列举该样本文件夹中所有图片文件
        files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if len(files) < num_frames:
            print(f"样本 {sample} 的图片数量不足 {num_frames} 帧，实际只有 {len(files)} 帧，跳过。")
            continue

        try:
            # 按照文件名中 face 后的数字排序，取前 num_frames 帧
            sorted_files = sorted(files, key=lambda x: int(face_pattern.search(x).group(1)))[:num_frames]
        except Exception as e:
            print(f"样本 {sample} 排序时出错：{e}")
            continue

        # 等差采样选取 select_frames 帧（此处 select_frames==num_frames=15，即全部帧）
        indices = np.linspace(0, num_frames - 1, select_frames, dtype=int)
        selected_files = [sorted_files[i] for i in indices]
        for file in selected_files:
            img_path = os.path.join(sample_dir, file)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = video_transform(img)  # shape: [3,256,256]
                all_images.append(img_tensor)
            except Exception as e:
                print(f"加载图片 {img_path} 时出错：{e}")
                continue

    if len(all_images) == 0:
        raise ValueError(f"未找到足够图片来计算 {modality} 的视频统计信息，请检查数据。")
    
    all_tensor = torch.stack(all_images, dim=0)  # shape: [N,3,256,256]
    mean = all_tensor.mean(dim=[0, 2, 3])
    std = all_tensor.std(dim=[0, 2, 3])
    return mean, std

# -------------------------------
# 2. 音频特征统计函数（针对指定样本列表）
# -------------------------------
def compute_audio_stats_for_samples(av_path, modality, sample_names, audio_subdir='audio/frame_15'):
    """
    计算指定 modality（'Freeform' 或 'Northwind'）中给定样本的音频特征统计量。
    
    每个样本对应一个 .npy 文件（形状为 (15,128)），
    将所有样本中的特征拉平成一维后，计算总体均值和标准差。
    """
    all_features = []
    audio_dir = os.path.join(av_path, modality, audio_subdir)
    for sample in sample_names:
        file_path = os.path.join(audio_dir, sample + ".npy")
        if not os.path.exists(file_path):
            print(f"音频文件 {file_path} 不存在，跳过。")
            continue
        try:
            feat = np.load(file_path).astype(np.float32)  # shape: (15,128)
        except Exception as e:
            print(f"加载 {file_path} 时出错：{e}")
            continue
        all_features.append(feat.flatten())
    
    if len(all_features) == 0:
        raise ValueError(f"在 {modality} 中没有找到可用于计算音频统计的信息。")
    
    all_features = np.concatenate(all_features, axis=0)
    mean = np.mean(all_features)
    std = np.std(all_features)
    return mean, std

# -------------------------------
# 3. 针对整个数据集（例如 train/dev/test）计算统计数据
# -------------------------------
def compute_split_stats(av_path, split_indices, num_frames=15, select_frames=15):
    """
    给定数据集根目录 av_path（例如 train/dev/test）和样本索引列表 split_indices，
    从 Freeform 与 Northwind 中选取对应样本，计算视频和音频的统计数据。
    
    注意：这里假设 Freeform 和 Northwind 下的样本顺序一致，
    且样本以文件夹形式存放（排除 'audio' 等文件夹）。
    """
    # 获取 Freeform 下所有样本（仅保留文件夹）
    ff_dir = os.path.join(av_path, 'Freeform')
    nw_dir = os.path.join(av_path, 'Northwind')
    ff_samples_all = sorted([f for f in os.listdir(ff_dir) 
                             if f not in ['audio', 'audio_npy'] and os.path.isdir(os.path.join(ff_dir, f))])
    nw_samples_all = sorted([f for f in os.listdir(nw_dir) 
                             if f not in ['audio', 'audio_npy'] and os.path.isdir(os.path.join(nw_dir, f))])
    
    ff_samples = [ff_samples_all[i] for i in split_indices if i < len(ff_samples_all)]
    nw_samples = [nw_samples_all[i] for i in split_indices if i < len(nw_samples_all)]
    
    print(f"在 {av_path} 中共选取 {len(ff_samples)} 个 Freeform 样本，{len(nw_samples)} 个 Northwind 样本。")
    
    # 计算视频帧统计量
    ff_video_mean, ff_video_std = compute_video_stats_for_samples(av_path, 'Freeform', ff_samples, num_frames, select_frames)
    nw_video_mean, nw_video_std = compute_video_stats_for_samples(av_path, 'Northwind', nw_samples, num_frames, select_frames)
    
    # 计算音频特征统计量
    ff_audio_mean, ff_audio_std = compute_audio_stats_for_samples(av_path, 'Freeform', ff_samples, audio_subdir='audio/frame_15')
    nw_audio_mean, nw_audio_std = compute_audio_stats_for_samples(av_path, 'Northwind', nw_samples, audio_subdir='audio/frame_15')
    
    print("----- 视频帧统计 -----")
    print("Freeform 视频均值 (R,G,B):", ff_video_mean)
    print("Freeform 视频标准差:", ff_video_std)
    print("Northwind 视频均值 (R,G,B):", nw_video_mean)
    print("Northwind 视频标准差:", nw_video_std)
    print("")
    print("----- 音频特征统计 -----")
    print("Freeform 音频均值:", ff_audio_mean)
    print("Freeform 音频标准差:", ff_audio_std)
    print("Northwind 音频均值:", nw_audio_mean)
    print("Northwind 音频标准差:", nw_audio_std)
    
    return {
        'ff_video_mean': ff_video_mean,
        'ff_video_std': ff_video_std,
        'nw_video_mean': nw_video_mean,
        'nw_video_std': nw_video_std,
        'ff_audio_mean': ff_audio_mean,
        'ff_audio_std': ff_audio_std,
        'nw_audio_mean': nw_audio_mean,
        'nw_audio_std': nw_audio_std
    }

# -------------------------------
# 4. 针对 train/dev/test 数据集计算统计信息
# -------------------------------
if __name__ == '__main__':
    # 数据集路径（请确保路径正确）
    train_data_path = '/home/b532root/data/b532zxy/AVEC2014_base/base_train'
    dev_data_path   = '/home/b532root/data/b532zxy/AVEC2014_base/base_dev'
    test_data_path  = '/home/b532root/data/b532zxy/AVEC2014_base/base_test'
    
    # -------------------------------
    # 针对 Train 数据集
    # -------------------------------
    ff_dir_train = os.path.join(train_data_path, 'Freeform')
    ff_samples_train = sorted([f for f in os.listdir(ff_dir_train) 
                               if f not in ['audio', 'audio_npy'] and os.path.isdir(os.path.join(ff_dir_train, f))])
    train_indices = list(range(len(ff_samples_train)))
    print("========== Train 数据集统计 ==========")
    stats_train = compute_split_stats(train_data_path, train_indices, num_frames=50, select_frames=15)
    
    # -------------------------------
    # 针对 Dev 数据集
    # -------------------------------
    ff_dir_dev = os.path.join(dev_data_path, 'Freeform')
    ff_samples_dev = sorted([f for f in os.listdir(ff_dir_dev) 
                             if f not in ['audio', 'audio_npy'] and os.path.isdir(os.path.join(ff_dir_dev, f))])
    dev_indices = list(range(len(ff_samples_dev)))
    print("\n========== Dev 数据集统计 ==========")
    stats_dev = compute_split_stats(dev_data_path, dev_indices, num_frames=50, select_frames=15)
    
    # -------------------------------
    # 针对 Test 数据集
    # -------------------------------
    ff_dir_test = os.path.join(test_data_path, 'Freeform')
    ff_samples_test = sorted([f for f in os.listdir(ff_dir_test) 
                              if f not in ['audio', 'audio_npy'] and os.path.isdir(os.path.join(ff_dir_test, f))])
    test_indices = list(range(len(ff_samples_test)))
    print("\n========== Test 数据集统计 ==========")
    stats_test = compute_split_stats(test_data_path, test_indices, num_frames=50, select_frames=15)


# ========== Train 数据集统计 ==========
# 在 /home/b532root/data/b532zxy/AVEC2014/train 中共选取 46 个 Freeform 样本，46 个 Northwind 样本。
# ----- 视频帧统计 -----
# Freeform 视频均值 (R,G,B): tensor([0.4513, 0.3201, 0.3194])
# Freeform 视频标准差: tensor([0.2726, 0.2528, 0.2840])
# Northwind 视频均值 (R,G,B): tensor([0.4539, 0.3249, 0.3231])
# Northwind 视频标准差: tensor([0.2744, 0.2541, 0.2874])

# ----- 音频特征统计 -----
# Freeform 音频均值: 0.03602639
# Freeform 音频标准差: 0.98511666
# Northwind 音频均值: 0.0041907146
# Northwind 音频标准差: 1.0071423

# ========== Dev 数据集统计 ==========
# 在 /home/b532root/data/b532zxy/AVEC2014/dev 中共选取 41 个 Freeform 样本，41 个 Northwind 样本。
# ----- 视频帧统计 -----
# Freeform 视频均值 (R,G,B): tensor([0.4362, 0.3039, 0.3068])
# Freeform 视频标准差: tensor([0.2815, 0.2492, 0.2829])
# Northwind 视频均值 (R,G,B): tensor([0.4501, 0.3185, 0.3162])
# Northwind 视频标准差: tensor([0.2844, 0.2500, 0.2825])

# ----- 音频特征统计 -----
# Freeform 音频均值: 0.050355453
# Freeform 音频标准差: 0.9747763
# Northwind 音频均值: 0.02494597
# Northwind 音频标准差: 0.98393005

# ========== Test 数据集统计 ==========
# 在 /home/b532root/data/b532zxy/AVEC2014/test 中共选取 46 个 Freeform 样本，46 个 Northwind 样本。
# ----- 视频帧统计 -----
# Freeform 视频均值 (R,G,B): tensor([0.4591, 0.3344, 0.3263])
# Freeform 视频标准差: tensor([0.2739, 0.2449, 0.2750])
# Northwind 视频均值 (R,G,B): tensor([0.4672, 0.3416, 0.3362])
# Northwind 视频标准差: tensor([0.2739, 0.2434, 0.2777])

# ----- 音频特征统计 -----
# Freeform 音频均值: 0.020628093
# Freeform 音频标准差: 0.9818894
# Northwind 音频均值: 0.008610017
# Northwind 音频标准差: 0.9842319