import os
import re
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

def compute_video_stats_for_samples(av_path, modality, sample_names, num_frames=50, select_frames=15):
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

        files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if len(files) < num_frames:
            print(f"样本 {sample} 的图片数量不足 {num_frames} 帧，实际只有 {len(files)} 帧，跳过。")
            continue

        try:
            sorted_files = sorted(files, key=lambda x: int(face_pattern.search(x).group(1)))[:num_frames]
        except Exception as e:
            print(f"样本 {sample} 排序时出错：{e}")
            continue

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

def compute_audio_stats_for_samples(av_path, modality, sample_names, audio_subdir='audio/frame_15'):
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

def compute_split_stats(av_path, split_indices, num_frames=15, select_frames=15):
    ff_dir = os.path.join(av_path, 'Freeform')
    nw_dir = os.path.join(av_path, 'Northwind')
    ff_samples_all = sorted([f for f in os.listdir(ff_dir) 
                             if f not in ['audio', 'audio_npy'] and os.path.isdir(os.path.join(ff_dir, f))])
    nw_samples_all = sorted([f for f in os.listdir(nw_dir) 
                             if f not in ['audio', 'audio_npy'] and os.path.isdir(os.path.join(nw_dir, f))])
    
    ff_samples = [ff_samples_all[i] for i in split_indices if i < len(ff_samples_all)]
    nw_samples = [nw_samples_all[i] for i in split_indices if i < len(nw_samples_all)]
    
    print(f"在 {av_path} 中共选取 {len(ff_samples)} 个 Freeform 样本，{len(nw_samples)} 个 Northwind 样本。")
    
    ff_video_mean, ff_video_std = compute_video_stats_for_samples(av_path, 'Freeform', ff_samples, num_frames, select_frames)
    nw_video_mean, nw_video_std = compute_video_stats_for_samples(av_path, 'Northwind', nw_samples, num_frames, select_frames)
    
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

if __name__ == '__main__':
    train_data_path = '/home/b532root/data/b532zxy/AVEC2014_base/base_train'
    dev_data_path   = '/home/b532root/data/b532zxy/AVEC2014_base/base_dev'
    test_data_path  = '/home/b532root/data/b532zxy/AVEC2014_base/base_test'
    
    ff_dir_train = os.path.join(train_data_path, 'Freeform')
    ff_samples_train = sorted([f for f in os.listdir(ff_dir_train) 
                               if f not in ['audio', 'audio_npy'] and os.path.isdir(os.path.join(ff_dir_train, f))])
    train_indices = list(range(len(ff_samples_train)))
    print("========== Train 数据集统计 ==========")
    stats_train = compute_split_stats(train_data_path, train_indices, num_frames=50, select_frames=15)
    
    ff_dir_dev = os.path.join(dev_data_path, 'Freeform')
    ff_samples_dev = sorted([f for f in os.listdir(ff_dir_dev) 
                             if f not in ['audio', 'audio_npy'] and os.path.isdir(os.path.join(ff_dir_dev, f))])
    dev_indices = list(range(len(ff_samples_dev)))
    print("\n========== Dev 数据集统计 ==========")
    stats_dev = compute_split_stats(dev_data_path, dev_indices, num_frames=50, select_frames=15)

    ff_dir_test = os.path.join(test_data_path, 'Freeform')
    ff_samples_test = sorted([f for f in os.listdir(ff_dir_test) 
                              if f not in ['audio', 'audio_npy'] and os.path.isdir(os.path.join(ff_dir_test, f))])
    test_indices = list(range(len(ff_samples_test)))
    print("\n========== Test 数据集统计 ==========")
    stats_test = compute_split_stats(test_data_path, test_indices, num_frames=50, select_frames=15)