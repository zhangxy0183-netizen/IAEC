import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import re
import numpy as np
import pandas as pd
from PIL import Image
import librosa
import random
import opensmile
import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead")
warnings.filterwarnings("ignore", category=FutureWarning)

class AudioVideoDataset(Dataset):
    def __init__(self, av_path, label_path, indices=None, num_frames=50, mode='train'):
        self.av_path = av_path
        self.label = pd.read_csv(label_path)
        self.num_frames = num_frames
        self.mode = mode
        self.face_pattern = re.compile(r'face(\d+)')
        self.freeform_pictures = sorted([f for f in os.listdir(os.path.join(av_path, 'Freeform'))
                                         if f not in ['audio', 'audio_npy']])
        self.northwind_pictures = sorted([f for f in os.listdir(os.path.join(av_path, 'Northwind'))
                                          if f not in ['audio', 'audio_npy']])
        train_id_path = "/home/b532root/data/b532zxy/AVEC15/train/Freeform"
        test_id_path = "/home/b532root/data/b532zxy/AVEC15/test/Freeform"
        dev_id_path = "/home/b532root/data/b532zxy/AVEC15/dev/Freeform"

        # 定义需要排除的文件夹名称
        exclude = {'audio'}

        def get_id_folders(path):
            return [f for f in os.listdir(path)
                    if f not in exclude and os.path.isdir(os.path.join(path, f))]

        # 分别从三个路径获取文件夹名称列表
        train_ids = get_id_folders(train_id_path)
        test_ids = get_id_folders(test_id_path)
        dev_ids = get_id_folders(dev_id_path)

        # 合并三个列表去重后排序
        id_set = sorted(set(train_ids + test_ids + dev_ids))
        person_ids = sorted(set([name.split('_')[0] for name in id_set]))
        self.person_id_to_index = {pid: idx for idx, pid in enumerate(person_ids)}
        self.num_id_classes = len(self.person_id_to_index)
        self.file_to_label = dict(zip(self.label['file'], self.label['label']))
        # 如果提供了 indices，则只保留指定的索引
        if indices is not None:
            self.freeform_pictures = [self.freeform_pictures[i] for i in indices]
            self.northwind_pictures = [self.northwind_pictures[i] for i in indices]

        if self.mode == 'train':
            # 对训练图像增强，先在 PIL 图像上进行变换，再转换为 Tensor、归一化，最后随机遮挡部分区域
            self.ff_video_transform = transforms.Compose([
                transforms.RandomResizedCrop((256, 256), scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4513, 0.3201, 0.3194], std=[0.2726, 0.2528, 0.2840]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ])
            self.nw_video_transform = transforms.Compose([
                transforms.RandomResizedCrop((256, 256), scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4539, 0.3249, 0.3231], std=[0.2744, 0.2541, 0.2874]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ])
        else:
            ff_eval_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4513, 0.3201, 0.3194], std=[0.2726, 0.2528, 0.2840]),
            ])
            nw_eval_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4539, 0.3249, 0.3231], std=[0.2744, 0.2541, 0.2874]),
            ])
            self.ff_video_transform = ff_eval_transform
            self.nw_video_transform = nw_eval_transform

    def __len__(self):
        return len(self.freeform_pictures)

    def __getitem__(self, idx):
        ff_dir_name = self.freeform_pictures[idx]
        ff_pictures_path = os.path.join(self.av_path, 'Freeform', ff_dir_name)
        ff_audio_features_path = os.path.join(self.av_path, 'Freeform/audio/frame_15/', ff_dir_name + ".npy")

        nw_dir_name = self.northwind_pictures[idx]
        nw_pictures_path = os.path.join(self.av_path, 'Northwind', nw_dir_name)
        nw_audio_features_path = os.path.join(self.av_path, 'Northwind/audio/frame_15/', nw_dir_name + ".npy")

        label = self.file_to_label[ff_dir_name]
        
        person_id = ff_dir_name.split('_')[0]
        id_index = self.person_id_to_index[person_id]

        ff_video_features, ff_heatmap_stack = self.get_sorted_images(ff_pictures_path, num_frames=self.num_frames, type='Freeform')
        ff_audio_features = np.load(ff_audio_features_path).astype(np.float32) 

        nw_video_features, nw_heatmap_stack = self.get_sorted_images(nw_pictures_path, num_frames=self.num_frames, type='Northwind')
        nw_audio_features = np.load(nw_audio_features_path).astype(np.float32)

        # 在训练模式下对音频特征添加噪声进行数据增强
        if self.mode == 'train':
            ff_audio_features = self.add_feature_noise(ff_audio_features, noise_level=0.01).astype(np.float32)
            nw_audio_features = self.add_feature_noise(nw_audio_features, noise_level=0.01).astype(np.float32)

        return {
            'ff_video_features': ff_video_features,
            'ff_audio_features': ff_audio_features,
            'nw_video_features': nw_video_features,
            'nw_audio_features': nw_audio_features,
            'ff_heatmap_stack': ff_heatmap_stack,
            'nw_heatmap_stack': nw_heatmap_stack,
            'dir_name': ff_dir_name,
            'identity': torch.tensor(id_index, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
        }

    def get_sorted_images(self, directory, num_frames=50, select_frames=15, type=None):
        # 列出目录中的所有图片文件
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(files) < num_frames:
            raise ValueError(f"文件夹中的图片数量不足 {num_frames} 张，只有 {len(files)} 张。")
        
        # 按照文件名中匹配的数字进行排序
        sorted_files = sorted(files, key=lambda x: int(self.face_pattern.search(x).group(1)))[:num_frames]
        
        # 等差选择图片索引
        indices = np.linspace(0, num_frames - 1, select_frames, dtype=int)
        selected_files = [sorted_files[i] for i in indices]
        
        # 加载并转换图片
        if type == 'Freeform':
            image_tensors = [self.ff_video_transform(Image.open(os.path.join(directory, file)).convert('RGB'))
                            for file in selected_files]
        else:
            image_tensors = [self.nw_video_transform(Image.open(os.path.join(directory, file)).convert('RGB'))
                            for file in selected_files]
        
        image_stack = torch.stack(image_tensors, dim=0)

        # -------------------------------
        # 处理 heatmaps 文件夹中的数据
        # -------------------------------
        heatmap_dir = os.path.join(directory, "heatmaps")
        heatmap_files = [f for f in os.listdir(heatmap_dir) if f.lower().endswith('.npy')]
        
        if len(heatmap_files) < num_frames:
            raise ValueError(f"heatmaps 文件夹中的文件数量不足 {num_frames} 张，只有 {len(heatmap_files)} 张。")
        
        # 同样使用 face_pattern 对 heatmap 文件进行排序
        sorted_heatmap_files = sorted(heatmap_files, key=lambda x: int(self.face_pattern.search(x).group(1)))[:num_frames]
        
        # 等差选择 heatmap 文件
        selected_heatmap_files = [sorted_heatmap_files[i] for i in indices]
        
        # 加载 heatmap 数据（假设每个 .npy 文件存储的是一个数组）
        heatmap_tensors = [torch.from_numpy(np.load(os.path.join(heatmap_dir, file)))
                        for file in selected_heatmap_files]
        
        # 假设这些 heatmap 的形状一致，可以堆叠起来
        heatmap_stack = torch.stack(heatmap_tensors, dim=0)
        
        # 返回图片张量和 heatmap 张量
        return image_stack, heatmap_stack

    def add_feature_noise(self, feature, noise_level=0.01):
        """
        对音频特征添加高斯噪声进行数据增强
        """
        noise = np.random.randn(*feature.shape) * noise_level
        return feature + noise
