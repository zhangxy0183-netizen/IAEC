import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AudioVideoDataset(Dataset):
    def __init__(self, root_dir: str, num_frames: int = 50, select_frames: int = 15, 
                 audio_noise_std: float = 0.0,
                 visual_occlusion_ratio: float = 0.0,
                 visual_occlusion_mode: str = "none",
                 audio_feature_type: str = "audio_npy"):
        self.root = root_dir
        self.num_frames = num_frames
        self.select_frames = select_frames
        self.mode = os.path.basename(root_dir)
        self.audio_noise_std = audio_noise_std
        self.visual_occlusion_ratio = visual_occlusion_ratio
        self.visual_occlusion_mode = visual_occlusion_mode
        self.audio_feature_type = audio_feature_type
        label_csv = os.path.join(root_dir, f"{self.mode}_label.csv")
        df = pd.read_csv(label_csv, dtype={'file': str})
        self.label_map = dict(zip(df['file'], df['label']))

        self.sample_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d != 'audio_npy' and d != 'wav2vec2' and d != 'hubert'
        ])

        base_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        label_paths = [
            os.path.join(base_dir, 'train', 'train_label.csv'),
            os.path.join(base_dir, 'dev',   'dev_label.csv'),
            os.path.join(base_dir, 'test',  'test_label.csv'),
        ]
        persons = set()
        for p in label_paths:
            dff = pd.read_csv(p, dtype={'file': str})
            persons.update(fname.split('_')[0] for fname in dff['file'])
        persons = sorted(persons)
        self.person2idx = {pid: i for i, pid in enumerate(persons)}

        self.num_pattern = re.compile(r'face(\d+)')

        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop((256,256), scale=(0.9,1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.2,0.2,0.2,0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.4526, 0.3225, 0.3212],[0.2735, 0.2535, 0.2857]),
                transforms.RandomErasing(p=0.3, scale=(0.02,0.33), ratio=(0.3,3.3))
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize([0.4526, 0.3225, 0.3212],[0.2735, 0.2535, 0.2857])
            ])

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        dir_name = self.sample_dirs[idx]
        sample_folder = os.path.join(self.root, dir_name)

        img_list = sorted([
            f for f in os.listdir(sample_folder)
            if f.lower().endswith(('.jpg','.png','.bmp'))
        ], key=lambda x: int(self.num_pattern.search(x).group(1)))[:self.num_frames]
        indices = np.linspace(0, self.num_frames-1, self.select_frames, dtype=int)
        imgs = []
        for i in indices:
            img = self.transforms(Image.open(os.path.join(sample_folder, img_list[i])).convert('RGB'))
            if self.mode == 'test':
                img = self.apply_occlusion(img)
            imgs.append(img)
        images = torch.stack(imgs, dim=0)

        hm_folder = os.path.join(sample_folder, 'heatmaps')
        hm_list = sorted([
            f for f in os.listdir(hm_folder) if f.lower().endswith('.npy')
        ], key=lambda x: int(self.num_pattern.search(x).group(1)))[:self.num_frames]
        hms = [
            torch.from_numpy(np.load(os.path.join(hm_folder, hm_list[i])))
            for i in indices
        ]
        heatmaps = torch.stack(hms, dim=0)

        audio_path = os.path.join(self.root, self.audio_feature_type, f"{dir_name}.npy")
        audio = torch.from_numpy(np.load(audio_path).astype(np.float32))
        if self.mode == 'train':
            audio += torch.randn_like(audio) * 0.01

        if self.mode == 'test' and self.audio_noise_std > 0:
            audio = audio + torch.randn_like(audio) * self.audio_noise_std

        file_id = "_".join(dir_name.split("_")[1:])
        label = torch.tensor(float(self.label_map[file_id]), dtype=torch.float32)
        pid = file_id.split('_')[0]
        identity = torch.tensor(self.person2idx[pid], dtype=torch.long)

        return {
            'video_features': images,
            'audio_features': audio,
            'heatmap_stacks': heatmaps,
            'label': label,
            'identity': identity,
            'dir_name': file_id
        }
    
    def apply_occlusion(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        img_tensor: [C, H, W]
        """
        if self.visual_occlusion_ratio <= 0 or self.visual_occlusion_mode == "none":
            return img_tensor

        c, h, w = img_tensor.shape
        occ_area = int(h * w * self.visual_occlusion_ratio)
        occ_h = int(np.sqrt(occ_area))
        occ_w = int(np.sqrt(occ_area))

        occ_h = max(1, min(occ_h, h))
        occ_w = max(1, min(occ_w, w))

        if self.visual_occlusion_mode == "center":
            top = (h - occ_h) // 2
            left = (w - occ_w) // 2
        elif self.visual_occlusion_mode == "upper":
            top = 0
            left = (w - occ_w) // 2
        elif self.visual_occlusion_mode == "lower":
            top = h - occ_h
            left = (w - occ_w) // 2
        elif self.visual_occlusion_mode == "random":
            top = np.random.randint(0, h - occ_h + 1)
            left = np.random.randint(0, w - occ_w + 1)
        else:
            return img_tensor

        img_tensor[:, top:top+occ_h, left:left+occ_w] = 0.0
        return img_tensor
    
if __name__ == "__main__":
    modes = ['train', 'dev', 'test']
    base_path = "/home/b532root/data/b532zxy/AVEC2014"
    for m in modes:
        root = os.path.join(base_path, m)
        print(f"\nMode: {m}")
        ds = AudioVideoDataset(root_dir=root, audio_feature_type="hubert")
        print(f"  Sample count: {len(ds)}")
        sample = ds[0]
        print("  dir_name:", sample['dir_name'])
        print("  video_features:", sample['video_features'].shape)
        print("  heatmaps:", sample['heatmap_stacks'].shape)
        print("  audio_features:", sample['audio_features'].shape)
        print("  identity:", sample['identity'])
        print("  label:", sample['label'])
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        print("  Batch dir_names:", batch['dir_name'])
