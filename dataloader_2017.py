import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class AVEC2017AudioVideoDataset(Dataset):
    """
    AVEC2017 数据集：加载预处理好的 video/audio/heatmap 特征和 PHQ8_Score 标签，
    并在训练模式下对音频特征添加高斯噪声增强。
    每个样本文件夹命名为 {pid}_P，其中 pid 为参与者 ID。
    """
    def __init__(self, base_root, label_csv, mode='train', noise_level=0.05,
                 eval_audio_noise_level=0.0,    # test-time degradation
                 eval_occlusion_ratio=0.0,      # test-time visual degradation
                 eval_occlusion_mode='none'):
        """
        :param base_root: 数据根目录，包含多个子文件夹 XXX_P，每个文件夹下有：
            - {pid}_image.npy      (15,3,256,256)
            - {pid}_audio.npy      (15,128)
            - {pid}_heatmap.npy    (15,64,64)
        :param label_csv: CSV 文件路径，包含 Participant_ID, PHQ8_Score 列
        :param mode: 'train' 或 'eval'，只有 'train' 模式下才加噪声
        :param noise_level: Gaussian 噪声标准差
        """
        self.base_root = base_root
        self.mode = mode
        self.noise_level = noise_level
        self.eval_audio_noise_level = eval_audio_noise_level
        self.eval_occlusion_ratio = eval_occlusion_ratio
        self.eval_occlusion_mode = eval_occlusion_mode

        # 1. 读取当前模式标签并保留 PID 列和 Score 列
        df = pd.read_csv(label_csv)
        pid_col   = [c for c in df.columns if 'participant' in c.lower() and 'id' in c.lower()][0]
        score_col = [c for c in df.columns if 'phq' in c.lower() and 'score' in c.lower()][0]
        df[pid_col] = df[pid_col].astype(int)
        self.ids = df[pid_col].tolist()
        self.labels = df.set_index(pid_col)[score_col].to_dict()

        # 2. 构建全局 identity 映射：扫描所有 *_label.csv 文件
        label_dir = os.path.dirname(label_csv)
        all_label_csvs = [os.path.join(label_dir, f)
                          for f in os.listdir(label_dir)
                          if f.endswith('_label.csv')]
        persons = set()
        for csv_path in all_label_csvs:
            dfi = pd.read_csv(csv_path)
            # 重新检测 pid_col
            pid_col_i = [c for c in dfi.columns if 'participant' in c.lower() and 'id' in c.lower()][0]
            dfi[pid_col_i] = dfi[pid_col_i].astype(int)
            persons.update(dfi[pid_col_i].tolist())
        unique_ids = sorted(persons)
        self.person2idx = {pid: idx for idx, pid in enumerate(unique_ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        folder = os.path.join(self.base_root, f"{pid}_P")

        # 加载预存的 numpy 特征
        video_np   = np.load(os.path.join(folder, f"{pid}_image.npy"))   # (15,3,256,256)
        audio_np   = np.load(os.path.join(folder, f"{pid}_audio.npy"))   # (15,128)
        heatmap_np = np.load(os.path.join(folder, f"{pid}_heatmap.npy")) # (15,64,64)

        # 转为 Tensor
        video_t   = torch.from_numpy(video_np).float()
        audio_t   = torch.from_numpy(audio_np).float()
        heatmap_t = torch.from_numpy(heatmap_np).float()
        label_t   = torch.tensor(self.labels[pid], dtype=torch.float32)

        # 添加高斯噪声（仅 train）
        if self.mode == 'train' and self.noise_level > 0:
            noise = torch.randn_like(audio_t) * self.noise_level
            audio_t = audio_t + noise
        audio_t[torch.isnan(audio_t)] = 0.0

        # eval-time audio degradation
        if self.mode == 'test' and self.eval_audio_noise_level > 0:
            eval_noise = torch.randn_like(audio_t) * self.eval_audio_noise_level
            audio_t = audio_t + eval_noise

        # eval-time visual occlusion
        if self.mode == 'test':
            video_t = self.apply_video_occlusion(video_t)

        # 构建 identity tensor
        identity = torch.tensor(self.person2idx[pid], dtype=torch.long)

        return {
            'video_features':   video_t,
            'audio_features':   audio_t,
            'heatmap_stacks':   heatmap_t,
            'label':            label_t,
            'identity':         identity,
            'dir_name':         f"{pid}_P"
        }
    
    def apply_video_occlusion(self, video_t: torch.Tensor) -> torch.Tensor:
        """
        video_t: [T, C, H, W]
        """
        if self.eval_occlusion_ratio <= 0 or self.eval_occlusion_mode == 'none':
            return video_t

        video_t = video_t.clone()
        T, C, H, W = video_t.shape
        occ_area = max(1, int(H * W * self.eval_occlusion_ratio))
        occ_h = int(np.sqrt(occ_area))
        occ_w = int(np.sqrt(occ_area))
        occ_h = max(1, min(occ_h, H))
        occ_w = max(1, min(occ_w, W))

        for t in range(T):
            if self.eval_occlusion_mode == 'center':
                top = (H - occ_h) // 2
                left = (W - occ_w) // 2
            elif self.eval_occlusion_mode == 'upper':
                top = 0
                left = (W - occ_w) // 2
            elif self.eval_occlusion_mode == 'lower':
                top = H - occ_h
                left = (W - occ_w) // 2
            elif self.eval_occlusion_mode == 'random':
                top = np.random.randint(0, H - occ_h + 1)
                left = np.random.randint(0, W - occ_w + 1)
            else:
                return video_t

            video_t[t, :, top:top+occ_h, left:left+occ_w] = 0.0

        return video_t

if __name__ == "__main__":
    # 测试 AVEC2017AudioVideoDataset
    base_root = "/home/b532root/data/b532zxy/AVEC2019"
    label_dir = "/home/b532root/data/b532zxy/AVEC2019"

    for mode in ['train', 'dev', 'test']:
        label_csv = os.path.join(label_dir, f"{mode}_label.csv")
        print(f"\nMode: {mode}")
        ds = AVEC2017AudioVideoDataset(base_root, label_csv, mode=mode)
        print(f"  Sample count: {len(ds)}")
        sample = ds[0]
        print("  dir_name:", sample['dir_name'])
        print("  video_features.shape:", sample['video_features'].shape)
        print("  audio_features.shape:", sample['audio_features'].shape)
        print("  heatmap_stacks.shape:", sample['heatmap_stacks'].shape)
        print("  label:", sample['label'].item())
        print("  identity:", sample['identity'].item())
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        print("  Batch dir_names:", batch['dir_name'])
