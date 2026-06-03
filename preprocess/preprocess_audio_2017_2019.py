import os
import numpy as np
import pandas as pd
import librosa
import opensmile
from sklearn.preprocessing import StandardScaler

import numpy as _np
if not hasattr(_np, 'complex'):
    _np.complex = complex

INPUT_ROOT = "/home/b532root/data/b532zxy/AVEC2019_jieya"
PROCESSED_ROOT = "/home/b532root/data/b532zxy/AVEC2019"
LABEL_PATHS = {
    'train': os.path.join(PROCESSED_ROOT, "train_label.csv"),
    'dev':   os.path.join(PROCESSED_ROOT, "dev_label.csv"),
    'test':  os.path.join(PROCESSED_ROOT, "test_label.csv")
}
NUM_SEGMENTS = 15
N_MFCC = 40
   
SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

def process_audio_segment(audio, sr, start_time, end_time):
    """
    提取单段音频的 MFCC + eGeMAPS 特征，返回 128 维向量。
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = audio[start_sample:end_sample]
    min_len = int(sr * 0.5)
    if len(segment) < min_len:
        segment = np.pad(segment, (0, min_len - len(segment)), mode='constant')
    if np.max(np.abs(segment)) < 1e-5:
        return np.zeros(N_MFCC + 88, dtype=np.float32)
    # MFCC
    mfcc_feats = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc_feats, axis=1)
    # eGeMAPS
    egemaps = SMILE.process_signal(segment, sampling_rate=sr).values[0]
    return np.concatenate([mfcc_mean, egemaps]).astype(np.float32)

def extract_features(wav_path):
    """
    对单个 wav 文件提取 (15,128) 特征矩阵。
    """
    audio, sr = librosa.load(wav_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    seg_len = duration / NUM_SEGMENTS
    # 75% 重叠: 下一个段的起点在当前段开始后 0.75*seg_len
    step = seg_len * 0.75
    feats = []
    for i in range(NUM_SEGMENTS):
        start = i * step
        end = start + seg_len
        if end > duration:
            end = duration
        vec = process_audio_segment(audio, sr, start, end)
        feats.append(vec)
    return np.stack(feats, axis=0)

print("== 特征提取阶段 ==")
for split, label_csv in LABEL_PATHS.items():
    df = pd.read_csv(label_csv)
    for pid in df['Participant_ID'].astype(str):
        wav_file = os.path.join(INPUT_ROOT, f"{pid}_P", f"{pid}_AUDIO.wav")
        out_dir = os.path.join(PROCESSED_ROOT, f"{pid}_P")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{pid}_audio.npy")
        if not os.path.exists(save_path):
            if os.path.exists(wav_file):
                feats = extract_features(wav_file)
                np.save(save_path, feats)
                print(f"[{split}] {pid}: saved {feats.shape}")
            else:
                print(f"[警告] WAV 文件不存在: {wav_file}")

print("== 归一化阶段 ==")
train_df = pd.read_csv(LABEL_PATHS['train'])
all_feats = []
for pid in train_df['Participant_ID'].astype(str):
    path = os.path.join(PROCESSED_ROOT, f"{pid}_P", f"{pid}_audio.npy")
    if os.path.exists(path):
        arr = np.load(path)
        all_feats.append(arr)
if not all_feats:
    raise RuntimeError("训练集特征文件为空！")
all_stack = np.concatenate(all_feats, axis=0)  # (N_segments,128)

mfcc_scaler = StandardScaler().fit(all_stack[:, :N_MFCC])
egemaps_scaler = StandardScaler().fit(all_stack[:, N_MFCC:])

mfcc_scaler.scale_[mfcc_scaler.scale_ == 0] = 1.0
egemaps_scaler.scale_[egemaps_scaler.scale_ == 0] = 1.0

print("Scalers fitted. Applying normalization...")

for split, label_csv in LABEL_PATHS.items():
    df = pd.read_csv(label_csv)
    for pid in df['Participant_ID'].astype(str):
        file_path = os.path.join(PROCESSED_ROOT, f"{pid}_P", f"{pid}_audio.npy")
        if os.path.exists(file_path):
            arr = np.load(file_path)
            mfcc_norm  = mfcc_scaler.transform(arr[:, :N_MFCC])
            egemap_norm= egemaps_scaler.transform(arr[:, N_MFCC:])
            arr_norm   = np.concatenate([mfcc_norm, egemap_norm], axis=1).astype(np.float32)

            arr_norm = np.nan_to_num(arr_norm, nan=0.0, posinf=0.0, neginf=0.0)

            np.save(file_path, arr_norm)
            print(f"[{split}] {pid}: normalized and saved")
        else:
            print(f"[跳过] 未找到 {file_path}")
