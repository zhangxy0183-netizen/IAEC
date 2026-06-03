import numpy as np
import os
import librosa
import opensmile
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
#!/usr/bin/env python3
import os
import glob
import numpy as np
import librosa
import opensmile
import argparse
from sklearn.preprocessing import StandardScaler

def get_files(path):
    file_info = os.walk(path)
    file_list = []
    for r, d, f in file_info:
        file_list += f
    return file_list

def get_dirs(path):
    file_info = os.walk(path)
    dirs = []
    for d, r, f in file_info:
        dirs.append(d)
    return dirs[1:]

def convert_video_to_wav(root_dir):
    # 遍历所有子目录
    for subdir, dirs, files in os.walk(root_dir):
        # 遍历当前目录下的所有文件
        for file in files:
            # 检查文件是否为视频文件
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(subdir, file)
                # 提取视频文件名（不含扩展名）
                video_name = os.path.splitext(file)[0]
                # 设置音频输出路径
                audio_output_path = os.path.join(subdir, f"{video_name}.wav")
                
                # 调用函数提取音频
                convert_video_to_wav(video_path, audio_output_path)
                print(f"Extracted audio from {video_path} to {audio_output_path}")

def process_audio_segment(audio, sr, start_time, end_time, n_mfcc=40):
    # 提取音频片段
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = audio[start_sample:end_sample]

    min_length = int(sr * 0.5)
    if len(segment) < min_length:
        segment = np.pad(segment, (0, min_length - len(segment)), mode='constant')

    if np.max(np.abs(segment)) < 1e-5:
        return np.zeros(n_mfcc + 88)

    mfcc_features = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc_features, axis=1)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    egemaps_features = smile.process_signal(segment, sampling_rate=sr).values[0]
    combined_features = np.concatenate((mfcc_mean, egemaps_features))
    return combined_features

def extract_features_and_save(audio_path, num_segments=15, n_mfcc=40):
    base_dir = os.path.dirname(audio_path)
    audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
    save_dir = os.path.join(base_dir, "frame_15")
    os.makedirs(save_dir, exist_ok=True)
    feature_save_path = os.path.join(save_dir, f"{audio_filename}.npy")
    
    audio, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    
    segment_length = duration / num_segments
    step_size = segment_length / 4  # 25% 重叠

    features = []
    for i in range(num_segments):
        start_time = i * step_size
        end_time = start_time + segment_length
        if end_time > duration:
            end_time = duration
        seg_feat = process_audio_segment(audio, sr, start_time, end_time, n_mfcc=n_mfcc)
        features.append(seg_feat)
    feature_matrix = np.array(features)  # (num_segments, 128)
    np.save(feature_save_path, feature_matrix)
    print(f"Features saved to {feature_save_path}")
    return feature_matrix

def load_all_features(feature_dir, file_pattern="*.npy"):
    """
    遍历指定目录下所有 npy 文件，将每个文件的 (num_segments, 128) 数据拼接成 (total_segments, 128)。
    """
    feature_files = glob.glob(os.path.join(feature_dir, file_pattern))
    all_features = []
    for file in feature_files:
        features = np.load(file)
        all_features.append(features)
    if all_features:
        return np.concatenate(all_features, axis=0)
    else:
        return None

def compute_scalers(training_dirs, n_mfcc=40):
    """
    使用训练集目录列表（例如 Freeform 和 Northwind 的训练集 npy 文件目录）来计算全局统计量，
    分别针对 MFCC（前 n_mfcc 维）和 eGeMAPS（后 88 维）返回两个 StandardScaler 对象。
    """
    all_features = []
    for d in training_dirs:
        feats = load_all_features(d)
        if feats is not None:
            all_features.append(feats)
    if not all_features:
        raise ValueError("没有找到训练集的 npy 文件！")
    all_features = np.concatenate(all_features, axis=0)
    
    mfcc_features = all_features[:, :n_mfcc]
    egemaps_features = all_features[:, n_mfcc:]
    
    mfcc_scaler = StandardScaler().fit(mfcc_features)
    egemaps_scaler = StandardScaler().fit(egemaps_features)
    print("训练集全局 MFCC 均值：", mfcc_scaler.mean_[:10])
    print("训练集全局 MFCC Std：", np.sqrt(mfcc_scaler.var_)[:10])
    print("训练集全局 eGeMAPS 均值：", egemaps_scaler.mean_[:10])
    print("训练集全局 eGeMAPS Std：", np.sqrt(egemaps_scaler.var_)[:10])
    return mfcc_scaler, egemaps_scaler

def normalize_features(features, mfcc_scaler, egemaps_scaler, n_mfcc=40):
    """
    使用训练集得到的 scaler 对 features 进行标准化，并返回归一化后的结果。
    """
    mfcc_part = features[:, :n_mfcc]
    egemaps_part = features[:, n_mfcc:]
    mfcc_norm = mfcc_scaler.transform(mfcc_part)
    egemaps_norm = egemaps_scaler.transform(egemaps_part)
    return np.concatenate([mfcc_norm, egemaps_norm], axis=1)

def transform_and_save_features(feature_dir, mfcc_scaler, egemaps_scaler, n_mfcc=40):
    """
    遍历 feature_dir 下的所有 npy 文件，加载后使用传入的 scaler 归一化，然后覆盖保存。
    """
    feature_files = glob.glob(os.path.join(feature_dir, "*.npy"))
    for file in feature_files:
        features = np.load(file)
        features_norm = normalize_features(features, mfcc_scaler, egemaps_scaler, n_mfcc=n_mfcc)
        np.save(file, features_norm)
        print(f"Normalized features saved to: {file}")

def main(args):
    
    train_freeform_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_train/Freeform/audio"
    train_northwind_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_train/Northwind/audio"
    dev_freeform_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_dev/Freeform/audio"
    dev_northwind_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_dev/Northwind/audio"
    test_freeform_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_test/Freeform/audio"
    test_northwind_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_test/Northwind/audio"
    
    if args.mode == "extract":        
        directories = [train_freeform_audio, train_northwind_audio,
                       dev_freeform_audio, dev_northwind_audio,
                       test_freeform_audio, test_northwind_audio]
        for directory in directories:
            for filename in os.listdir(directory):
                if filename.endswith(".wav"):
                    filepath = os.path.join(directory, filename)
                    extract_features_and_save(filepath, num_segments=args.num_segments, n_mfcc=args.n_mfcc)
                    
    elif args.mode == "normalize":
        train_freeform_npy = os.path.join(train_freeform_audio, "frame_15")
        train_northwind_npy = os.path.join(train_northwind_audio, "frame_15")
        training_dirs = [train_freeform_npy, train_northwind_npy]
        mfcc_scaler, egemaps_scaler = compute_scalers(training_dirs, n_mfcc=args.n_mfcc)
        
        dataset_dirs = {
            "train_Freeform": os.path.join(train_freeform_audio, "frame_15"),
            "train_Northwind": os.path.join(train_northwind_audio, "frame_15"),
            "dev_Freeform": os.path.join(dev_freeform_audio, "frame_15"),
            "dev_Northwind": os.path.join(dev_northwind_audio, "frame_15"),
            "test_Freeform": os.path.join(test_freeform_audio, "frame_15"),
            "test_Northwind": os.path.join(test_northwind_audio, "frame_15")
        }
        for name, feature_dir in dataset_dirs.items():
            if os.path.exists(feature_dir):
                transform_and_save_features(feature_dir, mfcc_scaler, egemaps_scaler, n_mfcc=args.n_mfcc)
            else:
                print(f"目录 {feature_dir} 不存在，跳过。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Feature Extraction and Normalization")
    parser.add_argument("--mode", type=str, choices=["extract", "normalize"],
                        help="运行模式：extract（提取特征）或 normalize（归一化特征）")
    parser.add_argument("--num_segments", type=int, default=15, help="分段数")
    parser.add_argument("--n_mfcc", type=int, default=40, help="MFCC 维度")
    args = parser.parse_args()
    args.mode = "normalize"
    main(args)



