import numpy as np
import os
import torch
import librosa
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import opensmile
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
from Depression_k.tools.utils import load_config
# config = load_config('/home/b532root/account/b532zxy/workspace/Dmine_Kfold/config.yaml')
# root_folder = config['dataset']['path']
# label_path = config['dataset']['all_label_path']
# processed_path = config['dataset']['processed_path']

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
# 从视频中提取音频的函数
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
# 利用emotion2vec提取音频的句子级别特征（1024维）
def extract_emotion2vec_plus_large(directory_path, batch_size=16):
    inference_pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model="iic/emotion2vec_plus_large",
            device = 'cuda:1' if torch.cuda.is_available() else 'cpu')
    files = get_files(directory_path)
    failed_files = []  # 保存处理失败的文件
    # 分批处理文件
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        
        # 处理当前批次的文件
        for file in batch_files:
            if file.endswith(".wav"):
                # 构造完整的文件路径
                full_path = os.path.join(directory_path, file)
                
                try:
                    # 调用推理管道
                    inference_pipeline(
                        full_path,
                        output_dir=os.path.join(directory_path, "audio_npys"),
                        granularity="utterance",
                        extract_embedding=True
                    )
                except Exception as e:
                    # 捕获异常并保存失败的文件名
                    print(f"处理文件 {full_path} 时出现错误: {e}")
                    failed_files.append(full_path)
        
        # 清理内存
        torch.cuda.empty_cache()

    # 输出处理失败的文件
    if failed_files:
        print("以下文件处理失败：")
        for failed_file in failed_files:
            print(failed_file+"正在重新处理 ...")
            inference_pipeline = pipeline(
                task=Tasks.emotion_recognition,
                model="iic/emotion2vec_plus_large",
                device = 'cpu')
            rec_result = inference_pipeline(failed_file, 
                                            output_dir=os.path.join(directory_path, "audio_npys"), granularity="utterance",
                                            extract_embedding=True, disable_update=True)
            print(failed_file+"处理完成")
    else:
        print("所有文件都处理成功。")

#!/usr/bin/env python3
import os
import glob
import numpy as np
import librosa
import opensmile
import argparse
from sklearn.preprocessing import StandardScaler

#######################
# 1. 特征提取部分
#######################
def process_audio_segment(audio, sr, start_time, end_time, n_mfcc=40):
    """
    提取音频片段，并提取 MFCC（40 维）与 eGeMAPS（88 维）特征，
    最后直接拼接成 128 维向量返回。
    """
    # 提取音频片段
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = audio[start_sample:end_sample]

    # 如果片段太短（不足 0.5 秒），进行零填充
    min_length = int(sr * 0.5)
    if len(segment) < min_length:
        segment = np.pad(segment, (0, min_length - len(segment)), mode='constant')

    # 若全为静音，则返回全零向量
    if np.max(np.abs(segment)) < 1e-5:
        return np.zeros(n_mfcc + 88)

    # 提取 MFCC 特征，取均值后 shape: (n_mfcc,)
    mfcc_features = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc_features, axis=1)

    # 提取 eGeMAPS 特征（opensmile 返回的是 DataFrame，取第一行的值）
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    egemaps_features = smile.process_signal(segment, sampling_rate=sr).values[0]

    # 拼接两个部分
    combined_features = np.concatenate((mfcc_mean, egemaps_features))
    return combined_features

def extract_features_and_save(audio_path, num_segments=15, n_mfcc=40):
    """
    对单个音频文件提取特征，并将 (num_segments,128) 的矩阵保存为 npy 文件，
    存放在音频文件所在目录下的 "frame_15" 文件夹中。
    """
    base_dir = os.path.dirname(audio_path)
    audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
    save_dir = os.path.join(base_dir, "frame_15")
    os.makedirs(save_dir, exist_ok=True)
    feature_save_path = os.path.join(save_dir, f"{audio_filename}.npy")
    
    # 加载音频文件
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

#######################
# 2. 归一化部分
#######################
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

#######################
# 3. 主函数：选择提取或归一化
#######################
def main(args):
    
    # 配置各数据集目录（请根据你的实际情况调整路径）
    train_freeform_audio = "/home/b532root/data/b532zxy/AVEC15/train/Freeform/audio"
    train_northwind_audio = "/home/b532root/data/b532zxy/AVEC15/train/Northwind/audio"
    dev_freeform_audio = "/home/b532root/data/b532zxy/AVEC15/dev/Freeform/audio"
    dev_northwind_audio = "/home/b532root/data/b532zxy/AVEC15/dev/Northwind/audio"
    test_freeform_audio = "/home/b532root/data/b532zxy/AVEC15/test/Freeform/audio"
    test_northwind_audio = "/home/b532root/data/b532zxy/AVEC15/test/Northwind/audio"
    
    if args.mode == "extract":
        # 对所有数据集进行特征提取
        
        directories = [train_freeform_audio, train_northwind_audio,
                       dev_freeform_audio, dev_northwind_audio,
                       test_freeform_audio, test_northwind_audio]
        for directory in directories:
            for filename in os.listdir(directory):
                if filename.endswith(".wav"):
                    filepath = os.path.join(directory, filename)
                    extract_features_and_save(filepath, num_segments=args.num_segments, n_mfcc=args.n_mfcc)
                    
    elif args.mode == "normalize":
        # 归一化阶段：只使用训练集的 npy 数据计算 scaler
        train_freeform_npy = os.path.join(train_freeform_audio, "frame_15")
        train_northwind_npy = os.path.join(train_northwind_audio, "frame_15")
        training_dirs = [train_freeform_npy, train_northwind_npy]
        mfcc_scaler, egemaps_scaler = compute_scalers(training_dirs, n_mfcc=args.n_mfcc)
        
        # 对各数据集的 npy 文件归一化
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



