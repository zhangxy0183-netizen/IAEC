import numpy as np
import os
import torch
import librosa
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import opensmile
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
from Depression_all.tools.utils import load_config
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

from transformers import Wav2Vec2Processor, Wav2Vec2Model
local_model_path = "/home/b532root/account/b532zxy/workspace/Depression_all/wav2vec2-base-960h"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained(
    local_model_path,
    local_files_only=True
)
wav2vec2_model = Wav2Vec2Model.from_pretrained(
    local_model_path,
    local_files_only=True
)
wav2vec2_model.to(device)
wav2vec2_model.eval()

def extract_wav2vec2(audio, sr, start_time, end_time):
    # 提取音频片段
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = audio[start_sample:end_sample]

    if len(segment) == 0:
        segment = np.zeros(int(sr * 1.0), dtype=np.float32)
    segment = segment.astype(np.float32)
    target_sr = 16000
    if sr != target_sr:
        segment = librosa.resample(segment, orig_sr=sr, target_sr=target_sr)
    segment = segment.astype(np.float32)
    inputs = processor(
        segment,
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        outputs = wav2vec2_model(input_values)
        hidden_states = outputs.last_hidden_state
        pooled_feature = hidden_states.mean(dim=1)

    segment_feature = pooled_feature.squeeze(0).cpu().numpy()
    return segment_feature

def process_audio_segment(audio_path, num_segments=15):
    if "base_train" in audio_path or "/train/" in audio_path:
        split_name = "train"
    elif "base_dev" in audio_path or "/dev/" in audio_path:
        split_name = "dev"
    elif "base_test" in audio_path or "/test/" in audio_path:
        split_name = "test"
    else:
        raise ValueError(f"无法从路径中判断 train/dev/test: {audio_path}")

    if "Freeform" in audio_path:
        task_name = "Freeform"
    elif "Northwind" in audio_path:
        task_name = "Northwind"
    else:
        raise ValueError(f"无法从路径中判断 Freeform/Northwind: {audio_path}")

    save_dir = os.path.join(
        "/home/b532root/data/b532zxy/AVEC2014",
        split_name,
        "wav2vec2"
    )
    os.makedirs(save_dir, exist_ok=True)

    audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
    parts = audio_filename.split("_")
    sample_id = "_".join(parts[:2])
    save_filename = f"{task_name}_{sample_id}.npy"
    feature_save_path = os.path.join(save_dir, save_filename)
    
    audio, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    
    segment_length = duration / num_segments
    step_size = segment_length / 4

    features = []
    for i in range(num_segments):
        start_time = i * step_size
        end_time = start_time + segment_length
        if end_time > duration:
            end_time = duration
        seg_feat = extract_wav2vec2(audio, sr, start_time, end_time)
        features.append(seg_feat)
    feature_matrix = np.array(features)  # (num_segments, 128)
    np.save(feature_save_path, feature_matrix)
    print(f"Features saved to {feature_save_path}")
    return feature_matrix

def main(args):
    
    train_freeform_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_train/Freeform/"
    train_northwind_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_train/Northwind/"
    dev_freeform_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_dev/Freeform/"
    dev_northwind_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_dev/Northwind/"
    test_freeform_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_test/Freeform/"
    test_northwind_audio = "/home/b532root/data/b532zxy/AVEC2014_base/base_test/Northwind/"
    
    directories = [train_freeform_audio, train_northwind_audio,
                   dev_freeform_audio, dev_northwind_audio,
                   test_freeform_audio, test_northwind_audio]
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                filepath = os.path.join(directory, filename)
                process_audio_segment(filepath, num_segments=args.num_segments)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Feature Extraction and Normalization")
    parser.add_argument("--num_segments", type=int, default=15, help="分段数")
    args = parser.parse_args()
    main(args)



