import numpy as np
import os
import pandas as pd
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
import dlib
import torch
from PIL import Image
import sys
sys.path.append('./')
from Depression_all.tools.utils import load_config
config = load_config('/home/b532root/account/b532zxy/workspace/Depression_all/config.yaml')

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

def generate_img(path, v_type, img_path):
    videos = get_files(path)
    for video in tqdm(videos, desc="Processing videos"):
        # 提取视频文件名的前5个字符作为名称 比如203_1
        name = video[:5]
        # 构建保存图像帧的路径，按照v_type和视频名称存放
        save_path = os.path.join(img_path, v_type, name)
        os.makedirs(save_path, exist_ok=True)

        # 打开视频文件
        cap = cv2.VideoCapture(os.path.join(path, video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 每个视频提取100张图片
        frame_indices = np.linspace(0, n_frames - 1, 100, dtype=int)
        # 创建视频处理的进度条
        loader = tqdm(frame_indices, desc=f"Processing video {name}", leave=False)

        # 遍历视频的每一帧
        for i, frame_idx in enumerate(loader):
            success = False
            attempts = 0
            while not success and attempts < 5:  # 尝试读取帧，最多尝试5次
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                attempts += 1

            if success:
                # 将帧保存为 JPEG 格式的图像文件，文件名格式为 {帧序号}.jpg
                cv2.imwrite(os.path.join(save_path, f"{i}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                loader.set_description(f"data: {path.split('/')[6]} type: {v_type} video: {name} frame: {i}")
            else:
                print(f"Failed to read frame at index {frame_idx}")
        
        cap.release()

def extract_frames(video_path, img_path, v_type, name):
    # 构建保存图像帧的路径，按照v_type和视频名称存放
    save_path = os.path.join(img_path, v_type, name)
    os.makedirs(save_path, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 每个视频提取100张图片
    frame_indices = np.linspace(0, n_frames - 1, 50, dtype=int)
    # 创建视频处理的进度条
    loader = tqdm(frame_indices, desc=f"Processing video {name}", leave=False)

    # 遍历视频的每一帧
    for i, frame_idx in enumerate(loader):
        success = False
        attempts = 0
        while not success and attempts < 5:  # 尝试读取帧，最多尝试5次
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            attempts += 1

        if success:
            # 将帧保存为 JPEG 格式的图像文件，文件名格式为 {帧序号}.jpg
            cv2.imwrite(os.path.join(save_path, f"{name}_face{i+1}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            loader.set_description(f"data: {video_path.split('/')[-1]} type: {v_type} video: {name} frame: {i}")
        else:
            print(f"Failed to read frame at index {frame_idx}")
    
    cap.release()
