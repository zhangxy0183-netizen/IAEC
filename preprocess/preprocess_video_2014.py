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
# root_folder = config['dataset']['path']
# all_label_path = config['dataset']['all_label_path']
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

def generate_img(path, v_type, img_path):
    # 获取指定路径下的所有视频文件名列表
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

# def get_img(if_processed):
#     if not if_processed:
#         return
#     print('get video frames....')
#     train_f = root_folder + '/train/Freeform/'
#     train_n = root_folder + '/train/Northwind/'
#     test_f = root_folder + '/test/Freeform/'
#     test_n = root_folder + '/test/Northwind/'
#     validate_f = root_folder + '/dev/Freeform/'
#     validate_n = root_folder + '/dev/Northwind/'
#     dirs = [train_f, train_n, test_f, test_n, validate_f, validate_n]
#     types = ['Freeform', 'Northwind', 'Freeform', 'Northwind', 'Freeform', 'Northwind']
#     img_path = [root_folder + '/processed/img' + '/train/', root_folder + '/processed/img' + '/train/', 
#                 root_folder + '/processed/img' + '/test/', root_folder + '/processed/img' + '/test/', 
#                 root_folder + '/processed/img' + '/validate/', root_folder + '/processed/img/' + '/validate/']
#     os.makedirs(root_folder + '/processed/img' + '/train/', exist_ok=True)
#     os.makedirs(root_folder + '/processed/img' + '/test/', exist_ok=True)
#     os.makedirs(root_folder + '/processed/img' + '/validate/', exist_ok=True)
#     for i in range(6):
#         generate_img(dirs[i], types[i], img_path[i])



def extract_frames(video_path, img_path, v_type, name):
    """
    从给定的视频文件中提取100帧并保存为图像文件。
    
    :param video_path: 视频文件的路径。
    :param img_path: 图像文件保存的路径。
    :param v_type: 视频类型，用于构建保存路径。
    :param name: 视频文件名的前5个字符，用于构建保存路径。
    """
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

if __name__ == '__main__':
    print("调用preprocess_video.py的main函数")
#     os.makedirs(root_folder + '/processed', exist_ok=True)
#     os.makedirs(root_folder + '/processed/img', exist_ok=True)
#     # os.makedirs(root_folder + '/processed/train', exist_ok=True)
#     # os.makedirs(root_folder + '/processed/test', exist_ok=True)
#     # os.makedirs(root_folder + '/processed/validate', exist_ok=True)
#     get_img(if_processed = False)
#     get_face()

# # 示例使用
# source_dir = "/home/b532root/data/b532zxy/database/AVEC2014/processed/img/validate/"
# target_dir = "/home/b532root/data/b532zxy/database/AVEC2014/processed/dev"

# # 调用函数
# extract_faces_from_images(source_dir, target_dir)
# face = extra_face("/home/b532root/data/b532zxy/database/AVEC2014/processed/img/train/Northwind/223_2/30.jpg")
# cv2.imwrite("/home/b532root/data/b532zxy/database/AVEC2014/processed/train/0.jpg",face)


# 示例调用
video_path = "/home/b532root/data/b532zxy/AVEC2014/test/Northwind/341_2_Northwind_video.mp4"
img_path = "/home/b532root/data/b532zxy/AVEC2014/test"
v_type = "Northwind"
name = "341_2"

extract_frames(video_path, img_path, v_type, name)