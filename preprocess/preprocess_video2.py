import sys
sys.path.append('./')
import numpy as np
import os
import pandas as pd
import cv2
from tqdm import tqdm
import torch
from Depression_all.tools.utils import load_config
config = load_config('/home/b532root/account/b532zxy/workspace/Depression_all/config.yaml')
root_folder = config['dataset']['path']
label_path = config['dataset']['all_label_path']
processed_path = config['dataset']['processed_path']
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
        name = video[:5]
        save_path = os.path.join(img_path, v_type, name)
        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(os.path.join(path, video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, n_frames - 1, 100, dtype=int)
        loader = tqdm(frame_indices, desc=f"Processing video {name}", leave=False)

        for i, frame_idx in enumerate(loader):
            success = False
            attempts = 0
            while not success and attempts < 5:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                attempts += 1

            if success:
                cv2.imwrite(os.path.join(save_path, f"{i}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                loader.set_description(f"data: {path.split('/')[6]} type: {v_type} video: {name} frame: {i}")
            else:
                print(f"Failed to read frame at index {frame_idx}")
        
        cap.release()

def get_img(if_processed):
    if not if_processed:
        return
    print('get video frames....')
    train_f = root_folder + '/train/Freeform/'
    train_n = root_folder + '/train/Northwind/'
    test_f = root_folder + '/test/Freeform/'
    test_n = root_folder + '/test/Northwind/'
    validate_f = root_folder + '/dev/Freeform/'
    validate_n = root_folder + '/dev/Northwind/'
    dirs = [train_f, train_n, test_f, test_n, validate_f, validate_n]
    types = ['Freeform', 'Northwind', 'Freeform', 'Northwind', 'Freeform', 'Northwind']
    img_path = [root_folder + '/processed/img' + '/train/', root_folder + '/processed/img' + '/train/', 
                root_folder + '/processed/img' + '/test/', root_folder + '/processed/img' + '/test/', 
                root_folder + '/processed/img' + '/validate/', root_folder + '/processed/img/' + '/validate/']
    os.makedirs(root_folder + '/processed/img' + '/train/', exist_ok=True)
    os.makedirs(root_folder + '/processed/img' + '/test/', exist_ok=True)
    os.makedirs(root_folder + '/processed/img' + '/validate/', exist_ok=True)
    for i in range(6):
        generate_img(dirs[i], types[i], img_path[i])

def get_face():
    print('get frame faces....')
    save_path = [processed_path + '/train/Freeform/', processed_path + '/train/Northwind/', processed_path + '/test/Freeform/',
                 processed_path + '/test/Northwind/', processed_path + '/validate/Freeform/', processed_path + '/validate/Northwind/']
    paths = [processed_path + '/img/train/Freeform/', processed_path + '/img/train/Northwind/', 
             processed_path + '/img/test/Freeform/', processed_path + '/img/test/Northwind/',
             processed_path + '/img/validate/Freeform/', processed_path + '/img/validate/Northwind/']
    
    for index, path in enumerate(paths):
        dirs = get_dirs(path)
        for d in dirs:
            os.makedirs(save_path[index] + d.split('/')[-1], exist_ok=True)
            files = get_files(d)
            loader = tqdm(files)
            for file in loader:
                img_path = d + '/' + file
                s_path = save_path[index] + d.split('/')[-1] + '/' + file[:-4]
                os.makedirs(s_path, exist_ok=True)
                loader.set_description_str(f"成功读取图像: {img_path}")
                loader.refresh()


if __name__ == '__main__':
    os.makedirs(root_folder + '/processed', exist_ok=True)
    os.makedirs(root_folder + '/processed/img', exist_ok=True)
    get_img(if_processed = False)
    get_face()
    