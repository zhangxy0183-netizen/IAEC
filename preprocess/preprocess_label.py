import numpy as np
import os
import pandas as pd
import cv2
from tqdm import tqdm
import torch
import sys
sys.path.append('./')
from Depression_k.tools.utils import load_config
from Depression_k.trash.extra_feats import splitface2eye_nose_mouth
config = load_config('/home/b532root/account/b532zxy/workspace/Depression_k/config.yaml')
root_folder = config['dataset']['path']
lable_path = config['dataset']['lable_path']
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

# 把所有lable的csv文件合并到一个csv中，格式:
# file   label
# 203_1   8
def generate_label_file():
    print('get label....')
    base_url = lable_path
    file_list = get_files(base_url)  # 假设 get_files 函数已经定义好
    labels = []
    loader = tqdm(file_list)
    for file in loader:
        label = pd.read_csv(os.path.join(base_url, file), header=None)
        labels.append([file[:file.find('_Depression_k.csv')], label.iloc[0, 0]])  # 使用 iloc 读取标签值
        loader.set_description('file:{}'.format(file))
    
    # 提取文件名和标签
    file_names = [item[0] for item in labels]
    label_values = [item[1] for item in labels]
    
    # 将标签转换为 PyTorch 的 Tensor
    labels_tensor = torch.tensor(label_values, dtype=torch.float32)
    
    # 将文件名和标签转换为 DataFrame
    df = pd.DataFrame({'file': file_names, 'label': labels_tensor.numpy()})
    
    # 保存为 CSV 文件
    df.to_csv(root_folder + '/processed/label.csv', index=False)
    
    return labels


if __name__ == '__main__':
    # 12345实现从总label中提取train、test、dev的标签 改第一步的三个文件夹就行
    # 1. 设置文件夹路径和标签文件路径
    mp4_folder_path = "/home/b532root/data/b532zxy/database/AVEC15/dev/Northwind/"  # MP4文件所在文件夹
    label_file_path = "/home/b532root/data/b532zxy/database/AVEC15/label.csv"   # label.csv文件路径
    output_file_path = "/home/b532root/data/b532zxy/database/AVEC15/dev/dev_label.csv"  # 输出结果的csv文件路径
    # 2. 获取所有MP4文件的前五个字符
    mp4_files = [f[:5] for f in os.listdir(mp4_folder_path) if f.endswith('.mp4')]
    # 3. 读取label.csv文件
    label_df = pd.read_csv(label_file_path)
    # label.csv的第一列是文件名
    # 4. 根据第一列进行匹配
    matched_data = label_df[label_df['file'].isin(mp4_files)]  # 'file'是第一列列名
    # 5. 保存匹配的行到一个新的CSV文件
    matched_data.to_csv(output_file_path, index=False)
    print(f"匹配结果已保存到 {output_file_path}")

    