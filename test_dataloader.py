import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
sys.path.append(".")
from Depression_k.dataloader import AudioVideoDataset
import torch.multiprocessing as mp
def test_dataloader():  
    # 设置音频和视频数据集路径和标签路径
    av_path = '/home/b532root/data/b532zxy/AVEC2014/face/train'
    label_path = '/home/b532root/data/b532zxy/AVEC2014/face/train/train_label.csv'

    # 初始化数据集
    dataset = AudioVideoDataset(av_path=av_path, label_path=label_path)

    # 使用 DataLoader 迭代数据
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)


    # 取一批数据进行测试
    for batch_idx, data in enumerate(dataloader):
        if data is None:
            print("Skipping due to invalid sample.")
            continue
        ff_video_features = data['ff_video_features']
        ff_audio_features = data['ff_audio_features']
        nw_video_features = data['nw_video_features']
        nw_audio_features = data['nw_audio_features']
        ff_heatmap_stack = data['ff_heatmap_stack']
        nw_heatmap_stack = data['nw_heatmap_stack']
        dir_name = data['dir_name']
        identity = data['identity']
        # ff_heatmaps = data['ff_heatmaps']
        # nw_heatmaps = data['nw_heatmaps']
        label = data['label']
        
        # print(f"Batch {batch_idx + 1}:")                             
        # # 打印 freeform 和 northwind 提取的图像和音频特征的形状
        print("ff_video_features:", ff_video_features.shape)
        print("ff_audio_features:", ff_audio_features.shape)
        print("nw_video_features", nw_video_features.shape)
        print("nw_audio_features:", nw_audio_features.shape)
        # 打印热图堆栈的形状
        print("ff_heatmap_stack:", ff_heatmap_stack.shape)
        print("nw_heatmap_stack:", nw_heatmap_stack.shape)
        # 打印 dir_name 和 identity 的形状
        print("dir_name:", dir_name)
        print("identity:", identity)
        # # 打印标签
        print("label:", label.unsqueeze(1).shape)
        # break  # 只测试一个batch

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()
    test_dataloader()
