import numpy as np
import torch
import torch.nn as nn
import yaml
import os
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

from torch.utils.data import DataLoader
from validate import validate_cam, validate_fea
from Depression_k.dataloader import AudioVideoDataset
from Depression_k.model.main_model import DModel
from Depression_k.model.CAM import CAM
config = load_config('/home/b532root/account/b532zxy/workspace/Depression_k/config.yaml')

def run_test(args):
    print("----------------------------------------------begin_test--------------------------------------------------")
    args.test_data_path = config['dataset']['test_data_path']
    args.test_label_path = config['dataset']['test_label_path']
    args.best_emonet_path = config[args.test_mode]['best_emonet_path']
    args.best_feature_path = config[args.test_mode]['best_feature_path']
    args.best_cam_path = config[args.test_mode]['best_cam_path']

    os.makedirs(args.log_dir, exist_ok=True)

    # 加载模型
    feature_model = DModel(args).cuda(args.device)
    checkpoint_file = args.best_feature_path

    if args.mode == "cam":
        print("执行cam————test", end="\t")
        cam_model = CAM(args).cuda(args.device)
        checkpoint_file = args.best_cam_path
    else:
        print("执行fea————test", end="\t")

    if os.path.exists(checkpoint_file): 
        print(f"检测到 best_{args.mode} 模型......")
        checkpoint = torch.load(checkpoint_file)
        if args.mode == "fea":
            feature_model.load_state_dict(checkpoint['model_state_dict'])
        elif args.mode == "cam":
            feature_model.load_state_dict(checkpoint['feature_model_state_dict'])
            cam_model.load_state_dict(checkpoint['CAM_model_state_dict'])
            cam_model.eval()
        feature_model.eval()

    dataset_test = AudioVideoDataset(
        av_path=args.test_data_path,
        label_path=args.test_label_path,
        num_frames=args.frame_num,
        mode='test'
    )
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    criterion_regression = nn.HuberLoss(delta=7.0)

    if args.mode == "fea":
        test_rmse, test_mae, test_loss, test_epoch_info = validate_fea(
            args, feature_model, test_loader, args.device, criterion_regression, mode='test')
    else:
        test_rmse, test_mae, test_loss, test_epoch_info = validate_cam(
            args, feature_model, cam_model, test_loader, args.device, criterion_regression)

    save_epoch_info(test_epoch_info, args.log_dir, phase=f"{args.test_mode}_{args.mode}", print_if = False)

    print(f"Test RMSE: {test_rmse:.4f}", end="\t")
    print(f"Test MAE: {test_mae:.4f}", end="\t")
    print(f"Test Loss: {test_loss:.4f}")
    print("----------------------------------------------end_test--------------------------------------------------")

class AdaptiveWingLoss(nn.Module):
    def __init__(self, alpha=1.5, omega=7, epsilon=0.1, theta=0.05):
        super(AdaptiveWingLoss, self).__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def forward(self, pred, target):
        # print(pred.shape)
        # print(target.shape)
        # 计算误差
        delta = torch.abs(target - pred)
        
        # 根据误差与 theta 的比较来选择合适的公式
        loss = torch.where(delta < self.theta,
                           self.omega * torch.log(1 + (delta / self.epsilon) ** self.alpha),
                           self.omega * (delta - self.theta))
        return torch.mean(loss)

def total_losses(args, epoch, id_logits, target_id, mask, heatmaps_ground_truth, regression_output, labels):
    loss_mask = AdaptiveWingLoss()(mask, heatmaps_ground_truth)
    loss_id = nn.CrossEntropyLoss()(id_logits, target_id)
    loss_regression = nn.HuberLoss(delta=7.0)(regression_output, labels)
    # print(loss_mask, 0.02 * loss_id, 0.002 * loss_regression)
    # 注意：此处的loss_id在反向传播时由于GRL作用，对特征提取器起反向效果
    if args.w_o_ID == 1:
        loss_id = 0 

    return loss_mask + 0.1 * loss_id + 0.01 * loss_regression



def save_epoch_info(sample_info, log_dir, phase, epoch=None, print_if = True):
    """
    保存每个样本的信息到 CSV 文件。
    
    Args:
        sample_info (list of dict): 每个样本的信息，包含 dir_name、predicted、label、lr。
        log_dir (str): 保存日志的目录。
        phase (str): 阶段（'train' 或 'val'）。
        epoch (int, optional): 当前训练轮次。如果为 None，则生成不包含 epoch 信息的文件名。
    """
    os.makedirs(log_dir, exist_ok=True)
    if epoch is None:
        file_path = os.path.join(log_dir, f'{phase}_info.csv')
    else:
        file_path = os.path.join(log_dir, f'{phase}_epoch_{epoch+1}_info.csv')
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['dir_name', 'predicted', 'label', 'lr'])
        writer.writeheader()
        writer.writerows(sample_info)
    if print_if is True:
        print(f"保存 {phase} 的样本信息到 {file_path}")


def apply_mask_and_save(images, masks, save_path="./visualized_images/"):
    """
    使用 `mask` 增强原始图像，并将其输入模型，同时保存中间结果。

    Args:
        images (torch.Tensor): 原始输入图像，形状为 [batch_size, num_frames, 3, H, W]。
        masks (torch.Tensor): 生成的 `mask`，形状为 [batch_size, num_frames, 1, h, w]。
        save_path (str): 保存可视化结果的路径。
    """
    batch_size, num_frames, C, H, W = images.shape
    _, _, _, h, w = masks.shape

    # 调整 mask 大小
    masks_resized = F.interpolate(masks.view(-1, 1, h, w), size=(H, W), mode="bilinear", align_corners=False)
    masks_resized = masks_resized.view(batch_size, num_frames, 1, H, W)

    enhanced_images = []
    for b in range(batch_size):
        for t in range(num_frames):
            image = images[b, t].cpu().numpy().transpose(1, 2, 0)
            mask = masks_resized[b, t, 0].cpu().numpy()
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)  # Normalize mask
            enhanced_image = image * mask[..., None]
            enhanced_images.append(enhanced_image)

    return torch.tensor(np.stack(enhanced_images).transpose(0, 3, 1, 2)).float()

def save_enhanced_images(images, enhanced_images, save_path="./enhanced_images/"):
    """
    保存原始图像和增强后的图像。

    Args:
        images (torch.Tensor): 原始图像，形状为 [batch_size, num_frames, 3, H, W]。
        enhanced_images (torch.Tensor): 增强后的图像，形状为 [batch_size, num_frames, 3, H, W]。
        save_path (str): 保存路径。
    """
    os.makedirs(save_path, exist_ok=True)

    batch_size, num_frames, _, _, _ = images.shape
    images = images.permute(0, 1, 3, 4, 2).cpu().numpy()  # 转为 [batch, frames, H, W, C]
    enhanced_images = enhanced_images.permute(0, 1, 3, 4, 2).cpu().numpy()

    for b in range(batch_size):
        for t in range(num_frames):
            original_path = os.path.join(save_path, f"batch_{b}_frame_{t}_original.png")
            enhanced_path = os.path.join(save_path, f"batch_{b}_frame_{t}_enhanced.png")
            plt.imsave(original_path, np.clip(images[b, t], 0, 1))
            plt.imsave(enhanced_path, np.clip(enhanced_images[b, t], 0, 1))

def save_heatmap_overlay(image_tensor, heatmap_tensor, save_path, epoch):
    """
    将预测的热力图叠加到原始图像上并保存。
    :param image_tensor: 输入的图像 Tensor, shape: (B, num_frames, C, H, W)
    :param heatmap_tensor: 预测的热力图 Tensor, shape: (B, num_frames, H_heatmap, W_heatmap)
    :param save_path: 保存路径
    :param epoch: 当前训练的轮次
    """
    os.makedirs(save_path, exist_ok=True)
    batch_size, num_frames, _, img_h, img_w = image_tensor.size()
    
    for i in range(batch_size):
        for j in range(num_frames):
            image = image_tensor[i, j].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            
            # 获取单通道的热力图
            heatmap = heatmap_tensor[i, j].unsqueeze(0).unsqueeze(0)  # (1, 1, H_heatmap, W_heatmap)
            upsampled_heatmap = F.interpolate(heatmap, size=(img_h, img_w), mode='bilinear', align_corners=False).squeeze()  # (H, W)
            
            # 转换为 numpy 以用于可视化
            upsampled_heatmap_np = upsampled_heatmap.detach().cpu().numpy()  # 使用 detach() 分离计算图
            
            # 绘制并保存叠加图像
            plt.figure(figsize=(5, 5))
            plt.imshow(image)
            plt.imshow(upsampled_heatmap_np, cmap='jet', alpha=0.5)  # 将热力图叠加在图像上
            plt.colorbar()
            plt.axis('off')
            
            # 保存叠加图像
            plt.savefig(os.path.join(save_path, f"epoch_{epoch}_batch_{i}_frame_{j}.png"))
            plt.close()

def save_true_heatmap_overlay(image_tensor, true_heatmap_tensor, save_path, epoch):
    """
    将真实的热力图叠加到原始图像上并保存。
    :param image_tensor: 输入的图像 Tensor, shape: (B, num_frames, C, H, W)
    :param true_heatmap_tensor: 真实的热力图 Tensor, shape: (B, num_frames, H_heatmap, W_heatmap)
    :param save_path: 保存路径
    :param epoch: 当前训练的轮次
    """
    os.makedirs(save_path, exist_ok=True)
    batch_size, num_frames, _, img_h, img_w = image_tensor.size()
    
    for i in range(batch_size):
        for j in range(num_frames):
            image = image_tensor[i, j].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            
            # 获取单通道的真实热力图
            heatmap = true_heatmap_tensor[i, j].unsqueeze(0).unsqueeze(0)  # (1, 1, H_heatmap, W_heatmap)
            upsampled_heatmap = F.interpolate(heatmap, size=(img_h, img_w), mode='bilinear', align_corners=False).squeeze()  # (H, W)
            
            # 转换为 numpy 以用于可视化
            upsampled_heatmap_np = upsampled_heatmap.detach().cpu().numpy()  # 使用 detach() 分离计算图
            
            # 绘制并保存叠加图像
            plt.figure(figsize=(5, 5))
            plt.imshow(image)
            plt.imshow(upsampled_heatmap_np, cmap='jet', alpha=0.5)  # 将热力图叠加在图像上
            plt.colorbar()
            plt.axis('off')
            
            # 保存叠加图像，文件名包含 "true_" 前缀以区分
            plt.savefig(os.path.join(save_path, f"epoch_{epoch}_batch_{i}_frame_{j}_true.png"))
            plt.close()
from skimage import transform

def save_overlay_images(images, heatmaps, labels, predictions, save_dir, epoch, mode="predicted"):
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(min(len(images), 10)):  # 只保存前 10 张
        image = images[idx].permute(1, 2, 0).cpu().detach().numpy()  # (256, 256, 3)
        image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]

        heatmap = heatmaps[idx].cpu().detach().numpy()  # (64, 64)
        heatmap_resized = transform.resize(heatmap, (256, 256), anti_aliasing=True)  # 缩放到 (256, 256)

        label = labels[idx].item()  # 获取对应的标签
        if predictions is not None:
            prediction = predictions[idx].item()  # 获取模型的预测标签

        plt.figure(figsize=(6, 6))
        plt.imshow(image)  # 原图
        plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)  # 叠加热力图
        plt.axis("off")
        # 在图片上添加标签和预测值
        # plt.text(10, 20, f"Label: {label}", color="white", fontsize=12, bbox=dict(facecolor="black", alpha=0.5))
        # 需要显示预测值，打开下面代码
        # plt.text(10, 40, f"Prediction: {prediction}", color="white", fontsize=12, bbox=dict(facecolor="black", alpha=0.5))

        plt.savefig(os.path.join(save_dir, f"{mode}_epoch_{epoch}_idx_{idx}.png"))
        plt.close()




# 检查GPU是否可用
def check_gpu():
    if torch.cuda.is_available():
        print("GPU可用！")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")
        print("cuDNN Version:", torch.backends.cudnn.version())  # 检查 cuDNN 版本
        print("cuDNN Enabled:", torch.backends.cudnn.enabled)    # 检查 cuDNN 是否启用
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU0: {torch.cuda.get_device_name(0)}")
        print(f"当前GPU1: {torch.cuda.get_device_name(1)}")
        print(f"当前GPU2: {torch.cuda.get_device_name(2)}")
        print(f"当前GPU3: {torch.cuda.get_device_name(3)}")
    else:
        print("GPU不可用，将使用CPU进行计算。")



class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")
        else:
            self.best_loss = val_loss
            self.counter = 0


if __name__ == "__main__":
    check_gpu()  # 检查gpu是否可用

    