import numpy as np
import torch
import torch.nn as nn
import yaml
import os
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
from skimage import transform

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
from torch.utils.data import DataLoader
from Depression_all.dataloader import AudioVideoDataset
from Depression_all.dataloader_2017 import AVEC2017AudioVideoDataset
from Depression_all.model.main_model import DModel
from Depression_all.model.CAM import CAM
config = load_config('/home/b532root/account/b532zxy/workspace/Depression_all/config.yaml')

def validate_fea(args, model, test_loader, device, criterion, mode='dev'):
    model.eval()
    val_epoch_info = []

    with torch.no_grad():
        rmse, mae, loss_all = 0.0, 0.0, 0.0
        step = 0
        for data in test_loader:
            # 将各输入数据转移到指定设备
            ff_video_features = data['video_features'].cuda(device)
            ff_audio_features = data['audio_features'].cuda(device)
            labels = data['label'].cuda(device).to(torch.float32).view(-1, 1)
            dir_name = data['dir_name']

            results = model(ff_video_features, ff_audio_features, mode='dev')
            outputs = results[0]
            loss_regression = criterion(outputs, labels)
            
            if mode == 'dev':
                ff_emotionAlignmentLoss = results[3]
                loss = (loss_regression +
                        torch.exp(-args.log_var) * ff_emotionAlignmentLoss + args.log_var)
            elif mode == 'test':
                loss = loss_regression
            else:
                raise ValueError(f"mode '{mode}' not supported")

            loss_all += loss.item()
            predicted = outputs.view(-1).cpu().numpy()
            true_labels = labels.view(-1).cpu().numpy()
            for i in range(len(predicted)):
                val_epoch_info.append({
                    'dir_name': dir_name[i],
                    'predicted': predicted[i],
                    'label': true_labels[i]
                })
            # 计算均方根误差和平均绝对误差
            rmse += torch.sqrt(torch.pow(torch.abs(outputs - labels), 2).mean()).item()
            mae += torch.abs(outputs - labels).mean().item()
            step += 1

        rmse /= step
        mae /= step
        loss_all /= step

    return rmse, mae, loss_all, val_epoch_info

def validate_cam(args, feature_model, CAM_model, test_loader, device, criterion):
    feature_model.eval()
    CAM_model.eval()  # Set model to evaluation mode
    val_epoch_info = []

    with torch.no_grad():
        rmse, mae, loss_all = 0., 0., 0.
        step = 0
        for data in test_loader:
            video_features = data['video_features'].cuda(device)
            audio_features = data['audio_features'].cuda(device)
            labels = data['label'].cuda(device).to(torch.float32).view(-1, 1)   
            dir_name = data['dir_name'] 

            ffv_features, ffa_features = feature_model(video_features, audio_features, mode='pretrain') 
            if args.mode == 'video':
                ffa_features = torch.zeros_like(ffa_features).cuda(device)
            elif args.mode == 'audio':
                ffv_features = torch.zeros_like(ffv_features).cuda(device)
            final_output = CAM_model(ffa_features, ffv_features)
            loss_regression = criterion(final_output, labels)
            loss = loss_regression
            loss_all += loss.item()

            final_output = final_output.to(args.device)
            final_output = final_output.view(-1, 1)
            predicted = final_output.view(-1).detach().cpu().numpy()
            true_labels = labels.view(-1).cpu().numpy()

            for i, sample_loss in enumerate(predicted):
                val_epoch_info.append({
                    'dir_name': dir_name[i],
                    'predicted': predicted[i],
                    'label': true_labels[i]
                })

            rmse += torch.sqrt(torch.pow(torch.abs(final_output - labels), 2).mean()).item()
            mae += torch.abs(final_output - labels).mean().item()
            step += 1
        
        rmse /= step
        mae /= step
        loss_all /= step
    return rmse, mae, loss_all, val_epoch_info

def get_best_cam_path(args):
    base_dir = os.path.join("/home/b532root/data/b532zxy", args.dataset, "result")
    
    test_mode_mapping = {
        'w_all': "weights_cam_ID_1_SE_1_SIM_1_VIDEO_1.pth",
        'w_o_ID': "weights_cam_ID_0_SE_1_SIM_1_VIDEO_1.pth",
        'w_o_SE': "weights_cam_ID_1_SE_0_SIM_1_VIDEO_1.pth",
        'w_o_SIM': "weights_cam_ID_1_SE_1_SIM_0_VIDEO_1.pth",
        'w_o_Video_Guide': "weights_cam_ID_1_SE_1_SIM_1_VIDEO_0.pth",
        'w_o_fs': "weights_cam_ID_1_SE_1_SIM_1_VIDEO_1_fs_0.pth"
    }
    
    if args.test_mode in test_mode_mapping:
        weights_file = test_mode_mapping[args.test_mode]
        return os.path.join(base_dir, weights_file)
    else:
        raise ValueError(f"Invalid test_mode: {args.test_mode}. Must be one of {list(test_mode_mapping.keys())}")


def model_test(args):
    args.w_o_ID = 1
    args.w_o_SE = 1
    args.w_o_SIM = 1
    args.w_o_Video_Guide = 1
    args.w_o_fs = 1

    if args.test_mode == 'w_all':
        pass
    elif args.test_mode == 'w_o_ID':
        args.w_o_ID = 0
    elif args.test_mode == 'w_o_SE':
        args.w_o_SE = 0
    elif args.test_mode == 'w_o_SIM':
        args.w_o_SIM = 0
    elif args.test_mode == 'w_o_Video_Guide':
        args.w_o_Video_Guide = 0
    elif args.test_mode == 'w_o_fs':
        args.w_o_fs = 0

    args.best_emonet_path = '/home/b532root/data/b532zxy/'+args.dataset+'/result/weights_mask_ID_'+str(args.w_o_ID)+'_SE_'+str(args.w_o_SE)+'.pth'
    args.best_feature_path = '/home/b532root/data/b532zxy/'+args.dataset+'/result/weights_feature_ID_'+str(args.w_o_ID)+'_SE_'+str(args.w_o_SE)+'_SIM_'+str(args.w_o_SIM)+'.pth'
    args.best_cam_path = get_best_cam_path(args)
    args.print_if = False
    os.makedirs(args.log_dir, exist_ok=True)

    feature_model = DModel(args).cuda(args.device)
    checkpoint_file = args.best_feature_path

    if args.stage == 2:
        print(f"执行cam————{args.test_mode}", end="\t")
        cam_model = CAM(args).cuda(args.device)
        checkpoint_file = args.best_cam_path
    elif args.stage == 1:
        print(f"执行fea————{args.test_mode}", end="\t")

    if os.path.exists(checkpoint_file): 
        checkpoint = torch.load(checkpoint_file, weights_only=True)
        if args.stage == 1:
            feature_model.load_state_dict(checkpoint['model_state_dict'])
        elif args.stage == 2:
            feature_model.load_state_dict(checkpoint['feature_model_state_dict'])
            cam_model.load_state_dict(checkpoint['CAM_model_state_dict'])
            cam_model.eval()
        feature_model.eval()

    if args.dataset == 'AVEC2014':
        dataset_test = AudioVideoDataset(root_dir=args.test_path, audio_noise_std=args.eval_audio_noise,
                                         visual_occlusion_ratio=args.eval_video_occlusion, 
                                         visual_occlusion_mode=args.eval_video_mode, audio_feature_type = args.audio_feature_type)
    else:
        dataset_test = AVEC2017AudioVideoDataset(base_root=args.base_root, label_csv=args.test_path,
                                                 eval_audio_noise_level=args.eval_audio_noise,
                                                 eval_occlusion_ratio=args.eval_video_occlusion,
                                                 eval_occlusion_mode=args.eval_video_mode, mode='test')
    
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    criterion_regression = nn.HuberLoss(delta=7.0)

    if args.stage == 1:
        test_rmse, test_mae, test_loss, test_epoch_info = validate_fea(
            args, feature_model, test_loader, args.device, criterion_regression, mode='test')
    elif args.stage == 2:
        test_rmse, test_mae, test_loss, test_epoch_info = validate_cam(
            args, feature_model, cam_model, test_loader, args.device, criterion_regression)
    save_epoch_info(test_epoch_info, args.log_dir, phase=f"{args.dataset}_{args.test_mode}_{args.mode}", print_if = False)
    print(f"{args.dataset}_{args.test_mode}_{args.mode}", end="\t")
    print(f"Test MAE: {test_mae:.4f}", end="\t")
    print(f"Test RMSE: {test_rmse:.4f}", end="\t")
    print(f"Test Loss: {test_loss:.4f}")
    print("------------------------------------------------------------------------------------------------")


def stage_test(args):
    args.w_o_ID = 1
    args.w_o_SE = 1
    args.w_o_SIM = 1
    args.w_o_Video_Guide = 1
    
    if args.test_mode == 'w_all':
        pass
    elif args.test_mode == 'w_o_ID':
        args.w_o_ID = 0
    elif args.test_mode == 'w_o_SE':
        args.w_o_SE = 0
    elif args.test_mode == 'w_o_SIM':
        args.w_o_SIM = 0
    elif args.test_mode == 'w_o_Video_Guide':
        args.w_o_Video_Guide = 0

    args.best_emonet_path = '/home/b532root/data/b532zxy/'+args.dataset+'/result/weights_mask_ID_'+str(args.w_o_ID)+'_SE_'+str(args.w_o_SE)+'.pth'
    args.best_feature_path = '/home/b532root/data/b532zxy/'+args.dataset+'/result/weights_feature_ID_'+str(args.w_o_ID)+'_SE_'+str(args.w_o_SE)+'_SIM_'+str(args.w_o_SIM)+'.pth'
    args.best_cam_path = get_best_cam_path(args)
    args.print_if = False
    os.makedirs(args.log_dir, exist_ok=True)

    feature_model = DModel(args).cuda(args.device)
    cam_model = CAM(args).cuda(args.device)
    checkpoint_file = args.best_cam_path

    if os.path.exists(checkpoint_file): 
        checkpoint = torch.load(checkpoint_file, weights_only=True)
        feature_model.load_state_dict(checkpoint['feature_model_state_dict'])
        cam_model.load_state_dict(checkpoint['CAM_model_state_dict'])
        cam_model.eval()
        feature_model.eval()

    if args.dataset == 'AVEC2014':
        dataset_test = AudioVideoDataset(root_dir=args.test_path)
    else:
        dataset_test = AVEC2017AudioVideoDataset(base_root=args.base_root, label_csv=args.test_path, mode='test')
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    criterion_regression = nn.HuberLoss(delta=7.0)

    test_rmse, test_mae, test_loss, test_epoch_info = validate_cam(
            args, feature_model, cam_model, test_loader, args.device, criterion_regression)

    save_epoch_info(test_epoch_info, args.log_dir, phase=f"{args.dataset}_{args.test_mode}_{args.mode}", print_if = False)
    print(f"{args.dataset}_{args.test_mode}_{args.mode}", end="\t")
    print(f"Test MAE: {test_mae:.4f}", end="\t")
    print(f"Test RMSE: {test_rmse:.4f}", end="\t")
    print(f"Test Loss: {test_loss:.4f}")
    print("------------------------------------------------------------------------------------------------")


def pretrain_validate(model, test_loader, args):
    model.eval()
    total_loss = 0.
    step = 0
    val_epoch_info = []

    with torch.no_grad():
        for images, heatmaps, labels, identity in test_loader:
            images = images.cuda(args.device)
            heatmaps = heatmaps.cuda(args.device)
            labels = labels.cuda(args.device).view(-1, 1)
            heatmaps = heatmaps / heatmaps.max()
            identity = identity.cuda(args.device).view(-1)
            
            mask, final_features, id_logits, regression_output = model(images)
            mask = mask.squeeze(dim=1)

            loss = total_losses(args, args.epoch, id_logits=id_logits, target_id=identity, mask=mask, heatmaps_ground_truth=heatmaps,
                                regression_output = regression_output, labels=labels)
            total_loss += loss.item()
            step += 1
            regression_output = regression_output.view(-1, 1)
            predicted = regression_output.view(-1).detach().cpu().numpy()
            true_labels = labels.view(-1).detach().cpu().numpy()
            for i, sample in enumerate(predicted):
                val_epoch_info.append({
                    'predicted': predicted[i],
                    'label': true_labels[i],
                })

        loss_all = total_loss / step

    save_overlay_images(images, mask, identity.view(-1,1), id_logits.view(-1, 1), save_dir=args.heatmap, epoch=args.epoch, mode="predicted")
    save_overlay_images(images, heatmaps, identity.view(-1,1), id_logits.view(-1, 1), save_dir=args.heatmap, epoch=0, mode="true")

    return loss_all, val_epoch_info


class AdaptiveWingLoss(nn.Module):
    def __init__(self, alpha=1.5, omega=7, epsilon=0.1, theta=0.05):
        super(AdaptiveWingLoss, self).__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def forward(self, pred, target):
        delta = torch.abs(target - pred)
        loss = torch.where(delta < self.theta,
                           self.omega * torch.log(1 + (delta / self.epsilon) ** self.alpha),
                           self.omega * (delta - self.theta))
        return torch.mean(loss)

def total_losses(args, epoch, id_logits, target_id, mask, heatmaps_ground_truth, regression_output, labels):
    loss_mask = AdaptiveWingLoss()(mask, heatmaps_ground_truth)
    loss_id = nn.CrossEntropyLoss()(id_logits, target_id)
    loss_regression = nn.HuberLoss(delta=7.0)(regression_output, labels)
    if args.w_o_ID == 0:
        loss_id = 0 

    return loss_mask + 0.05 * loss_id + 0.002 * loss_regression



def save_epoch_info(sample_info, log_dir, phase, epoch=None, print_if = True):
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
    batch_size, num_frames, C, H, W = images.shape
    _, _, _, h, w = masks.shape

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


def save_overlay_images(images, heatmaps, labels, predictions, save_dir, epoch, mode="predicted"):
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(min(len(images), 10)):  # 只保存前 10 张
        image = images[idx].permute(1, 2, 0).cpu().detach().numpy()  # (256, 256, 3)
        image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]

        heatmap = heatmaps[idx].cpu().detach().numpy()  # (64, 64)
        heatmap_resized = transform.resize(heatmap, (256, 256), anti_aliasing=True)  # 缩放到 (256, 256)

        plt.figure(figsize=(6, 6))
        plt.imshow(image)  # 原图
        plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)  # 叠加热力图
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"{mode}_epoch_{epoch}_idx_{idx}.png"))
        plt.close()

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

    