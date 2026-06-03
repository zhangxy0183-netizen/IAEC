import argparse
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
from Depression_all.model.main_model import DModel
from Depression_all.tools.utils import EarlyStopping, load_config, save_epoch_info, validate_fea
from Depression_all.dataloader import AudioVideoDataset
from Depression_all.dataloader_2017 import AVEC2017AudioVideoDataset as AudioVideoDataset2
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
TORCH_USE_CUDA_DSA = 1
warnings.filterwarnings("ignore")
config = load_config('/home/b532root/account/b532zxy/workspace/Depression_all/config.yaml')
dataset = config['dataset']
if dataset != 'AVEC2014':
    base_root = config[dataset]['base_root']
else:
    base_root = None
train_path = config[dataset]['train_path']
dev_path = config[dataset]['dev_path']
w_o_ID = config['w_o_ID']
w_o_SE = config['w_o_SE']
w_o_SIM = config['w_o_SIM']
w_o_Video_Guide = config['w_o_Video_Guide']
audio_feature_type = config['audio_feature_type']
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser(description='Trainer for Multimodal Model')
parser.add_argument('--dataset', default=dataset, help="选择数据集类型")
parser.add_argument('--train_path', default=train_path, type=str, help='train标签路径')
parser.add_argument('--dev_path', default=dev_path, type=str, help='dev标签路径')
parser.add_argument('--base_root', default=base_root, type=str, help='dev标签路径')
parser.add_argument('--best_emonet_path', default=f'/home/b532root/data/b532zxy/{dataset}/weights_mask_ID_{w_o_ID}_SE_{w_o_SE}/mask.pth',
                     type=str, help='emonet的预训练权重保存路径')
parser.add_argument('--best_model_path', default=f'/home/b532root/data/b532zxy/{dataset}/weights_feature_ID_{w_o_ID}_SE_{w_o_SE}_SIM_{w_o_SIM}',
                     type=str, help='最好的模型保存路径')
parser.add_argument('--save_path', default=f'/home/b532root/data/b532zxy/{dataset}/weights_feature_ID_{w_o_ID}_SE_{w_o_SE}_SIM_{w_o_SIM}',
                     type=str, help='模型保存路径')
parser.add_argument('--best_log', default=f'/home/b532root/account/b532zxy/workspace/Depression_all/result/{dataset}_{current_time}_fea_ID_{w_o_ID}_SE_{w_o_SE}_SIM_{w_o_SIM}/best_log.txt',
                     type=str, help='最佳模型日志保存路径')
parser.add_argument('--training_log', default=f'/home/b532root/account/b532zxy/workspace/Depression_all/result/{dataset}_{current_time}_fea_ID_{w_o_ID}_SE_{w_o_SE}_SIM_{w_o_SIM}/training_log.txt',
                     type=str, help='训练日志保存路径')
parser.add_argument('--use_best_model', default=True, type=bool, help='使用模型')
parser.add_argument('--epochs', default=50, type=int, help='训练轮次')
parser.add_argument('--batch_size', default=4, type=int, help='批处理大小')
if dataset == 'AVEC2014':
    parser.add_argument('--lr', default=5e-4, type=float, help='学习率')
elif dataset == 'AVEC2017':
    parser.add_argument('--lr', default=5e-3, type=float, help='学习率')
else:
    parser.add_argument('--lr', default=5e-3, type=float, help='学习率')
parser.add_argument('--device', default=1, type=int, help='使用的GPU设备编号')
parser.add_argument('--momentum', default=0.9, type=float, help='动量参数')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减')
parser.add_argument('--best_l', default=1000, type=float, help='初始损失')
parser.add_argument('--video_feature_dim', default=128, type=int, help='视频特征维度')
parser.add_argument('--audio_feature_dim', default=128, type=int, help='音频特征维度')
parser.add_argument('--output_dim', default=1, type=int, help='输出维度')
parser.add_argument('--lstm_hidden_dim', default=128, type=int, help='lstm的隐藏层维度')
parser.add_argument('--temperature', default=0.8, type=float, help='情感对齐模块的温度参数')
parser.add_argument('--lambda_similarity', default=1.0, type=float, help='情感对齐损失权重')
parser.add_argument('--frame_num', default=15, type=int, help='帧数')
parser.add_argument('--CA_num_heads', default=8, type=int, help='融合模块的头数')
parser.add_argument('--CA_num_layers', default=2, type=int, help='可学习的注意力模块的层数')
parser.add_argument('--y_min', default=0, type=int, help='')
parser.add_argument('--y_max', default=41, type=int, help='')
parser.add_argument('--dropout', default=0.4, type=float, help='dropout')
parser.add_argument('--w_o_ID', default=w_o_ID, type=int, help='1 contain / 0 not contain')
parser.add_argument('--w_o_SE', default=w_o_SE, type=int, help='1 contain / 0 not contain')
parser.add_argument('--w_o_SIM', default=w_o_SIM, type=int, help='1 contain / 0 not contain')

args = parser.parse_args()

def main(args):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

    print("GPU是否可用......", end="")
    if torch.cuda.is_available():
        print("\t\tGPU可用√")
    else:
        print("\t\t使用CPU训练×") 
    
    print("创建保存路径......", end="")
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.best_model_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.training_log), exist_ok=True)
    os.makedirs(os.path.dirname(args.best_log), exist_ok=True)
    print("\t\t创建保存路径完成√")

    if args.dataset == 'AVEC2014':
        dataset_train = AudioVideoDataset(root_dir=args.train_path, audio_feature_type = audio_feature_type)
        dataset_dev = AudioVideoDataset(root_dir=args.dev_path, audio_feature_type = audio_feature_type)
    else:
        dataset_train = AudioVideoDataset2(base_root=args.base_root, label_csv=args.train_path, mode='train')
        dataset_dev = AudioVideoDataset2(base_root=args.base_root, label_csv=args.dev_path, mode='dev')
    labels_train = [dataset_train[i]['label'].item() for i in range(len(dataset_train))]

    if args.dataset == 'AVEC2014':
        num_samples_above_20 = sum(1 for l in labels_train if l >=20)
        num_samples_below_or_equal_20 = len(labels_train) - num_samples_above_20
        if num_samples_above_20 == 0:
            num_samples_above_20 = 1
        if num_samples_below_or_equal_20 == 0:
            num_samples_below_or_equal_20 = 1
        weights = [1.0 / num_samples_above_20 if l >= 20 else 1.0 / num_samples_below_or_equal_20 for l in labels_train]
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset_train), replacement=True)
        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, sampler=sampler, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)
    dev_loader = DataLoader(dataset=dataset_dev, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)
    
    train(train_loader, dev_loader, args)


def train(train_loader, test_loader, args):
    best_l, train_losses, val_losses = args.best_l, [], []
    model = DModel(args).cuda(args.device)
    
    print("加载已有权重......", end="")

    checkpoint_file = os.path.join(args.save_path, "checkpoint.pth")
    best_file = os.path.join(args.best_model_path, "best.pth")
    start_epoch = 0

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        if checkpoint['epoch'] < args.epochs - 1:
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            best_l = checkpoint.get('best_l', float('inf'))
            print(f"\t\t检测到检查点, 正在加载...... 从 epoch {start_epoch} 处继续训练√")
        else:
            print("\t\t检测到检查点文件为最后一轮保存，重新从头开始训练√")
            start_epoch = 0
    elif args.use_best_model:
        if os.path.exists(best_file):
            checkpoint = torch.load(best_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("\t\t加载了最佳模型权重，开始从头训练（不使用断点）√")
            start_epoch = 0
        else:
            print("\t\t最佳模型文件不存在，从头开始训练×")
            start_epoch = 0
    else:
        print("\t\t不使用最佳模型，从头开始训练×")
        start_epoch = 0
    
    print("初始化优化器......", end="")
    log_var = torch.zeros(1, requires_grad=True, device=args.device)
    optimizer_e = torch.optim.AdamW(
        list(model.parameters()) + [log_var],
        lr=args.lr, weight_decay=args.weight_decay
    )
    if dataset == 'AVEC2014':
        scheduler = CosineAnnealingWarmRestarts(optimizer_e, T_0=10, T_mult=2, eta_min=1e-5)
    elif dataset == 'AVEC2017':
        scheduler = CosineAnnealingWarmRestarts(optimizer_e, T_0=20, T_mult=1, eta_min=1e-5)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer_e, T_0=20, T_mult=1, eta_min=1e-5)


    print("\t\t初始化优化器完成√")
    print("早停正在启动......", end="")
    early_stopping = None
    if early_stopping is not None:
        print("\t\t早停机制启动完成√")
    else:
        print("\t\t早停机制启动失败×")

    criterion = nn.HuberLoss(delta=7.0)
    print("------------------begin training------------------")
    with open(args.training_log, "a") as f_log, open(args.best_log, "a") as f_best:
        for epoch in range(start_epoch, args.epochs):

            model.train()
            train_rmse, train_mae, train_loss, step, train_epoch_info = 0., 0., 0., 0., []
            loader = tqdm(train_loader)
            args.mode = 'train'
            for batch_idx, data in enumerate(loader):
                video_features = data['video_features'].cuda(args.device)
                audio_features = data['audio_features'].cuda(args.device)
                labels = data['label'].cuda(args.device).view(-1, 1)
                dir_name = data['dir_name']

                optimizer_e.zero_grad()

                outputs, _, _, ff_emotionAlignmentLoss = model(video_features, audio_features, mode='train')  
                loss_regression = criterion(outputs, labels)
                if args.w_o_SIM == 1:
                    loss = loss_regression \
                        + torch.exp(-log_var) * ff_emotionAlignmentLoss + log_var
                else:
                    loss = loss_regression
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer_e.step()
                
                outputs = outputs.view(-1, 1)
                predicted = outputs.view(-1).detach().cpu().numpy()
                true_labels = labels.view(-1).detach().cpu().numpy()
                for i, sample in enumerate(predicted):
                    train_epoch_info.append({
                        'dir_name': dir_name[i],
                        'predicted': predicted[i],
                        'label': true_labels[i],
                        'lr': optimizer_e.param_groups[0]['lr']
                    })
                
                outputs = outputs.to(args.device)
                rmse = torch.sqrt(torch.pow(torch.abs(outputs - labels), 2).mean()).item()
                mae = torch.abs(outputs - labels).mean().item()
                train_loss += loss.item()
                train_rmse += rmse                
                train_mae += mae
                step += 1
                
                loader.set_description(f"Epoch:{epoch+1} Step:{step} RMSE:{rmse:.2f} MAE:{mae:.2f}")
                

            train_rmse /= step
            train_mae /= step
            train_loss /= step
            train_losses.append(train_loss)

            model.eval()
            args.log_var = log_var
            val_rmes, val_mae, val_loss, val_epoch_info = validate_fea(args, model, test_loader, args.device, criterion, mode='dev')
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            plt.figure()
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Training and Validation Loss Curve (Epoch {epoch+1})')
            plt.xticks(range(1, len(train_losses) + 1))
            plt.close()

            if early_stopping is not None:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("早停机制触发，停止训练")
                    break

            checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_e.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_l': best_l
                }
            torch.save(checkpoint, checkpoint_file)

            if val_loss < best_l:
                torch.save(checkpoint, best_file)
                print(f"new model, Loss Improve: {best_l - val_loss:.2f}")
                best_l = val_loss
                f_best.write(f"Best Epoch {epoch + 1}: Train MAE:{train_mae:.2f} Train RMSE:{train_rmse:.2f}    Val MAE:{val_mae:.2f} Val RMSE:{val_rmes:.2f}  \n")
                f_best.flush()
            f_log.write(f"Epoch {epoch + 1}: Train MAE:{train_mae:.2f} Train RMSE:{train_rmse:.2f}      Val MAE:{val_mae:.2f} Val RMSE:{val_rmes:.2f} \n")
            f_log.flush()
            print('Train MAE:{:.2f} RMSE:{:.2f} \t Val MAE:{:.2f} RMSE:{:.2f}'.format(train_mae, train_rmse, val_mae, val_rmes))

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curve')

if __name__ == '__main__':
    main(args)