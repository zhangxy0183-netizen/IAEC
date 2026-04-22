import argparse

import os
import warnings
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from skimage import transform
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
from Depression_k.model.emonet import EmoNet
from Depression_k.tools.utils import EarlyStopping, load_config, total_losses, save_epoch_info
from Depression_k.pretrain.dataloader import create_dataloaders
from Depression_k.pretrain.validate import validate

os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
TORCH_USE_CUDA_DSA = 1
warnings.filterwarnings("ignore")

# 读取配置文件
config = load_config('/home/b532root/account/b532zxy/workspace/Depression_k/config.yaml')
train_csv_path = config['pretrain']['train_csv_path']
val_csv_path = config['pretrain']['val_csv_path']
test_csv_path = config['pretrain']['test_csv_path']
file_identity_path = config['pretrain']['file_identity_path']
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 定义超参数及参数解析器
parser = argparse.ArgumentParser(description='Trainer for Multimodal Model')
parser.add_argument('--train_csv_path', default=train_csv_path, type=str, help='train_csv_path')
parser.add_argument('--val_csv_path', default=val_csv_path, type=str, help='val_csv_path')
parser.add_argument('--test_csv_path', default=test_csv_path, type=str, help='test_csv_path')
parser.add_argument('--save_path', default='/home/b532root/data/b532zxy/AVEC15/weights_mask',
                    type=str, help='模型保存路径')
parser.add_argument('--best_log', default=f'/home/b532root/account/b532zxy/workspace/Depression_k/result/{current_time}_mask/{current_time}_best_log.txt',
                    type=str, help='最佳模型日志保存路径')
parser.add_argument('--log_dir', default=f'/home/b532root/account/b532zxy/workspace/Depression_k/result/{current_time}_mask/log',
                     type=str, help='日志保存路径')
parser.add_argument('--training_log', default=f'/home/b532root/account/b532zxy/workspace/Depression_k/result/{current_time}_mask/{current_time}_training_log.txt',
                    type=str, help='训练日志保存路径')
parser.add_argument('--loss_curve', default=f'/home/b532root/account/b532zxy/workspace/Depression_k/result/{current_time}_mask/{current_time}_loss_curve.png',
                    type=str, help='损失曲线保存路径')
parser.add_argument('--epoch_curve', default=f'/home/b532root/account/b532zxy/workspace/Depression_k/result/{current_time}_mask/{current_time}_epoch_curve.png',
                    type=str, help='每轮损失曲线保存路径')
parser.add_argument('--heatmap', default=f'/home/b532root/account/b532zxy/workspace/Depression_k/result/{current_time}_mask/heatmap_overlays/',
                    type=str, help='heatmap保存路径')

parser.add_argument('--identity_records_path', default=f'/home/b532root/account/b532zxy/workspace/Depression_k/result/{current_time}_mask/identity_records_path/',
                    type=str, help='heatmap保存路径')
parser.add_argument('--epochs', default=100, type=int, help='训练轮次')
parser.add_argument('--batch_size', default=100, type=int, help='批处理大小')
parser.add_argument('--lr', default=5e-5, type=float, help='学习率')
parser.add_argument('--device', default=3, type=int, help='使用的GPU设备编号')
parser.add_argument('--momentum', default=0.9, type=float, help='动量参数')
parser.add_argument('--dropout_rate', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减')
parser.add_argument('--best_l', default=1000, type=float, help='初始损失')
parser.add_argument('--grl_lambda', default=0.5, type=float, help='grl_lambda')
# 新增超参数：标签一致性损失的权重
parser.add_argument('--lambda_heatmap', default=0.2, type=float, help='标签一致性损失权重')
# parser.add_argument('--scale', default=5.0, type=float, help='标签一致性损失权重')
parser.add_argument('--max_val', default=41, type=int, help='')
parser.add_argument('--min_val', default=0, type=int, help='')
parser.add_argument('--w_o_ID', default=1, type=int, help='1 contain / 0 not contain')
parser.add_argument('--w_o_SE', default=1, type=int, help='1 contain / 0 not contain')
args = parser.parse_args()

def main(args):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

    print("查看GPU是否可用......")
    if torch.cuda.is_available():
        print("GPU可用√")
    else:
        print("使用CPU训练......")

    print("创建保存路径...")
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.training_log), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.best_log), exist_ok=True)
    os.makedirs(os.path.dirname(args.loss_curve), exist_ok=True)
    os.makedirs(os.path.dirname(args.epoch_curve), exist_ok=True)
    os.makedirs(os.path.dirname(args.heatmap), exist_ok=True)
    os.makedirs(os.path.dirname(args.identity_records_path), exist_ok=True)

    print("创建保存路径完成√") 

    train_csv_path = args.train_csv_path
    val_csv_path = args.val_csv_path
    test_csv_path =args.test_csv_path

    train_loader, val_loader, test_loader = create_dataloaders(train_csv_path, val_csv_path, test_csv_path, file_identity_path, batch_size=args.batch_size)
    
    train(train_loader, val_loader, test_loader, args)

def train(train_loader, val_loader, test_loader, args):
    # 初始化最佳验证损失为一个较大的值
    best_l = args.best_l
    train_losses, val_losses = [], []

    # 初始化模型并将其移动到指定的设备上
    model = EmoNet(n_expression=128, grl_lambda=args.grl_lambda, w_o_SE = args.w_o_SE, dropout_rate = args.dropout_rate).cuda(args.device)

    best_file = os.path.join(args.save_path, "mask.pth")
    start_epoch = 0
    if os.path.exists(best_file):
        checkpoint = torch.load(best_file)
        # 判断检查点是否为最后一轮保存的（假设总训练轮次为 args.epochs）
        state_dict = checkpoint['YYJC']
        model.load_state_dict(state_dict)
        # best_l = checkpoint.get('best_l', float('inf'))
        start_epoch = checkpoint['epoch'] + 1
        print(f"\t\t检测到检查点，正在加载...... 从 epoch {start_epoch} 处继续训练√")
    
    print("初始化优化器......")
    optimizer_e = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_e, mode='min', factor=0.5, patience=20, verbose=True)
    # scheduler = CosineAnnealingWarmRestarts(optimizer_e, T_0=10, T_mult=3, eta_min=5e-8)
    print("初始化优化器完成")

    print("早停正在启动......", end="")
    early_stopping = EarlyStopping(patience=10, verbose=True)
    # early_stopping = None
    if early_stopping is not None:
        print("\t\t早停机制启动完成√")
    else:
        print("\t\t早停机制启动失败×")

    print("开始训练")
    with open(args.training_log, "w") as f_log, open(args.best_log, "w") as f_best:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            train_loss, train_epoch_info = 0., []
            step = 0
            loader = tqdm(train_loader)
            if epoch==80:
                best_l = 1000.0
            for images, heatmaps, labels, identity in loader:
                images = images.cuda(args.device)
                heatmaps = heatmaps.cuda(args.device)
                # labels 形状为 (batch, 1)
                labels = labels.cuda(args.device).view(-1, 1)
                identity = identity.cuda(args.device).view(-1)
                heatmaps = heatmaps / heatmaps.max()

                optimizer_e.zero_grad()
                mask, final_features, id_logits, regression_output = model(images)
                mask = mask.squeeze(dim=1)

                loss = total_losses(args, epoch, id_logits=id_logits, target_id=identity, mask=mask, heatmaps_ground_truth=heatmaps, 
                                    regression_output = regression_output, labels=labels)
                
                loss.backward()
                optimizer_e.step()

                regression_output = regression_output.view(-1, 1)
                predicted = regression_output.view(-1).detach().cpu().numpy()
                true_labels = labels.view(-1).detach().cpu().numpy()
                for i, sample in enumerate(predicted):
                    train_epoch_info.append({
                        'predicted': predicted[i],
                        'label': true_labels[i],
                        'lr': optimizer_e.param_groups[0]['lr']
                    })

                train_loss += loss.item()
                step += 1
                loader.set_description(f"Epoch:{epoch+1} Step:{step} LOSS:{loss:.6f}")

            save_epoch_info(train_epoch_info, args.log_dir, phase='train', epoch = epoch)

            train_loss /= step
            train_losses.append(train_loss)

            model.eval()
            args.epoch = epoch
            val_loss, val_epoch_info = validate(model, val_loader, args)
            save_epoch_info(val_epoch_info, args.log_dir, phase='eval', epoch = epoch)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            f_log.write(f"Epoch {epoch + 1}: Train loss:{train_loss:.6f}   Val loss:{val_loss:.6f}\n")
            f_log.flush()

            plt.figure()
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Training and Validation Loss Curve (Epoch {epoch+1})')
            plt.xticks(range(1, len(train_losses) + 1))
            plt.savefig(args.epoch_curve)
            plt.close()

            if early_stopping is not None:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("早停机制触发，停止训练")
                    break

            if val_loss < best_l:
                torch.save({'YYJC': model.state_dict(),
                            'best_l': best_l,
                            'epoch': epoch}, '{}/mask.pth'.format(args.save_path))
                print(f"保存模型, Loss Improve: {best_l - val_loss:.2f}")
                best_l = val_loss
                f_best.write(f"Best Epoch {epoch + 1}: Train loss:{train_loss:.6f}   Val loss:{val_loss:.6f}\n")
                f_best.flush()

            print('Train loss:{:.6f}   Val loss:{:.6f}'.format(train_loss, val_loss))

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curve')
    plt.savefig(args.loss_curve)

if __name__ == '__main__':
    main(args=args)


