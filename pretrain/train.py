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
from Depression_all.model.emonet import EmoNet
from Depression_all.tools.utils import EarlyStopping, load_config, total_losses, save_epoch_info
from Depression_all.pretrain.dataloader import create_dataloaders
from Depression_all.tools.utils import pretrain_validate

os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
TORCH_USE_CUDA_DSA = 1
warnings.filterwarnings("ignore")

config = load_config('/home/b532root/account/b532zxy/workspace/Depression_all/config.yaml')
dataset = config['dataset']
train_csv_path = config['pretrain']['train_csv_path']
val_csv_path = config['pretrain']['val_csv_path']
test_csv_path = config['pretrain']['test_csv_path']
file_identity_path = config['pretrain']['file_identity_path']
w_o_ID = config['w_o_ID']
w_o_SE = config['w_o_SE']
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser(description='Trainer for Multimodal Model')
parser.add_argument('--train_csv_path', default=train_csv_path, type=str, help='train_csv_path')
parser.add_argument('--val_csv_path', default=val_csv_path, type=str, help='val_csv_path')
parser.add_argument('--test_csv_path', default=test_csv_path, type=str, help='test_csv_path')
parser.add_argument('--save_path', default=f'/home/b532root/data/b532zxy/{dataset}/weights_mask_ID_{w_o_ID}_SE_{w_o_SE}',
                    type=str, help='模型保存路径')
parser.add_argument('--best_log', default=f'/home/b532root/account/b532zxy/workspace/Depression_all/result/{dataset}_{current_time}_mask_ID_{w_o_ID}_SE_{w_o_SE}/{current_time}_best_log.txt',
                    type=str, help='最佳模型日志保存路径')
parser.add_argument('--training_log', default=f'/home/b532root/account/b532zxy/workspace/Depression_all/result/{dataset}_{current_time}_mask_ID_{w_o_ID}_SE_{w_o_SE}/{current_time}_training_log.txt',
                    type=str, help='训练日志保存路径')
parser.add_argument('--heatmap', default=f'/home/b532root/account/b532zxy/workspace/Depression_all/result/{dataset}_{current_time}_mask_ID_{w_o_ID}_SE_{w_o_SE}/heatmap_overlays/',
                    type=str, help='heatmap保存路径')
parser.add_argument('--epochs', default=100, type=int, help='训练轮次')
parser.add_argument('--batch_size', default=100, type=int, help='批处理大小')
parser.add_argument('--lr', default=5e-5, type=float, help='学习率')
parser.add_argument('--device', default=1, type=int, help='使用的GPU设备编号')
parser.add_argument('--momentum', default=0.9, type=float, help='动量参数')
parser.add_argument('--dropout_rate', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减')
parser.add_argument('--best_l', default=10, type=float, help='初始损失')
parser.add_argument('--grl_lambda', default=0.5, type=float, help='grl_lambda')
parser.add_argument('--lambda_heatmap', default=0.2, type=float, help='')
parser.add_argument('--w_o_ID', default=w_o_ID, type=int, help='1 contain / 0 not contain')
parser.add_argument('--w_o_SE', default=w_o_SE, type=int, help='1 contain / 0 not contain')
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
    os.makedirs(os.path.dirname(args.best_log), exist_ok=True)
    os.makedirs(os.path.dirname(args.heatmap), exist_ok=True)

    print("创建保存路径完成√") 

    train_csv_path = args.train_csv_path
    val_csv_path = args.val_csv_path
    test_csv_path =args.test_csv_path

    train_loader, val_loader, test_loader = create_dataloaders(train_csv_path, val_csv_path, test_csv_path, file_identity_path, batch_size=args.batch_size)
    
    train(train_loader, val_loader, test_loader, args)

def train(train_loader, val_loader, test_loader, args):
    best_l = args.best_l
    train_losses, val_losses = [], []
    model = EmoNet(n_expression=128, grl_lambda=args.grl_lambda, w_o_SE = args.w_o_SE, dropout_rate = args.dropout_rate).cuda(args.device)

    best_file = os.path.join(args.save_path, "mask.pth")
    start_epoch = 0
    if os.path.exists(best_file):
        checkpoint = torch.load(best_file)
        state_dict = checkpoint['YYJC']
        model.load_state_dict(state_dict)
        start_epoch = checkpoint['epoch'] + 1
        print(f"\t\t检测到检查点，正在加载......")
    
    print("初始化优化器......")
    optimizer_e = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_e, mode='min', factor=0.5, patience=20, verbose=True)
    print("初始化优化器完成")

    print("早停正在启动......", end="")
    early_stopping = None
    if early_stopping is not None:
        print("\t\t早停机制启动完成√")

    print("开始训练")
    with open(args.training_log, "w") as f_log, open(args.best_log, "w") as f_best:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            train_loss = 0.
            step = 0
            loader = tqdm(train_loader)
            if epoch==80:
                best_l = 1000.0
            for images, heatmaps, labels, identity in loader:
                images = images.cuda(args.device)
                heatmaps = heatmaps.cuda(args.device)
                # labels (batch, 1)
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

                train_loss += loss.item()
                step += 1
                loader.set_description(f"Epoch:{epoch+1} Step:{step} LOSS:{loss:.6f}")

            train_loss /= step
            train_losses.append(train_loss)

            model.eval()
            args.epoch = epoch
            val_loss, val_epoch_info = pretrain_validate(model, val_loader, args)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            f_log.write(f"Epoch {epoch + 1}: Train loss:{train_loss:.6f}   Val loss:{val_loss:.6f}\n")
            f_log.flush()

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

if __name__ == '__main__':
    main(args=args)


