import torch
import os
import argparse
import datetime
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
import matplotlib.pyplot as plt
from Depression_all.tools.utils import load_config, save_overlay_images, total_losses
from Depression_all.pretrain.dataloader import create_dataloaders
config = load_config('/home/b532root/account/b532zxy/workspace/Depression_all/config.yaml')
test_csv_path = config['pretrain']['test_csv_path']
file_identity_path = config['pretrain']['file_identity_path']
w_o_ID = config['w_o_ID']
w_o_SE = config['w_o_SE']
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
from Depression_all.model.emonet import EmoNet

parser = argparse.ArgumentParser(description='Trainer for Multimodal Model')
parser.add_argument('--test_csv_path', default=test_csv_path, type=str, help='test_csv_path')
parser.add_argument('--file_identity_path', default=file_identity_path, type=str, help='file_identity_path')

parser.add_argument('--save_path', default=f'/home/b532root/data/b532zxy/AVEC2014/result/weights_mask_ID_{w_o_ID}_SE_{w_o_SE}.pth',
                    type=str, help='模型保存路径')
parser.add_argument('--heatmap', default=f'/home/b532root/account/b532zxy/workspace/Depression_all/result/A_test_emo/{current_time}_ID_{w_o_ID}_SE_{w_o_SE}/heatmap_overlays/',
                    type=str, help='heatmap保存路径')
parser.add_argument('--epochs', default=300, type=int, help='训练轮次')
parser.add_argument('--batch_size', default=100, type=int, help='批处理大小')
parser.add_argument('--lr', default=1e-5, type=float, help='学习率')
parser.add_argument('--device', default=1, type=int, help='使用的GPU设备编号')
parser.add_argument('--momentum', default=0.9, type=float, help='动量参数')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减')
parser.add_argument('--best_l', default=1000, type=float, help='初始损失')
parser.add_argument('--w_o_ID', default=w_o_ID, type=int, help='1 contain / 0 not contain')
parser.add_argument('--w_o_SE', default=w_o_SE, type=int, help='1 contain / 0 not contain')
parser.add_argument('--lambda_consistency', default=0.1, type=float, help='标签一致性损失权重')

args = parser.parse_args()

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()
_, _, test_loader = create_dataloaders("", "", args.test_csv_path, args.file_identity_path, batch_size=args.batch_size)

checkpoint_file = os.path.join(args.save_path)
if os.path.exists(checkpoint_file): 
    print("检测到模型......")
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
model = EmoNet(n_expression=128, w_o_SE=args.w_o_SE, dropout_rate=0)
model.load_state_dict(checkpoint['YYJC'])
model.to(args.device)
model.eval()
step= 0
print("创建保存路径...")
os.makedirs(os.path.dirname(args.heatmap), exist_ok=True)
print("创建保存路径完成√") 

with torch.no_grad():
    for images, heatmaps, labels, identity in test_loader:
        images = images.cuda(args.device)
        heatmaps = heatmaps.cuda(args.device)
        labels = labels.cuda(args.device).view(-1, 1)
        identity = identity.cuda(args.device).view(-1)
        heatmaps = heatmaps / heatmaps.max()
        mask, _, _, _ = model(images)
        mask = mask.squeeze(dim=1)
        step += 1
        if step == 9:
            save_overlay_images(images, mask, labels.view(-1), None, save_dir=args.heatmap, epoch=0, mode="predicted")
            save_overlay_images(images, heatmaps, labels.view(-1), None, save_dir=args.heatmap, epoch=0, mode="true")
            break
