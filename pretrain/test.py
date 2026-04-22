import torch
import os
import argparse
import datetime
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
import matplotlib.pyplot as plt
from Depression_k.tools.utils import load_config, save_overlay_images, total_losses
from Depression_k.pretrain.dataloader import create_dataloaders
config = load_config('/home/b532root/account/b532zxy/workspace/Depression_k/config.yaml')
test_csv_path = config['pretrain']['test_csv_path']
file_identity_path = config['pretrain']['file_identity_path']
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
from Depression_k.pretrain.emonet import EmoNet

parser = argparse.ArgumentParser(description='Trainer for Multimodal Model')
parser.add_argument('--test_csv_path', default=test_csv_path, type=str, help='test_csv_path')
parser.add_argument('--file_identity_path', default=file_identity_path, type=str, help='file_identity_path')

parser.add_argument('--save_path', default='/home/b532root/data/b532zxy/AVEC15/weights_mask',
                    type=str, help='模型保存路径')
parser.add_argument('--mask_visual_picture_path', default=f'/home/b532root/account/b532zxy/workspace/Depression_k/result/test_emo/{current_time}/mask_visual_picture/fea_visual.png',
                    type=str, help='mask_visual_picture_path')
parser.add_argument('--heatmap', default=f'/home/b532root/account/b532zxy/workspace/Depression_k/result/test_emo/{current_time}/heatmap_overlays/',
                    type=str, help='heatmap保存路径')
parser.add_argument('--epochs', default=300, type=int, help='训练轮次')
parser.add_argument('--batch_size', default=100, type=int, help='批处理大小')
parser.add_argument('--lr', default=1e-5, type=float, help='学习率')
parser.add_argument('--device', default=1, type=int, help='使用的GPU设备编号')
parser.add_argument('--momentum', default=0.9, type=float, help='动量参数')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减')
parser.add_argument('--best_l', default=1000, type=float, help='初始损失')
# 新增超参数：标签一致性损失的权重
parser.add_argument('--lambda_consistency', default=0.1, type=float, help='标签一致性损失权重')

args = parser.parse_args()

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()
_, _, test_loader = create_dataloaders("", "", args.test_csv_path, args.file_identity_path, batch_size=args.batch_size)

checkpoint_file = os.path.join(args.save_path, "mask.pth")
if os.path.exists(checkpoint_file): 
    print("检测到模型......")
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
model = EmoNet(n_expression=128)
model.load_state_dict(checkpoint['YYJC'])
model.to(args.device)
model.eval()
total_loss = 0.
step = 0

print("创建保存路径...")
os.makedirs(os.path.dirname(args.heatmap), exist_ok=True)
os.makedirs(os.path.dirname(args.mask_visual_picture_path), exist_ok=True)
print("创建保存路径完成√") 

with torch.no_grad():
    for images, heatmaps, labels, identity in test_loader:
        images = images.cuda(args.device)
        heatmaps = heatmaps.cuda(args.device)
        labels = labels.cuda(args.device).view(-1, 1)
        identity = identity.cuda(args.device).view(-1)
        heatmaps = heatmaps / heatmaps.max()

        # 前向传播，得到 heatmap 和特征向量
        mask, final_features, id_logits = model(images)
        mask = mask.squeeze(dim=1)

        # 计算联合损失
        loss = total_losses(id_logits=id_logits, target_id=identity, mask=mask, heatmaps_ground_truth=heatmaps)

        total_loss += loss.item()
        step += 1

    loss_all = total_loss / step


# 保存最后一个 batch 的热力图叠加效果
save_overlay_images(images, mask, labels.view(-1), None, save_dir=args.heatmap, epoch=0, mode="predicted")
save_overlay_images(images, heatmaps, labels.view(-1), None, save_dir=args.heatmap, epoch=0, mode="true")
