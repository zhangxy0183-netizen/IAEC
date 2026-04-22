import torch
import os
from skimage import transform
import matplotlib.pyplot as plt
from Depression_k.tools.utils import total_losses, save_overlay_images
import csv

def validate(model, test_loader, args):
    model.eval()
    total_loss = 0.
    step = 0
    val_epoch_info = []
    # 用于保存每个样本的真实身份和预测身份
    identity_records = []

    with torch.no_grad():
        for images, heatmaps, labels, identity in test_loader:
            images = images.cuda(args.device)
            heatmaps = heatmaps.cuda(args.device)
            labels = labels.cuda(args.device).view(-1, 1)
            heatmaps = heatmaps / heatmaps.max()
            identity = identity.cuda(args.device).view(-1)
            
            # 前向传播，得到 heatmap 和特征向量以及身份预测（id_logits）
            mask, final_features, id_logits, regression_output = model(images)
            mask = mask.squeeze(dim=1)

            # 计算损失
            loss = total_losses(args, args.epoch, id_logits=id_logits, target_id=identity, mask=mask, heatmaps_ground_truth=heatmaps,
                                regression_output = regression_output, labels=labels)
            total_loss += loss.item()
            step += 1

            # 将 id_logits 转换为预测身份（假设是分类问题）
            # 如果 id_logits 是多分类的 logits，则可以用 argmax 获取预测的类别
            predicted_identity = torch.argmax(id_logits, dim=1)
            # 若模型是回归或输出单一数值，则可以直接记录 id_logits 的值
            # predicted_identity = id_logits.view(-1)
            regression_output = regression_output.view(-1, 1)
            predicted = regression_output.view(-1).detach().cpu().numpy()
            true_labels = labels.view(-1).detach().cpu().numpy()
            for i, sample in enumerate(predicted):
                val_epoch_info.append({
                    'predicted': predicted[i],
                    'label': true_labels[i],
                })


            # 将当前 batch 的真实身份和预测身份记录下来
            for true_id, pred_id in zip(identity.cpu().numpy(), predicted_identity.cpu().numpy()):
                identity_records.append([true_id, pred_id])

        loss_all = total_loss / step

    # 保存热力图叠加效果（已写好的函数）
    save_overlay_images(images, mask, identity.view(-1,1), id_logits.view(-1, 1), save_dir=args.heatmap, epoch=args.epoch, mode="predicted")
    save_overlay_images(images, heatmaps, identity.view(-1,1), id_logits.view(-1, 1), save_dir=args.heatmap, epoch=0, mode="true")

    # 将真实身份和预测身份保存为 CSV 文件
    csv_path = os.path.join(args.identity_records_path, f'identity_records_epoch_{args.epoch}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['True Identity', 'Predicted Identity'])
        writer.writerows(identity_records)
    print(f"身份记录已保存至 {csv_path}")

    return loss_all, val_epoch_info
