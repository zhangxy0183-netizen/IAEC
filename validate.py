import torch
import torch.nn as nn
def validate_fea(args, model, test_loader, device, criterion, mode='dev'):
    """
    统一计算测试/验证阶段的指标和保存每个样本信息。
    
    Args:
        args: 参数对象，包含 log_var_ff、log_var_nw 等属性（在 with_alignment=True 时会用到）。
        model: 网络模型。
        test_loader: 数据加载器。
        device: 设备（如 cuda）。
        criterion: 损失函数。
        with_alignment (bool): 是否计算额外的情感对齐损失。
            True 表示同时计算 ff_emotionAlignmentLoss 和 nw_emotionAlignmentLoss，
            False 表示仅计算回归损失。
    
    Returns:
        rmse, mae, loss_all, val_epoch_info
    """
    model.eval()
    val_epoch_info = []  # 保存每个样本的信息

    with torch.no_grad():
        rmse, mae, loss_all = 0.0, 0.0, 0.0
        step = 0
        for data in test_loader:
            # 将各输入数据转移到指定设备
            ff_video_features = data['ff_video_features'].cuda(device)
            ff_audio_features = data['ff_audio_features'].cuda(device)
            nw_video_features = data['nw_video_features'].cuda(device)
            nw_audio_features = data['nw_audio_features'].cuda(device)
            labels = data['label'].cuda(device).to(torch.float32).view(-1, 1)
            dir_name = data['dir_name']

            results = model(ff_video_features, ff_audio_features, nw_video_features, nw_audio_features, mode='eval')
            outputs = results[0]
            loss_regression = criterion(outputs, labels)
            
            if mode == 'dev':
                ff_emotionAlignmentLoss = results[5]
                nw_emotionAlignmentLoss = results[6]
                loss = (loss_regression +
                        torch.exp(-args.log_var_ff) * ff_emotionAlignmentLoss + args.log_var_ff +
                        torch.exp(-args.log_var_nw) * nw_emotionAlignmentLoss + args.log_var_nw)
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
    """
    对模型进行验证。

    参数：
    model: 神经网络模型。
    test_loader: 验证数据加载器。
    device: 计算设备（如'cuda'或'cpu'）。
    criterion: 用于计算损失的函数。

    返回：
    rmse: 均方根误差。
    mae: 平均绝对误差。
    loss_all: 平均损失。
    """
    # 在不计算梯度的情况下进行验证，以减少内存消耗
    feature_model.eval()
    CAM_model.eval()  # Set model to evaluation mode
    val_epoch_info = []  # 记录每个样本的信息

    with torch.no_grad():
        # 初始化误差和损失变量
        rmse, mae, loss_all = 0., 0., 0.
        # 初始化步数变量，用于计算平均误差和损失
        step = 0
        # 遍历测试加载器中的每个样本
        for data in test_loader:
            # 从数据中解包各个组成部分
            ff_video_features = data['ff_video_features'].cuda(device)
            ff_audio_features = data['ff_audio_features'].cuda(device)
            nw_video_features = data['nw_video_features'].cuda(device)
            nw_audio_features = data['nw_audio_features'].cuda(device)
            labels = data['label'].cuda(device).to(torch.float32).view(-1, 1)   
            dir_name = data['dir_name'] 
            # labels = (labels - args.y_min) / (args.y_max - args.y_min)

            ffv_features, nwv_features, ffa_features, nwa_features = feature_model(ff_video_features, ff_audio_features, \
                                nw_video_features, nw_audio_features, mode='pretrain') 
            # outputs = outputs.view(-1, 1)
            final_output = CAM_model(ffa_features, ffv_features, nwa_features, nwv_features)
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
