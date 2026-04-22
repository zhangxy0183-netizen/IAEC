import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import torch
import torch.nn as nn

def compute_final_score(class_logits, offset_logits):
    """
    根据分类和偏移 logits 计算最终预测分数。
    Args:
        class_logits (torch.Tensor): 二分类 logits，形状为 (B, 2)。
        offset_logits (torch.Tensor): 偏移 logits，形状为 (B, 30)。
    Returns:
        torch.Tensor: 最终预测分数，形状为 (B,)。
    """
    # 找到分类概率最大的类别
    class_pred = torch.argmax(class_logits, dim=-1)  # (B,) -> 类别索引 {0, 1}

    # 找到偏移概率最大的索引
    offset_pred = torch.argmax(offset_logits, dim=-1)  # (B,) -> 偏移索引 {0, 1, ..., 29}

    # 计算基础值
    base_value = torch.where(class_pred == 0, 15, 30)  # (B,) -> 基础值 {15, 30}

    # 计算偏移值
    offset_value = offset_pred - 15  # 偏移值范围 [-15, +15]

    # 计算最终分数
    final_score = base_value + offset_value  # (B,)

    return final_score


def generate_labels(gt_labels):
    """
    根据真实标签生成 class_labels 和 offset_labels。
    Args:
        gt_labels (torch.Tensor): 原始标签，形状为 (B,)。
    Returns:
        torch.Tensor: class_labels，表示类别标签，形状为 (B,)。
        torch.Tensor: offset_labels，表示偏移标签，形状为 (B,)。
    """
    # class_labels: 二分类标签
    # 如果 gt_label >= 30，则属于第二类（class_label = 1）；否则属于第一类（class_label = 0）
    class_labels = torch.zeros_like(gt_labels)
    class_labels[gt_labels >= 30] = 1
    class_labels = class_labels.long()

    # base_value: 每个类别的基础值
    # 如果 class_label == 0，基础值为 15；如果 class_label == 1，基础值为 30
    base_value = torch.where(class_labels == 0, 15, 30)

    # offset_labels: 偏移量标签，范围 [0, 29]
    # 偏移量 = gt_label - 基础值 + 15
    offset_labels = gt_labels - base_value + 15
    offset_labels = offset_labels.long()
    return class_labels, offset_labels


def compute_loss(gt_labels, class_logits, offset_logits, alpha=0.5):
    """
    计算综合损失，包括分类损失和偏移损失。
    Args:
        gt_labels (torch.Tensor): 原始标签，形状为 (B,)。
        class_logits (torch.Tensor): 二分类 logits，形状为 (B, 2)。
        offset_logits (torch.Tensor): 偏移 logits，形状为 (B, 30)。
        alpha (float): 分类损失和偏移损失的平衡系数。
    Returns:
        torch.Tensor: 总损失值。
        torch.Tensor: 分类损失值。
        torch.Tensor: 偏移损失值。
    """
    # 生成类别标签和偏移标签
    class_labels, offset_labels = generate_labels(gt_labels)

    # 分类损失
    class_labels = class_labels.squeeze(-1).long()
    classification_loss = F.cross_entropy(class_logits, class_labels)

    # 偏移损失
    offset_labels = offset_labels.squeeze(-1).long()
    offset_loss = F.cross_entropy(offset_logits, offset_labels)

    # 综合损失
    total_loss = alpha * classification_loss + (1 - alpha) * offset_loss

    return total_loss, classification_loss, offset_loss



class CCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        """
        CCC损失函数，用于计算预测值和真实值之间的一致性
        Args:
            eps: 防止数值计算中的除零问题
        """
        super(CCCLoss, self).__init__()
        self.eps = eps

    def forward(self, predictions, target):
        """
        Args:
            predictions: [batch, 1]，模型的预测值
            target: [batch, 1]，真实值
        Returns:
            1 - CCC值
        """
        # 展平预测值和真实值，转换为 [batch]
        predictions = predictions.view(-1)
        target = target.view(-1)

        # 计算皮尔逊相关系数的分子和分母
        vx = predictions - predictions.mean()
        vy = target - target.mean()
        rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + self.eps)

        # 计算均值和标准差
        x_m, y_m = predictions.mean(), target.mean()
        x_s, y_s = predictions.std(), target.std()

        # 计算CCC
        ccc = 2 * rho * x_s * y_s / (x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2 + self.eps)

        # 返回 1 - CCC 作为损失
        return 1 - ccc


class SmoothLabelLoss(nn.Module):
    def __init__(self, num_classes=46, sigma=2.0):
        super(SmoothLabelLoss, self).__init__()
        self.num_classes = num_classes
        self.sigma = sigma  # 控制平滑标签的宽度，sigma 越大，平滑范围越广

    def forward(self, logits, labels):
        """
        计算模型输出与平滑标签分布之间的差异。
        :param logits: 模型输出，形状为 (batch_size, num_classes)
        :param labels: 真实标签，形状为 (batch_size,)
        :return: 损失值
        """
        # 生成平滑标签
        smooth_labels = self.create_smooth_labels(labels)  # 形状为 (batch_size, num_classes)

        # 使用KL散度计算损失
        loss = F.kl_div(F.log_softmax(logits, dim=-1), smooth_labels, reduction='batchmean')
        return loss

    def create_smooth_labels(self, labels):
        """
        生成平滑标签分布
        :param labels: 长度为 batch_size 的标签数组，每个标签值在 [0, num_classes-1] 范围内
        :return: 平滑标签矩阵，形状为 (batch_size, num_classes)
        """
        batch_size = labels.size(0)
        smooth_labels = torch.zeros(batch_size, self.num_classes).to(labels.device)

        for i in range(batch_size):
            label = labels[i].item()  # 获取标签值（0-45）

            # 创建高斯分布的平滑标签，距离标签越近的概率越大
            dist = torch.arange(self.num_classes).float().to(labels.device)  # [0, 1, 2, ..., num_classes-1]
            dist = torch.abs(dist - label)  # 计算每个位置与标签的绝对距离
            dist = torch.exp(-dist**2 / (2 * self.sigma**2))  # 使用高斯分布进行平滑
            dist = dist / dist.sum()  # 归一化，使得总和为1，得到概率分布

            smooth_labels[i, :] = dist

        return smooth_labels