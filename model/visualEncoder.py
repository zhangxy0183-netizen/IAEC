import torch
import torch.nn as nn
from Depression_k.model.emonet import EmoNet

class VisualFeatureExtractor(nn.Module):
    def __init__(self, best_emonet_path=None, dropout=0.1, output_dim=128, freeze_layers=None):
        """
        视觉特征提取模块，支持预训练权重加载和分批处理。

        Args:
            best_emonet_path (str): 预训练权重路径。
            dropout (float): Dropout 概率。
            output_dim (int): 输出特征维度。
            freeze_layers (list of str): 需要冻结的层名称列表。
        """
        super(VisualFeatureExtractor, self).__init__()
        self.features = EmoNet(n_expression=output_dim)  # 使用 EmoNet 提取视觉特征
        self.dropout_layer = nn.Dropout(dropout)

        # 加载预训练权重
        if best_emonet_path:
            checkpoint = torch.load(best_emonet_path, map_location="cpu")
            state_dict = checkpoint['YYJC']
            print("加载emonet权重......", end="")
            self.features.load_state_dict(state_dict, strict=False)
            print("\t\t加载emonet权重完成√")
            # 加载预训练权重之后，先冻结整个模型
            for param in self.features.parameters():
                param.requires_grad = False

            for param in self.features.conv1x1_input_emo_2.parameters():
                param.requires_grad = True

            # 解冻 emo_net_2
            for param in self.features.emo_net_2.parameters():
                param.requires_grad = True

            # 解冻 emo_fc_2
            for param in self.features.emo_fc_2.parameters():
                param.requires_grad = True

            # 解冻 LSTM
            for param in self.features.lstm.parameters():
                param.requires_grad = True

            # 解冻 ConvBlock 模块
            for param in self.features.conv2.parameters():
                param.requires_grad = True
            for param in self.features.conv3.parameters():
                param.requires_grad = True
            for param in self.features.conv4.parameters():
                param.requires_grad = True

            # # 解冻身份判别分支
            # for param in self.features.identity_classifier.parameters():
            #     param.requires_grad = True
            # for param in self.features.depression_regression.parameters():
            #     param.requires_grad = True
        else:
            for param in self.features.parameters():
                param.requires_grad = True
            print("\t\t无emonet权重×")

    def forward(self, images, frame_batch_size=5):
        """
        前向传播，提取视觉特征，支持分批处理以节省显存。

        Args:
            images (torch.Tensor): 输入图像，形状为 [batch_size, num_frames, 3, H, W]。
            frame_batch_size (int): 每次处理的帧数。

        Returns:
            torch.Tensor: 提取的特征，形状为 [batch_size, num_frames, output_dim]。
        """
        batch_size, num_frames, C, H, W = images.size()
        outputs = []

        # 分批处理输入帧
        for start in range(0, num_frames, frame_batch_size):
            end = min(start + frame_batch_size, num_frames)
            batch_images = images[:, start:end].contiguous().reshape(-1, C, H, W)
            _ , batch_output, _, _ = self.features(batch_images)
            outputs.append(batch_output.view(batch_size, end - start, -1))

        # 合并所有帧的特征
        emonet_output = torch.cat(outputs, dim=1)  # [batch_size, num_frames, output_dim]

        # 应用 Dropout
        return self.dropout_layer(emonet_output)
