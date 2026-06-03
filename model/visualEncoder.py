import torch
import torch.nn as nn
from Depression_all.model.emonet import EmoNet

class VisualFeatureExtractor(nn.Module):
    def __init__(self, best_emonet_path=None, dropout=0.1, output_dim=128, freeze_layers=None, print_if=True):
        """
            best_emonet_path (str): 预训练权重路径。
            dropout (float): Dropout 概率。
            output_dim (int): 输出特征维度。
            freeze_layers (list of str): 需要冻结的层名称列表。
        """
        super(VisualFeatureExtractor, self).__init__()
        self.features = EmoNet(n_expression=output_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # 加载预训练权重
        if best_emonet_path:
            checkpoint = torch.load(best_emonet_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint['YYJC']
            # print("visualEncoder---冻结hourGlass......")
            if print_if is True:
                print("加载emonet权重......", end="")
            self.features.load_state_dict(state_dict, strict=False)
            if print_if is True:
                print("\t\t加载emonet权重完成√")
            # 加载预训练权重之后，先冻结整个模型
            for param in self.features.parameters():
                param.requires_grad = False

            for param in self.features.conv1x1_input_emo_2.parameters():
                param.requires_grad = True

            for param in self.features.emo_net_2.parameters():
                param.requires_grad = True

            for param in self.features.emo_fc_2.parameters():
                param.requires_grad = True

            for param in self.features.lstm.parameters():
                param.requires_grad = True

            for param in self.features.conv2.parameters():
                param.requires_grad = True
            for param in self.features.conv3.parameters():
                param.requires_grad = True
            for param in self.features.conv4.parameters():
                param.requires_grad = True

        else:
            for param in self.features.parameters():
                param.requires_grad = True
            print("\t\t无emonet权重,训练所有参数×")

    def forward(self, images, frame_batch_size=5):
        batch_size, num_frames, C, H, W = images.size()
        outputs = []

        for start in range(0, num_frames, frame_batch_size):
            end = min(start + frame_batch_size, num_frames)
            batch_images = images[:, start:end].contiguous().reshape(-1, C, H, W)
            _ , batch_output, _, _ = self.features(batch_images)
            outputs.append(batch_output.view(batch_size, end - start, -1))

        emonet_output = torch.cat(outputs, dim=1)
        return self.dropout_layer(emonet_output)
