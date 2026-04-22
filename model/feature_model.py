import torch
import torch.nn as nn
import torch.nn.functional as F

# 类内注意力
class IntraTypeMultiLocalAttention(nn.Module):
    def __init__(self, audio_feature_dim, dropout=0.2):
        super(IntraTypeMultiLocalAttention, self).__init__()
        self.feature_dim = audio_feature_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, frame_length, feature_dim)
        batch_size, frame_length, feature_dim = x.size()
        mean_vector = x.mean(dim=1, keepdim=True)  # (batch, 1, feature_dim)
        cosine_sim = F.cosine_similarity(x, mean_vector, dim=-1)  # (batch, frame_length)
        weights = F.softmax(cosine_sim, dim=1).unsqueeze(-1)  # (batch, frame_length, 1)
        attended_features = weights * x  # (batch, frame_length, feature_dim)
        attended_features = self.dropout(attended_features)

        return attended_features

# 类间注意力
class CrossTypeGlobalAttention(nn.Module):
    def __init__(self, audio_feature_dim, hidden_dim, dropout=0.2):
        super(CrossTypeGlobalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(audio_feature_dim, audio_feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ff_features, nw_features):
        ff_out_mean_vector = ff_features.mean(dim=1, keepdim=True)  # (batch, 1, feature_dim)
        nw_out_mean_vector = nw_features.mean(dim=1, keepdim=True)
        cosine_sim_ff = F.cosine_similarity(ff_features, nw_out_mean_vector, dim=-1)  # (batch, frame_length)
        weights_ff = F.softmax(cosine_sim_ff, dim=1).unsqueeze(-1)  # (batch, frame_length, 1)
        ff_attended_features = weights_ff * ff_features  # (batch, frame_length, hidden_dim)
        cosine_sim_nw = F.cosine_similarity(nw_features, ff_out_mean_vector, dim=-1)  # (batch, frame_length)
        weights_nw = F.softmax(cosine_sim_nw, dim=1).unsqueeze(-1)  # (batch, frame_length, 1)
        nw_attended_features = weights_nw * nw_features  # (batch, frame_length, hidden_dim)
        ff_attended_features = self.dropout(ff_attended_features)
        nw_attended_features = self.dropout(nw_attended_features)

        return ff_attended_features, nw_attended_features


class MultiLevelAttentionModel(nn.Module):
    def __init__(self, feature_dim, lstm_hidden_dim, dropout=0.2):
        super(MultiLevelAttentionModel, self).__init__()
        self.it_mla = IntraTypeMultiLocalAttention(feature_dim, dropout=dropout)
        self.ct_ga = CrossTypeGlobalAttention(feature_dim, hidden_dim=lstm_hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.lstm_ff = nn.LSTM(feature_dim, lstm_hidden_dim, 1, batch_first=True, dropout=dropout)
        self.lstm_nw = nn.LSTM(feature_dim, lstm_hidden_dim, 1, batch_first=True, dropout=dropout)
    def forward(self, ff_features, nw_features):
        # 应用 IT-MLA 机制
        ff_it_mla = self.it_mla(ff_features)  # 对阅读音频特征应用局部注意力
        nw_it_mla = self.it_mla(nw_features)  # 对自发音频特征应用局部注意力
        ff_attended_features, nw_attended_features = self.ct_ga(ff_it_mla, nw_it_mla)
        ff_attended_features, _ = self.lstm_ff(ff_attended_features)  # (batch, frame_length, hidden_dim)
        nw_attended_features, _ = self.lstm_nw(nw_attended_features)  # (batch, frame_length, hidden_dim)

        # Apply Dropout to LSTM outputs
        ff_attended_features = self.dropout(ff_attended_features)
        nw_attended_features = self.dropout(nw_attended_features)

        return ff_attended_features, nw_attended_features

import torch
import torch.nn as nn
from Depression_k.model.emonet import EmoNet

class VisualFeatureExtractor(nn.Module):
    def __init__(self, best_emonet_path=None, dropout=0.3, output_dim=128, freeze_layers=None):
        super(VisualFeatureExtractor, self).__init__()
        self.features = EmoNet(n_expression=output_dim)  # 使用 EmoNet 提取视觉特征
        self.dropout_layer = nn.Dropout(dropout)

        # 加载预训练权重
        if best_emonet_path:
            checkpoint = torch.load(best_emonet_path, map_location=torch.device('cpu'))
            if 'YYJC' in checkpoint:
                state_dict = checkpoint['YYJC']
                self.features.load_state_dict(state_dict, strict=False)
            else:
                raise KeyError("The key 'YYJC' is not present in the checkpoint.")

        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, images, frame_batch_size=5):
        batch_size, num_frames, C, H, W = images.size()
        outputs = []

        for start in range(0, num_frames, frame_batch_size):
            end = min(start + frame_batch_size, num_frames)
            batch_images = images[:, start:end].contiguous().reshape(-1, C, H, W)
            _, batch_output = self.features(batch_images)
            outputs.append(batch_output.view(batch_size, end - start, -1))

        emonet_output = torch.cat(outputs, dim=1)  # [batch_size, num_frames, output_dim]

        return self.dropout_layer(emonet_output)


class DModel(nn.Module):
    def __init__(self, args):
        super(DModel, self).__init__()
        # self.video_processModel = VisualFeatureExtractor(best_emonet_path=args.best_emonet_path, dropout=args.dropout)
        self.video_processModel = VisualFeatureExtractor(best_emonet_path=None, dropout=args.dropout)

        self.audio_processModel = MultiLevelAttentionModel(feature_dim=args.audio_feature_dim, 
                                                           lstm_hidden_dim=args.lstm_hidden_dim, dropout=args.dropout)

        # 融合层
        self.fc = nn.Sequential(
            nn.Linear(2 * args.lstm_hidden_dim + 2 * args.lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, args.output_dim)  # args.output_dim 是最终输出的维度
        )

    def forward(self, ff_video_features, ff_audio_features, nw_video_features, nw_audio_features, mode='train'):
        # 提取视频特征
        ffv_features = self.video_processModel(ff_video_features)  # 自发视频特征
        nwv_features = self.video_processModel(nw_video_features)  # 阅读视频特征

        # 提取音频特征
        ff_audio_features, nw_audio_features = self.audio_processModel(ff_audio_features, nw_audio_features)  # 音频特征

        # 归一化视频和音频特征
        ff_video_features = F.normalize(ffv_features, p=2, dim=-1)  
        nw_video_features = F.normalize(nwv_features, p=2, dim=-1)
        ff_audio_features = F.normalize(ff_audio_features, p=2, dim=-1)
        nw_audio_features = F.normalize(nw_audio_features, p=2, dim=-1)

        # 假设 `ff_video_features` 是输入的视频特征
        print("FF 视频特征最大值:", ffv_features.max().item())
        print("FF 视频特征最小值:", ffv_features.min().item())
        print("FF 视频特征均值:", ffv_features.mean().item())
        print("FF 视频特征标准差:", ffv_features.std().item())

        # 假设 `ff_audio_features` 是输入的音频特征
        print("FF 音频特征最大值:", ff_audio_features.max().item())
        print("FF 音频特征最小值:", ff_audio_features.min().item())
        print("FF 音频特征均值:", ff_audio_features.mean().item())
        print("FF 音频特征标准差:", ff_audio_features.std().item())

        # 对视频和音频特征进行池化
        ffv_pooled = ffv_features.mean(dim=1)  # (batch_size, feature_dim)
        nwv_pooled = nwv_features.mean(dim=1)  # (batch_size, feature_dim)
        ff_audio_pooled = ff_audio_features.mean(dim=1)  # (batch_size, feature_dim)
        nw_audio_pooled = nw_audio_features.mean(dim=1)  # (batch_size, feature_dim)

        # 拼接特征
        fusion_features = torch.cat([ffv_pooled, nwv_pooled, ff_audio_pooled, nw_audio_pooled], dim=-1)  # (batch_size, 4 * feature_dim)

        # 融合特征并输出
        output = self.fc(fusion_features)  # 最终输出 (batch_size, output_dim)

        return output

