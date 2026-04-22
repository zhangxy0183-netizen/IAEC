import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
from Depression_k.model.visualEncoder import VisualFeatureExtractor   # 视频特征编码器
from Depression_k.model.FrameSim import EmotionAlignmentLoss


class DModel(nn.Module):
    def __init__(self, args):
        super(DModel, self).__init__()
        # freeze_layers = ['conv1', 'conv2', 'conv3']
        self.video_processModel1 = VisualFeatureExtractor(best_emonet_path=args.best_emonet_path, dropout=args.dropout)
        self.video_processModel2 = VisualFeatureExtractor(best_emonet_path=args.best_emonet_path, dropout=args.dropout)

        # self.video_processModel = VisualFeatureExtractor(best_emonet_path=None, dropout=args.dropout)
        # self.audio_processModel = MultiLevelAttentionModel(feature_dim=args.audio_feature_dim, lstm_hidden_dim=args.lstm_hidden_dim, dropout=args.dropout)
        self.emotionAlignmentModel1 = EmotionAlignmentLoss(lambda_similarity=args.lambda_similarity, temperature=args.temperature)
        self.emotionAlignmentModel2 = EmotionAlignmentLoss(lambda_similarity=args.lambda_similarity, temperature=args.temperature)
        # 使用MLP 层
        self.fc = nn.Sequential(
            nn.Linear(2 * args.lstm_hidden_dim + 2 * args.lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 64),  # args.output_dim 是最终输出的维度
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, args.output_dim)
        )
        
    def forward(self, ff_video_features, ff_audio_features, nw_video_features, nw_audio_features,
                mode='train'):
        # 1、提取音频和视觉特征
        # torch.Size([2, 50, 128])
        ffv_features = self.video_processModel1(ff_video_features)
        # torch.Size([2, 50, 128])
        nwv_features = self.video_processModel2(nw_video_features)
        # 2、帧级特征
        ff_emotionAlignmentLoss = self.emotionAlignmentModel1(ffv_features, ff_audio_features)
        nw_emotionAlignmentLoss = self.emotionAlignmentModel2(nwv_features, nw_audio_features)
        # torch.Size([2, 50, 128]) torch.Size([2, 50, 128])
        # ff_audio_features, nw_audio_features = self.audio_processModel(ff_audio_features, nw_audio_features)

        # ffv_features = F.normalize(ffv_features, p=2, dim=-1)  # 对每一帧的特征进行L2归一化
        # nwv_features = F.normalize(nwv_features, p=2, dim=-1)  # 对每一帧的特征进行L2归一化

        # 使用 tanh 将特征限制在 [-1, 1]，避免范围过大
        ff_audio_features = torch.tanh(ff_audio_features)
        nw_audio_features = torch.tanh(nw_audio_features)

        if mode == 'pretrain':
            return ffv_features, nwv_features, ff_audio_features, nw_audio_features
        else:
            ffv_pooled = ffv_features.mean(dim=1)  # (batch_size, feature_dim)
            nwv_pooled = nwv_features.mean(dim=1)  # (batch_size, feature_dim)
            ff_audio_pooled = ff_audio_features.mean(dim=1)  # (batch_size, feature_dim)
            nw_audio_pooled = nw_audio_features.mean(dim=1)  # (batch_size, feature_dim)
            fusion_features = torch.cat([ffv_pooled, nwv_pooled, ff_audio_pooled, nw_audio_pooled], dim=-1)  # (batch_size, 4 * feature_dim)
            output = self.fc(fusion_features)  # 最终输出 (batch_size, output_dim)
            output = output.view(-1, 1)
            return output, ffv_features, nwv_features, ff_audio_features, nw_audio_features, ff_emotionAlignmentLoss, nw_emotionAlignmentLoss
        # class_logits, offset_logits = self.mlp(ff_output, nw_output)

        # MLP
        # output = self.fc(ff_fusion_features, nw_fusion_features)
        # return final_output, ff_emotionAlignmentLoss, nw_emotionAlignmentLoss
            

        # KAN
        # output1, _ , _ , _ = self.kan_ff_nw(torch.cat([final_ff_features, final_nw_features], dim=1))
        # output2, _ , _ , _ = self.kan_ffa_attended(ffa_attended_features)
        # output3, _ , _ , _ = self.kan_nwa_attended(nwa_attended_features)
        # return output1, output2, output3
        
 
        