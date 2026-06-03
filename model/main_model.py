import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/b532root/account/b532zxy/workspace')
from Depression_all.model.visualEncoder import VisualFeatureExtractor   # 视频特征编码器
from Depression_all.model.FrameSim import EmotionAlignmentLoss

class DModel(nn.Module):
    def __init__(self, args):
        super(DModel, self).__init__()
        # freeze_layers = ['conv1', 'conv2', 'conv3']
        self.video_processModel = VisualFeatureExtractor(
            best_emonet_path=args.best_emonet_path,
            dropout=args.dropout,
            **({'print_if': args.print_if} if hasattr(args, 'print_if') else {})
        )
        self.emotionAlignmentModel = EmotionAlignmentLoss(lambda_similarity=args.lambda_similarity, temperature=args.temperature)

        fc_in = args.video_feature_dim + args.audio_feature_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, args.output_dim)
        )
        
    def forward(self, video_features, audio_features, mode='train'):
        v_features = self.video_processModel(video_features)
        emotionAlignmentLoss = self.emotionAlignmentModel(v_features, audio_features)
        if mode == 'pretrain':
            return v_features, audio_features
        else:
            v_pooled = v_features.mean(dim=1)  # (batch_size, feature_dim)
            audio_pooled = audio_features.mean(dim=1)  # (batch_size, feature_dim)
            fusion_features = torch.cat([v_pooled, audio_pooled], dim=-1)
            output = self.fc(fusion_features)  # 最终输出 (batch_size, output_dim)
            output = output.view(-1, 1)
            return output, v_features, audio_features, emotionAlignmentLoss        
 
        