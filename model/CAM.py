from __future__ import absolute_import
from __future__ import division

from torch.nn import init
import torch
import math
from torch import nn
from torch.nn import functional as F
from .av_crossatten import DCNLayer
from .layer import LSTM
from .audguide_att import BottomUpExtract

class CAM(nn.Module):
    def __init__(self, args):
        super(CAM, self).__init__()
        self.ff_video_attn = BottomUpExtract(128, 128)
        self.nw_video_attn = BottomUpExtract(128, 128)

        self.ff_coattn1 = DCNLayer(args.video_feature_dim, args.audio_feature_dim, 1, args.dropout)
        self.ff_coattn2 = DCNLayer(args.video_feature_dim, args.audio_feature_dim, 1, args.dropout)
        self.nw_coattn1 = DCNLayer(args.video_feature_dim, args.audio_feature_dim, 1, args.dropout)
        self.nw_coattn2 = DCNLayer(args.video_feature_dim, args.audio_feature_dim, 1, args.dropout)

        # 在LSTM中添加dropout
        self.ff_Joint = LSTM(256, 128, 2, dropout=args.dropout, residual_embeddings=True)
        self.nw_Joint = LSTM(256, 128, 2, dropout=args.dropout, residual_embeddings=True)

        # 在注意力机制层添加dropout
        self.ff_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(args.dropout),  # 添加dropout
            nn.Linear(64, 1)
        )
        self.nw_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(args.dropout),  # 添加dropout
            nn.Linear(64, 1)
        )
        
        # 特征融合层保持不变，已经有dropout
        self.fusion_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, 1),
            nn.LeakyReLU()
        )
        
        # 添加一个额外的dropout层，用于在forward中的特征融合前使用
        self.feature_dropout = nn.Dropout(args.dropout)

        


    def forward(self, ffa_features, ffv_features, nwa_features, nwv_features):
        ffv_features = self.ff_video_attn(ffv_features, ffa_features)
        ffv_features, ffa_features = self.ff_coattn1(ffv_features, ffa_features)
        ffv_features, ffa_features = self.ff_coattn2(ffa_features, ffv_features)
        ff_avfeatures = torch.cat((ffv_features, ffa_features), -1)
        spontaneous_features = self.ff_Joint(ff_avfeatures)
        # 添加dropout
        spontaneous_features = self.feature_dropout(spontaneous_features)

        nwv_features = self.nw_video_attn(nwv_features, nwa_features)
        nwv_features, nwa_features = self.nw_coattn1(nwv_features, nwa_features)
        nwv_features, nwa_features = self.nw_coattn2(nwa_features, nwv_features)
        nw_avfeatures = torch.cat((nwv_features, nwa_features), -1)
        response_features = self.nw_Joint(nw_avfeatures)
        # 添加dropout
        response_features = self.feature_dropout(response_features)
        
        batch_size, num_frames, feature_dim = response_features.shape

        # Step 1: 使用注意力机制计算帧权重
        attention_weights_spontaneous = F.softmax(
            self.ff_attention(spontaneous_features.view(-1, feature_dim)).view(batch_size, num_frames), dim=1
        )  # (batch_size, num_frames)
        
        attention_weights_response = F.softmax(
            self.nw_attention(response_features.view(-1, feature_dim)).view(batch_size, num_frames), dim=1
        )  # (batch_size, num_frames)
        
        # Step 2: 加权求和得到每个样本的全局特征
        weighted_spontaneous = torch.sum(
            spontaneous_features * attention_weights_spontaneous.unsqueeze(-1), dim=1
        )  # (batch_size, feature_dim)
        
        weighted_response = torch.sum(
            response_features * attention_weights_response.unsqueeze(-1), dim=1
        )  # (batch_size, feature_dim)
        
        # Step 3: 融合两种特征
        fused_features = torch.cat([weighted_spontaneous, weighted_response], dim=-1)  # (batch_size, feature_dim * 2)
        depression_score = self.fusion_layer(fused_features)  # (batch_size, 1)
        
        return depression_score