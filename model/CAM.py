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
        self.dataset = args.dataset
        self.audio_attn = BottomUpExtract(args.video_feature_dim, args.audio_feature_dim)
        if args.w_o_Video_Guide == 1:
            self.video_attn = BottomUpExtract(args.audio_feature_dim, args.video_feature_dim)
            self.coattn = DCNLayer(args.audio_feature_dim, args.video_feature_dim, 1, args.dropout)
        else:
            self.coattn = DCNLayer(args.video_feature_dim, args.video_feature_dim, 1, args.dropout)
        self.Joint = LSTM(args.audio_feature_dim + args.video_feature_dim, 128, 2, dropout=args.dropout, residual_embeddings=True)
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            nn.Linear(64, 1)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, 1),
            nn.LeakyReLU()
        )
        
        self.feature_dropout = nn.Dropout(args.dropout)
        self.w_o_Video_Guide = getattr(args, 'w_o_Video_Guide', 1)
        self.w_o_fs = getattr(args, 'w_o_fs', 1)

    def forward(self, a_features, v_features, mode = 'multimodal'):
        guided_a_features = self.audio_attn(v_features, a_features)
        if self.w_o_Video_Guide == 1:
            guided_v_features = self.video_attn(a_features, v_features)
            coed_v_features, coed_a_features = self.coattn(guided_v_features, guided_a_features)
        else:
            coed_v_features, coed_a_features = self.coattn(v_features, guided_a_features)
        avfeatures = torch.cat((coed_v_features, coed_a_features), -1)
        final_features = self.feature_dropout(self.Joint(avfeatures))
        
        batch_size, num_frames, feature_dim = final_features.shape

        # Step 1: 使用注意力机制计算帧权重
        if self.w_o_fs == 1:
            attention_weights = F.softmax(
                self.attention(final_features.view(-1, feature_dim)).view(batch_size, num_frames), dim=1
            )  # (batch_size, num_frames)

            final_features = torch.sum(
                final_features * attention_weights.unsqueeze(-1), dim=1
            )  # (batch_size, feature_dim)
        else:
            # 不使用帧级注意力
            final_features = torch.mean(final_features, dim=1)  # (batch_size, feature_dim)
        
        depression_score = self.fusion_layer(final_features)  # (batch_size, 1)
        
        return depression_score