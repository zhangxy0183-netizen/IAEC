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

        # 计算每个帧的平均特征向量
        mean_vector = x.mean(dim=1, keepdim=True)  # (batch, 1, feature_dim)

        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(x, mean_vector, dim=-1)  # (batch, frame_length)
        weights = F.softmax(cosine_sim, dim=1).unsqueeze(-1)  # (batch, frame_length, 1)
        
        # 加权求和得到注意力特征
        attended_features = weights * x  # (batch, frame_length, feature_dim)

        # Apply Dropout
        attended_features = self.dropout(attended_features)

        return attended_features


# 类间注意力
class CrossTypeGlobalAttention(nn.Module):
    def __init__(self, audio_feature_dim, hidden_dim, lstm_layers=3, dropout=0.2):
        super(CrossTypeGlobalAttention, self).__init__()
        self.lstm_ff = nn.LSTM(audio_feature_dim, hidden_dim, lstm_layers, batch_first=True, dropout=dropout)
        self.lstm_nw = nn.LSTM(audio_feature_dim, hidden_dim, lstm_layers, batch_first=True, dropout=dropout)
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(audio_feature_dim, audio_feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ff_features, nw_features):
        # LSTM 处理音频特征
        ff_out, _ = self.lstm_ff(ff_features)  # (batch, frame_length, hidden_dim)
        nw_out, _ = self.lstm_nw(nw_features)  # (batch, frame_length, hidden_dim)

        # 计算余弦相似度，基于LSTM输出进行交叉类型注意力计算
        ff_out_mean_vector = ff_out.mean(dim=1, keepdim=True)  # (batch, 1, feature_dim)
        nw_out_mean_vector = nw_out.mean(dim=1, keepdim=True)

        cosine_sim_ff = F.cosine_similarity(ff_out, nw_out_mean_vector, dim=-1)  # (batch, frame_length)
        weights_ff = F.softmax(cosine_sim_ff, dim=1).unsqueeze(-1)  # (batch, frame_length, 1)
        ff_attended_features = weights_ff * ff_out  # (batch, frame_length, hidden_dim)

        cosine_sim_nw = F.cosine_similarity(nw_out, ff_out_mean_vector, dim=-1)  # (batch, frame_length)
        weights_nw = F.softmax(cosine_sim_nw, dim=1).unsqueeze(-1)  # (batch, frame_length, 1)
        nw_attended_features = weights_nw * nw_out  # (batch, frame_length, hidden_dim)

        # Apply Dropout to attended features
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

        # 应用 CT-GA 机制 
        ff_attended_features, nw_attended_features = self.ct_ga(ff_it_mla, nw_it_mla)

        ff_attended_features, _ = self.lstm_ff(ff_attended_features)  # (batch, frame_length, hidden_dim)
        nw_attended_features, _ = self.lstm_nw(nw_attended_features)  # (batch, frame_length, hidden_dim)

        ff_attended_features = self.dropout(ff_attended_features)
        nw_attended_features = self.dropout(nw_attended_features)

        return ff_attended_features, nw_attended_features