import torch
import torch.nn as nn
import torch.nn.functional as F

class BottomUpExtract(nn.Module):
    """
    用于将视频和音频特征做融合，内部调用 PositionAttn 实现音频引导注意力。
    
    参数:
        embed_dim: 输入特征维度 (这里为 128)
        dim:       中间/输出特征维度 (可自由设置, 例如 128, 256 等)
    输入:
        video: (B, T, 128)
        audio: (B, T, 128)
    输出:
        feat: (B, dim)，是加权汇总后的视频表示
    """
    def __init__(self, embed_dim=128, dim=128):
        super(BottomUpExtract, self).__init__()
        self.attn = PositionAttn(embed_dim, dim)

    def forward(self, video, audio):
        # video: (B, T, 128)
        # audio: (B, T, 128)
        feat = self.attn(video, audio)
        return feat


class PositionAttn(nn.Module):
    """
    使用“音频引导”的方式，对 T 帧视频做注意力加权 (得到一个全局的视频向量)。
    这里将 T 帧音频简单聚合为一个全局音频向量，用它来对 T 帧视频做注意力。
    """
    def __init__(self, embed_dim=128, dim=128):
        super(PositionAttn, self).__init__()
        # 将输入的 audio、video 投影到同样大小 dim 的空间中
        self.affine_audio = nn.Linear(embed_dim, dim)
        self.affine_video = nn.Linear(embed_dim, dim)

        # 用于生成注意力的线性层
        # affine_v:  (dim -> 1)       每帧视频映射到 1 个分数
        # affine_g:  (dim -> 1)       全局音频映射到 1 个分数
        # affine_h:  (1 -> 1)         可做二次变换, 保持和原逻辑对应 (也可删去简化)
        self.affine_v = nn.Linear(dim, 1, bias=False)
        self.affine_g = nn.Linear(dim, 1, bias=False)
        self.affine_h = nn.Linear(1, 1, bias=False)

        # 对最终汇总后的视频向量再做一次线性映射
        self.affine_feat = nn.Linear(embed_dim, dim)

        self.relu = nn.ReLU()

    def forward(self, video, audio):
        """
        video: (B, T, 128)
        audio: (B, T, 128)
        return: (B, dim)
        """

        B, T, D = video.size()

        # 1) 将 T 帧音频做简单聚合（这里用 mean 作为示例）
        audio_agg = audio.mean(dim=1)           # (B, 128)

        # 2) 分别对 video、audio_agg 投影到 dim
        #    video_t: (B, T, dim)
        #    audio_t: (B, dim)
        video_t = self.relu(self.affine_video(video))    # (B, T, dim)
        audio_t = self.relu(self.affine_audio(audio_agg))# (B, dim)

        # 3) 计算注意力分数:
        #    affine_v(video_t): (B, T, 1)
        #    affine_g(audio_t): (B, 1)   -> unsqueeze(1)后 => (B, 1, 1) or (B, 1)
        #    使得每帧视频都加上同一份来自音频的引导信息
        content_v = self.affine_v(video_t) + self.affine_g(audio_t).unsqueeze(1)
        #    content_v: (B, T, 1)

        # 4) 过激活 + affine_h => 得到最终注意力 logits
        content_v = torch.tanh(content_v)
        z_t = self.affine_h(content_v)   # (B, T, 1)
        z_t = z_t.squeeze(-1)            # (B, T)

        # 5) softmax 得到注意力权重 alpha
        alpha_t = F.softmax(z_t, dim=-1) # (B, T)

        # 6) 用 alpha_t 对原始 video 做加权求和 (得到一个全局表示)
        alpha_t = alpha_t.unsqueeze(-1)  # (B, T, 1)
        c_t = video * alpha_t		# (B, T, 128)

        return c_t
