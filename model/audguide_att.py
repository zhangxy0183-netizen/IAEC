import torch
import torch.nn as nn
import torch.nn.functional as F

class BottomUpExtract(nn.Module):
    def __init__(self, embed_dim=128, dim=128):
        super(BottomUpExtract, self).__init__()
        self.attn = PositionAttn(embed_dim, dim)

    def forward(self, video, audio):
        feat = self.attn(video, audio)
        return feat


class PositionAttn(nn.Module):
    def __init__(self, embed_dim=128, dim=128):
        super(PositionAttn, self).__init__()
        self.affine_audio = nn.Linear(embed_dim, dim)
        self.affine_video = nn.Linear(embed_dim, dim)

        self.affine_v = nn.Linear(dim, 1, bias=False)
        self.affine_g = nn.Linear(dim, 1, bias=False)
        self.affine_h = nn.Linear(1, 1, bias=False)

        self.affine_feat = nn.Linear(embed_dim, dim)

        self.relu = nn.ReLU()

    def forward(self, video, audio):
        B, T, D = video.size()

        audio_agg = audio.mean(dim=1)

        video_t = self.relu(self.affine_video(video))
        audio_t = self.relu(self.affine_audio(audio_agg))

        content_v = self.affine_v(video_t) + self.affine_g(audio_t).unsqueeze(1)
        content_v = torch.tanh(content_v)
        z_t = self.affine_h(content_v)
        z_t = z_t.squeeze(-1)

        alpha_t = F.softmax(z_t, dim=-1) # (B, T)

        alpha_t = alpha_t.unsqueeze(-1)
        c_t = video * alpha_t

        return c_t
