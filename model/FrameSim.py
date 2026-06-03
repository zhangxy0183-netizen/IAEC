import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionAlignmentLoss(nn.Module):
    def __init__(self, lambda_similarity=1.0, temperature=1.0):
        super(EmotionAlignmentLoss, self).__init__()
        self.lambda_similarity = lambda_similarity
        self.temperature = temperature

    def calculate_similarity_matrix(self, features):
        """
        计算特征之间的相似度矩阵（余弦相似度 + softmax归一化）。
        """
        features = F.normalize(features, p=2, dim=-1)
        sim_matrix = torch.matmul(features, features.transpose(1, 2))  # (B, 50, 50)
        return F.softmax(sim_matrix / self.temperature, dim=-1)  # optional: smooth the distribution

    def js_divergence(self, p, q):
        """
        Jensen-Shannon Divergence（稳定、对称）。
        输入: 两个经过 softmax 的分布张量，shape: (B, N, N)
        """
        m = 0.5 * (p + q)
        p_log = torch.log(p + 1e-8)
        q_log = torch.log(q + 1e-8)

        js = 0.5 * (F.kl_div(p_log, m, reduction='batchmean') +
                    F.kl_div(q_log, m, reduction='batchmean'))
        return js

    def L_similarity_loss(self, video_matrix, audio_matrix):
        """
        计算视频与音频相似度矩阵之间的 MSE + JS 散度损失。
        """
        mse_loss = F.mse_loss(video_matrix, audio_matrix)
        js_loss = self.js_divergence(video_matrix, audio_matrix)

        loss = mse_loss + 0.1 * js_loss
        return loss

    def forward(self, video_features, audio_features):
        """
        综合损失函数，用于情感特征的对齐。
        """
        S_V = self.calculate_similarity_matrix(video_features)
        S_A = self.calculate_similarity_matrix(audio_features)

        L_similarity = self.L_similarity_loss(S_V, S_A)

        total_loss = self.lambda_similarity * L_similarity
        return total_loss

# === 测试用例 ===
if __name__ == '__main__':
    batch_size = 2
    video_features = torch.randn(batch_size, 50, 512)
    audio_features = torch.randn(batch_size, 50, 128)

    emotion_loss_fn = EmotionAlignmentLoss(lambda_similarity=1.0, temperature=1.0)
    loss = emotion_loss_fn(video_features, audio_features)
    print(f"Emotion Alignment Loss：{loss:.6f}")
