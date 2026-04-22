import math
import torch
import torch.nn as nn
class DenseCoAttn(nn.Module):
    def __init__(self, dim1, dim2, dropout):
        super(DenseCoAttn, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])
        self.query1_linear = nn.Linear(dim1 + dim2, dim1)  # Query1映射到Key1的维度
        self.query2_linear = nn.Linear(dim1 + dim2, dim2)  # Query2映射到Key2的维度
        self.key1_linear = nn.Linear(dim1, dim1)
        self.key2_linear = nn.Linear(dim2, dim2)
        self.value1_linear = nn.Linear(dim1, dim1)
        self.value2_linear = nn.Linear(dim2, dim2)
        self.relu = nn.ReLU()

    def forward(self, value1, value2):
        batch_size, seq_len1, dim1 = value1.size()
        batch_size, seq_len2, dim2 = value2.size()

        # Concatenate features along the last dimension, not the sequence dimension
        joint = torch.cat((value1, value2), dim=-1)  # Shape: (batch_size, seq_len1, dim1 + dim2)

        # Linear transformations for Query
        query1 = self.query1_linear(joint)  # Shape: (batch_size, seq_len1, dim1)
        query2 = self.query2_linear(joint)  # Shape: (batch_size, seq_len2, dim2)

        # Linear transformations for Key and Value
        key1 = self.key1_linear(value1)  # Shape: (batch_size, seq_len1, dim1)
        key2 = self.key2_linear(value2)  # Shape: (batch_size, seq_len2, dim2)
        value1 = self.value1_linear(value1)  # Shape: (batch_size, seq_len1, dim1)
        value2 = self.value2_linear(value2)  # Shape: (batch_size, seq_len2, dim2)

        # Attention computation
        weighted1, attn1 = self.qkv_attention(query1, key1, value1, dropout=self.dropouts[0])
        weighted2, attn2 = self.qkv_attention(query2, key2, value2, dropout=self.dropouts[1])

        return weighted1, weighted2

    def qkv_attention(self, query, key, value, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled dot-product attention
        scores = torch.tanh(scores)

        if dropout:
            scores = dropout(scores)

        weighted = torch.matmul(scores, value)  # Weighted sum
        return self.relu(weighted), scores

