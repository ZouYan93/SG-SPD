import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""池化层"""
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        first_token_tensor = x[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

"""位置编码"""
class PositionEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

"""EEG通道与频段注意力模块"""
class EEGChannelBandAttention(nn.Module):
    def __init__(
            self,
            in_feature=840,
            num_channels=105,
            num_bands=8,
            reduction=8,
            dropout=0.0):
        super(EEGChannelBandAttention, self).__init__()
        if in_feature != num_channels * num_bands:
            raise ValueError(
                f'in_feature 必须等于 num_channels*num_bands, '
                f'当前为 {in_feature} != {num_channels}*{num_bands}'
            )
        self.in_feature = in_feature
        self.num_channels = num_channels
        self.num_bands = num_bands

        channel_hidden = max(1, num_channels // reduction)
        band_hidden = max(1, num_bands // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Linear(num_channels, channel_hidden),
            nn.ReLU(),
            nn.Linear(channel_hidden, num_channels),
        )
        self.band_mlp = nn.Sequential(
            nn.Linear(num_bands, band_hidden),
            nn.ReLU(),
            nn.Linear(band_hidden, num_bands),
        )
        self.dropout = nn.Dropout(dropout)
        self.last_channel_attention = None
        self.last_band_attention = None

    def forward(self, input_batch, **kwargs):
        """
            input_batch: [batch_size, seq_len, 840]
            输出形状与输入一致，可直接接入原有EEG编码器。
        """
        batch_size, seq_len, feature_dim = input_batch.size()
        if feature_dim != self.in_feature:
            raise ValueError(f'输入特征维度应为 {self.in_feature}, 实际为 {feature_dim}')

        x = input_batch.view(batch_size, seq_len, self.num_channels, self.num_bands)

        channel_avg = x.mean(dim=3)
        channel_max = x.max(dim=3).values
        channel_attention = torch.sigmoid(
            self.channel_mlp(channel_avg) + self.channel_mlp(channel_max)
        ).unsqueeze(-1)

        band_avg = x.mean(dim=2)
        band_max = x.max(dim=2).values
        band_attention = torch.sigmoid(
            self.band_mlp(band_avg) + self.band_mlp(band_max)
        ).unsqueeze(-2)

        attended = x * channel_attention * band_attention
        attended = self.dropout(attended)

        self.last_channel_attention = channel_attention.detach()
        self.last_band_attention = band_attention.detach()
        return attended.reshape(batch_size, seq_len, feature_dim)

def masked_mean_pool(sequence_output, masks_batch=None, invert_masks_batch=None, eps=1e-8):
    """
        sequence_output: [batch_size, seq_len, hidden_dim]
        masks_batch: [batch_size, seq_len], 1表示有效位置
        invert_masks_batch: [batch_size, seq_len], 1表示padding位置
    """
    if masks_batch is None and invert_masks_batch is not None:
        masks_batch = (~invert_masks_batch.bool()).float()
    if masks_batch is None:
        return sequence_output.mean(dim=1)

    weights = masks_batch.to(sequence_output.device).float().unsqueeze(-1)
    pooled = (sequence_output * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(eps)
    return pooled / denom

"""跨模态语义对齐损失"""
class SemanticAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07, soft_negative=False):
        super(SemanticAlignmentLoss, self).__init__()
        self.temperature = temperature
        self.soft_negative = soft_negative

    def forward(self, eeg_sentence_repr, text_sentence_repr):
        eeg_repr = F.normalize(eeg_sentence_repr, dim=-1)
        text_repr = F.normalize(text_sentence_repr, dim=-1)
        logits = torch.matmul(eeg_repr, text_repr.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        if not self.soft_negative:
            return F.cross_entropy(logits, labels)

        with torch.no_grad():
            text_sim = torch.matmul(text_repr, text_repr.t()).clamp_min(0.0)
            weights = 1.0 - text_sim
            weights.fill_diagonal_(1.0)

        exp_logits = torch.exp(logits)
        weighted_denominator = (exp_logits * weights).sum(dim=1)
        positive_logits = logits.diagonal()
        return -(positive_logits - torch.log(weighted_denominator.clamp_min(1e-8))).mean()

"""EEG与随机噪声输入的区分约束"""
class EEGNoiseContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(EEGNoiseContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, eeg_sentence_repr, noise_sentence_repr, text_sentence_repr):
        eeg_repr = F.normalize(eeg_sentence_repr, dim=-1)
        noise_repr = F.normalize(noise_sentence_repr, dim=-1)
        text_repr = F.normalize(text_sentence_repr, dim=-1)

        eeg_text_sim = F.cosine_similarity(eeg_repr, text_repr, dim=-1)
        noise_text_sim = F.cosine_similarity(noise_repr, text_repr, dim=-1)
        return F.relu(self.margin - eeg_text_sim + noise_text_sim).mean()