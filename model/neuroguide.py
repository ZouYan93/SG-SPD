import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooler(nn.Module):
    """用一个可学习 query 从序列 hidden states 中汇聚全局向量。"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(AttentionPooler, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, hidden_states, mask=None):
        batch_size = hidden_states.size(0)
        query = self.query.expand(batch_size, -1, -1)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
        pooled = self.attn(query, hidden_states, hidden_states, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return pooled.squeeze(1)


class NeuroGuide(nn.Module):
    """
    无四标签版本的 SemKey Aligner。

    保留原论文/源码中最关键的两类对齐：
    1. 全局对齐：v_eeg 与 v_text 做 batch 内对比学习。
    2. 序列对齐：E_eeg 与 LLM encoder hidden states 做 commitment/MSE。
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, temperature=0.07, commitment_loss='mse'):
        super(NeuroGuide, self).__init__()
        self.eeg_pooler = AttentionPooler(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.text_pooler = AttentionPooler(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.temperature = temperature
        self.commitment_loss = commitment_loss

    def pool_eeg(self, eeg_embeds, eeg_mask=None):
        return self.eeg_pooler(eeg_embeds, eeg_mask)

    def pool_text(self, text_hidden, text_mask=None):
        return self.text_pooler(text_hidden, text_mask)

    def contrastive_loss(self, eeg_vector, text_vector):
        eeg_norm = F.normalize(eeg_vector, p=2, dim=-1)
        text_norm = F.normalize(text_vector, p=2, dim=-1)
        logits = torch.matmul(eeg_norm, text_norm.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()
            cosine = (eeg_norm * text_norm).sum(dim=-1).mean()
        return loss, {
            'neuroguide_clip_loss': float(loss.detach().cpu()),
            'neuroguide_clip_acc': float(acc.detach().cpu()),
            'neuroguide_clip_cosine': float(cosine.detach().cpu()),
        }

    def commitment_alignment(self, eeg_embeds, text_hidden, eeg_mask=None, text_mask=None):
        min_len = min(eeg_embeds.size(1), text_hidden.size(1))
        eeg_seq = F.normalize(eeg_embeds[:, :min_len, :], p=2, dim=-1)
        text_seq = F.normalize(text_hidden[:, :min_len, :], p=2, dim=-1)

        if eeg_mask is None:
            eeg_valid = torch.ones(eeg_seq.shape[:2], device=eeg_seq.device)
        else:
            eeg_valid = eeg_mask[:, :min_len].float()
        if text_mask is None:
            text_valid = torch.ones(text_seq.shape[:2], device=text_seq.device)
        else:
            text_valid = text_mask[:, :min_len].float()
        valid = (eeg_valid * text_valid).unsqueeze(-1)

        if self.commitment_loss == 'kl_div':
            loss = F.kl_div(eeg_seq * valid, text_seq * valid, reduction='batchmean')
        else:
            loss = ((eeg_seq - text_seq) ** 2 * valid).sum()
            loss = loss / (valid.sum() * eeg_seq.size(-1)).clamp_min(1.0)
        return loss, {
            'neuroguide_commitment_loss': float(loss.detach().cpu()),
        }
