import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PriorLabelEmbedder(nn.Module):
    """先验标签嵌入模块，对应论文/源码中的 task、dataset、subject 注入。"""

    TASK_KEYS = ['<UNK>', '<SR>', '<NR>', '<TSR>']
    DATASET_KEYS = ['<UNK>', 'ZuCo1', 'ZuCo2']
    SUBJECT_KEYS = [
        '<UNK>', 'ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB',
        'ZKH', 'ZKW', 'ZMG', 'ZPH', 'YAC', 'YAG', 'YAK', 'YDG', 'YDR',
        'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 'YRH', 'YRK',
        'YRP', 'YSD', 'YSL', 'YTL'
    ]
    PROMPT_NUMS = [len(TASK_KEYS), len(DATASET_KEYS), len(SUBJECT_KEYS)]

    def __init__(self, dim, dropout=0.0):
        super(PriorLabelEmbedder, self).__init__()
        self.task_embedder = nn.Embedding(len(self.TASK_KEYS), dim)
        self.dataset_embedder = nn.Embedding(len(self.DATASET_KEYS), dim)
        self.subject_embedder = nn.Embedding(len(self.SUBJECT_KEYS), dim)
        self.mix_weight = nn.Parameter(torch.tensor([num / sum(self.PROMPT_NUMS) for num in self.PROMPT_NUMS], dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def encode_one(cls, task_key, dataset_key, subject_key):
        def find_id(keys, value):
            return keys.index(value) if value in keys else 0
        return [
            find_id(cls.TASK_KEYS, task_key),
            find_id(cls.DATASET_KEYS, dataset_key),
            find_id(cls.SUBJECT_KEYS, subject_key),
        ]

    def forward(self, prior_ids):
        weights = torch.softmax(self.mix_weight, dim=0)
        return self.dropout(
            weights[0] * self.task_embedder(prior_ids[:, 0].long())
            + weights[1] * self.dataset_embedder(prior_ids[:, 1].long())
            + weights[2] * self.subject_embedder(prior_ids[:, 2].long())
        )


def _sinusoidal_position_embedding(seq_len, dim, device):
    """生成一维正余弦位置编码。"""
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
    return pe.unsqueeze(0)


class SemKeyEncoderBlock(nn.Module):
    """带先验调制的 EEG Transformer 编码块，对齐 SemKey/GLIM 的 label injection。"""

    def __init__(self, dim, seq_len, nhead=8, mlp_ratio=4, dropout=0.1):
        super(SemKeyEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.time_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * seq_len))
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)
        nn.init.zeros_(self.time_mod[1].weight)
        nn.init.zeros_(self.time_mod[1].bias)

    def forward(self, x, key_padding_mask=None, prior=None):
        if prior is None:
            prior = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)

        scale_t, shift_t = self.time_mod(prior).chunk(2, dim=1)
        if scale_t.size(1) != x.size(1):
            scale_t = F.interpolate(scale_t.unsqueeze(1), size=x.size(1), mode='linear', align_corners=False).squeeze(1)
            shift_t = F.interpolate(shift_t.unsqueeze(1), size=x.size(1), mode='linear', align_corners=False).squeeze(1)
        x = x * (1 + scale_t.unsqueeze(-1)) + shift_t.unsqueeze(-1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(prior).chunk(6, dim=1)
        attn_in = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out = self.attn(attn_in, attn_in, attn_in, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * attn_out

        mlp_in = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        return x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)


class SemKeyQueryBlock(nn.Module):
    """learnable queries 通过 cross-attention 从 EEG 序列中汇聚固定长度 latent tokens。"""

    def __init__(self, dim, nhead=8, mlp_ratio=4, dropout=0.1):
        super(SemKeyQueryBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries, eeg_tokens, eeg_padding_mask=None):
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        q = self.norm2(queries)
        queries = queries + self.cross_attn(q, eeg_tokens, eeg_tokens, key_padding_mask=eeg_padding_mask, need_weights=False)[0]
        return queries + self.mlp(self.norm3(queries))


class SemKeyStyleEncoder(nn.Module):
    """
    SemKey 风格 EEG encoder。

    当前仍接收工程里的词级 EEG 特征 [B, L, 840]，但结构对齐源码：
    先验标签注入 -> Transformer EEG encoder -> learnable query merger。
    """

    def __init__(
            self,
            in_feature=840,
            input_len=60,
            hidden_dim=128,
            hidden_len=96,
            n_in_blocks=6,
            n_out_blocks=6,
            nhead=8,
            mlp_ratio=4,
            dropout=0.1):
        super(SemKeyStyleEncoder, self).__init__()
        self.output_dim = hidden_dim
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.input_proj = nn.Linear(in_feature, hidden_dim)
        self.prior_embedder = PriorLabelEmbedder(hidden_dim, dropout=dropout)
        self.in_blocks = nn.ModuleList([
            SemKeyEncoderBlock(hidden_dim, input_len, nhead=nhead, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_in_blocks)
        ])
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.shared_queries = nn.Parameter(torch.randn(1, hidden_len, hidden_dim) * 0.02)
        self.query_blocks = nn.ModuleList([
            SemKeyQueryBlock(hidden_dim, nhead=nhead, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_out_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_batch, invert_masks_batch=None, prior_ids=None, **kwargs):
        batch_size, seq_len, _ = input_batch.shape
        key_padding_mask = invert_masks_batch.bool() if invert_masks_batch is not None else None

        x = self.input_proj(input_batch)
        x = x + _sinusoidal_position_embedding(seq_len, x.size(-1), x.device)

        if prior_ids is None:
            prior_ids = torch.zeros(batch_size, 3, dtype=torch.long, device=x.device)
        prior = self.prior_embedder(prior_ids)

        for block in self.in_blocks:
            x = block(x, key_padding_mask=key_padding_mask, prior=prior)
        x = self.in_norm(x)

        queries = self.shared_queries.expand(batch_size, -1, -1)
        for block in self.query_blocks:
            queries = block(queries, x, eeg_padding_mask=key_padding_mask)
        return self.out_norm(queries)


class AdditionalTransformerEncoder(nn.Module):
    """Transformer 编码器。"""

    def __init__(self, in_feature=840, nhead=8, feedforward_dim=2048, num_layers=6):
        super(AdditionalTransformerEncoder, self).__init__()
        self.output_dim = in_feature
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            batch_first=True,
        )
        self.additional_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_batch, invert_masks_batch, **kwargs):
        return self.additional_encoder(input_batch, src_key_padding_mask=invert_masks_batch)


class FullyConnectedEncoder(nn.Module):
    """全连接编码器。"""

    def __init__(self, in_feature, hidden_dim=(512, 256)):
        super(FullyConnectedEncoder, self).__init__()
        self.output_dim = hidden_dim[-1]
        layers = [nn.Linear(in_feature, hidden_dim[0])]
        for i in range(1, len(hidden_dim)):
            layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
        self.fc = nn.Sequential(*layers)

    def forward(self, input_batch, **kwargs):
        return self.fc(input_batch)


class CNNEncoder(nn.Module):
    """CNN 编码器。"""

    def __init__(self, in_feature, out_feature):
        super(CNNEncoder, self).__init__()
        self.output_dim = out_feature
        self.conv1 = nn.Conv1d(in_feature, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, out_feature)

    def forward(self, input_batch, **kwargs):
        x = input_batch.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        return self.fc(x)


class LSTMEncoder(nn.Module):
    """LSTM 编码器。"""

    def __init__(self, in_feature, hidden_dim, num_layers):
        super(LSTMEncoder, self).__init__()
        self.output_dim = hidden_dim
        self.lstm = nn.LSTM(in_feature, hidden_dim, num_layers, batch_first=True)

    def forward(self, input_batch, **kwargs):
        output, _ = self.lstm(input_batch)
        return output
