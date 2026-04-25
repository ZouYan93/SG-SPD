import torch
import torch.nn as nn

"""Transformer编码器模块"""
class AdditionalTransformerEncoder(nn.Module):
    def __init__(self, 
                 in_feature = 840, 
                 nhead=8, 
                 feedforward_dim = 2048, 
                 num_layers = 6):
        super(AdditionalTransformerEncoder, self).__init__()
        self.output_dim = in_feature
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, 
                                                                   nhead=nhead, 
                                                                   dim_feedforward = feedforward_dim, 
                                                                   batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, 
                                                        num_layers=num_layers)
        
    def forward(self, input_batch, invert_masks_batch, **kwargs):
        encoded_embedding = self.additional_encoder(input_batch, src_key_padding_mask=invert_masks_batch)
        return encoded_embedding

"""全连接层编码器模块"""
class FullyConnectedEncoder(nn.Module):
    def __init__(self, 
                 in_feature, 
                 hidden_dim=[512, 256]):
        super(FullyConnectedEncoder, self).__init__()
        self.output_dim = hidden_dim[-1]
        layers = []
        layers.append(nn.Linear(in_feature, hidden_dim[0]))
        for i in range(1, len(hidden_dim)):
            layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
        self.fc = nn.Sequential(*layers)

    def forward(self, input_batch, **kwargs):
        encoded_embedding = self.fc(input_batch)
        return encoded_embedding

"""CNN编码器模块"""
class CNNEncoder(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(CNNEncoder, self).__init__()
        self.output_dim = out_feature
        self.conv1 = nn.Conv1d(in_feature, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, out_feature)

    def forward(self, input_batch, **kwargs):
        x = input_batch.permute(0, 2, 1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x

"""LSTM编码器模块"""
class LSTMEncoder(nn.Module):
    def __init__(self, in_feature, hidden_dim, num_layers):
        super(LSTMEncoder, self).__init__()
        self.output_dim = hidden_dim
        self.lstm = nn.LSTM(in_feature, hidden_dim, num_layers, batch_first=True)

    def forward(self, input_batch, **kwargs):
        output, (_, _) = self.lstm(input_batch)
        return output  # 返回所有时间步的输出作为编码结果