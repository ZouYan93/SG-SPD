import torch
import torch.nn as nn

"""解码器框架"""
class Decoder(nn.Module):
    def __init__(self, pretrained_name, pretrained_model, encoder_model, tokenizer=None):
        super(Decoder, self).__init__()
        self.pretrained_name = pretrained_name
        self.pretrained = pretrained_model
        self.encoder = encoder_model
        self.tokenizer = tokenizer
        self.feature_dim = self.encoder.output_dim
        self.decoder_embedding_size = self.pretrained.config.d_model

        # 线性层将编码器输出的特征维度转换为解码器的输入维度
        self.fc1 = nn.Linear(self.feature_dim, self.decoder_embedding_size)

    def forward(self, inputs_batch, masks_batch, invert_masks_batch, target_ids_batch):
        """
            input_batch: [batch_size, seq_len, feature_dim]
            masks_batch: [batch_size, seq_len] (解码器输入的注意力掩码, 0表示需要mask的位置)
            invert_masks_batch: [batch_size, seq_len] (编码器输入的注意力掩码, 1表示需要mask的位置)
            target_ids_batch: [batch_size, target_seq_len] (解码器目标序列的token ids)
        """
        # 获取编码器输出的特征
        encoder_outputs = self.encoder(inputs_batch, 
                                       src_key_padding_mask=invert_masks_batch)
        # 投影编码器输出到解码器输入维度
        encoded_embedding = self.fc1(encoder_outputs)
        
        if self.pretrained_name == 'T5':
            input_ids = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(encoded_embedding.device)
            self.task_embedding = self.pretrained.shared(input_ids).to(encoded_embedding.device)
            task_embedding = self.task_embedding.repeat(encoded_embedding.size(0), 1, 1).to(encoded_embedding.device)
            encoded_embedding = torch.cat((task_embedding, encoded_embedding), dim=1)
            input_masks_batch = torch.cat((torch.ones(encoded_embedding.size(0), task_embedding.size(1)).to(encoded_embedding.device), input_masks_batch), dim=1)

        # 解码器生成输出
        output = self.pretrained(inputs_embeds=encoded_embedding, attention_mask=masks_batch, 
                                return_dict=True, labels=target_ids_batch)
        return output
    
    @torch.no_grad()
    def generate(
            self,
            inputs_batch, 
            masks_batch, 
            invert_masks_batch, 
            generation_config = None,
            logits_processor = None,
            stopping_criteria = None,
            prefix_allowed_tokens_fn= None,
            synced_gpus= None,
            assistant_model = None,
            streamer= None,
            negative_prompt_ids= None,
            negative_prompt_attention_mask = None,
            **kwargs
    ):
        # 获取编码器输出的特征
        encoder_outputs = self.encoder(inputs_batch, 
                                       src_key_padding_mask=invert_masks_batch)
        # 投影编码器输出到解码器输入维度
        encoded_embedding = self.fc1(encoder_outputs)

        if self.pretrained_name == 'T5':
            input_ids = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(encoded_embedding.device)
            self.task_embedding = self.pretrained.shared(input_ids).to(encoded_embedding.device)
            task_embedding = self.task_embedding.repeat(encoded_embedding.size(0), 1, 1).to(encoded_embedding.device)
            encoded_embedding = torch.cat((task_embedding, encoded_embedding), dim=1)
            input_masks_batch = torch.cat((torch.ones(encoded_embedding.size(0), task_embedding.size(1)).to(encoded_embedding.device), input_masks_batch), dim=1)

        # 解码器生成输出
        output=self.pretrained.generate(
            inputs_embeds = encoded_embedding,
            attention_mask = masks_batch,
            return_dict = True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs)

        return output