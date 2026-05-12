import torch
import torch.nn as nn
from model.helper import EEGChannelBandAttention, SemanticAlignmentLoss, EEGNoiseContrastiveLoss, masked_mean_pool

"""解码器框架"""
class Decoder(nn.Module):
    def __init__(self, pretrained_name, pretrained_model, encoder_model, tokenizer=None, enhancement_config=None):
        super(Decoder, self).__init__()
        self.pretrained_name = pretrained_name
        self.pretrained = pretrained_model
        self.encoder = encoder_model
        self.tokenizer = tokenizer
        self.enhancement_config = enhancement_config or {}
        self.eeg_dim = self.encoder.in_feature
        self.feature_dim = self.encoder.output_dim
        self.decoder_embedding_size = self.pretrained.config.d_model

        # 线性层将编码器输出的特征维度转换为解码器的输入维度
        self.fc1 = nn.Linear(self.feature_dim, self.decoder_embedding_size)
        self.eeg_attention = None
        self.semantic_alignment_loss = None
        self.noise_contrastive_loss = None

        attention_config = self.enhancement_config.get('attention', {})
        if attention_config.get('enabled', False):
            self.eeg_attention = EEGChannelBandAttention(
                in_feature=attention_config.get('in_feature', self.eeg_dim),
                num_channels=attention_config.get('num_channels', 105),
                num_bands=attention_config.get('num_bands', 8),
                reduction=attention_config.get('reduction', 8),
                dropout=attention_config.get('dropout', 0.0),
            )

        alignment_config = self.enhancement_config.get('semantic_alignment', {})
        if alignment_config.get('enabled', False):
            self.semantic_alignment_loss = SemanticAlignmentLoss(
                temperature=alignment_config.get('temperature', 0.07),
                soft_negative=alignment_config.get('soft_negative', False),
            )

        noise_config = self.enhancement_config.get('noise_contrast', {})
        if noise_config.get('enabled', False):
            self.noise_contrastive_loss = EEGNoiseContrastiveLoss(
                margin=noise_config.get('margin', 0.2),
            )

    def _apply_eeg_attention(self, inputs_batch):
        if self.eeg_attention is None:
            return inputs_batch
        return self.eeg_attention(inputs_batch)

    def _encode_eeg(self, inputs_batch, invert_masks_batch):
        attended_inputs = self._apply_eeg_attention(inputs_batch)
        return self.encoder(attended_inputs, invert_masks_batch=invert_masks_batch)

    def _project_to_pretrained_space(self, encoder_outputs):
        return self.fc1(encoder_outputs)

    def _append_t5_task_embedding(self, encoded_embedding, masks_batch):
        if self.pretrained_name != 'T5':
            return encoded_embedding, masks_batch
        input_ids = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(encoded_embedding.device)
        task_embedding = self.pretrained.shared(input_ids).to(encoded_embedding.device)
        task_embedding = task_embedding.repeat(encoded_embedding.size(0), 1, 1).to(encoded_embedding.device)
        encoded_embedding = torch.cat((task_embedding, encoded_embedding), dim=1)
        masks_batch = torch.cat((torch.ones(encoded_embedding.size(0), task_embedding.size(1)).to(encoded_embedding.device), masks_batch), dim=1)
        
        return encoded_embedding, masks_batch

    def _labels_to_text_encoder_inputs(self, target_ids_batch):
        input_ids = target_ids_batch.clone()
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        input_ids[input_ids == -100] = pad_token_id
        attention_mask = (input_ids != pad_token_id).long()
        return input_ids, attention_mask

    def _get_text_sentence_repr(self, target_ids_batch):
        input_ids, attention_mask = self._labels_to_text_encoder_inputs(target_ids_batch)
        encoder = self.pretrained.get_encoder()
        text_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return masked_mean_pool(text_outputs.last_hidden_state, masks_batch=attention_mask)

    def _compute_auxiliary_loss(self, inputs_batch, masks_batch, invert_masks_batch, target_ids_batch, encoded_embedding):
        aux_loss = encoded_embedding.new_tensor(0.0)
        aux_loss_dict = {}
        needs_text_repr = self.semantic_alignment_loss is not None or self.noise_contrastive_loss is not None
        if not needs_text_repr:
            return aux_loss, aux_loss_dict

        eeg_sentence_repr = masked_mean_pool(encoded_embedding, masks_batch=masks_batch)
        text_sentence_repr = self._get_text_sentence_repr(target_ids_batch)

        alignment_config = self.enhancement_config.get('semantic_alignment', {})
        if self.semantic_alignment_loss is not None:
            weight = alignment_config.get('weight', 0.1)
            align_loss = self.semantic_alignment_loss(eeg_sentence_repr, text_sentence_repr)
            aux_loss = aux_loss + weight * align_loss
            aux_loss_dict['semantic_alignment_loss'] = float(align_loss.detach().cpu())

        noise_config = self.enhancement_config.get('noise_contrast', {})
        if self.noise_contrastive_loss is not None:
            weight = noise_config.get('weight', 0.1)
            noise_std = noise_config.get('noise_std', 1.0)
            noise_inputs = torch.randn_like(inputs_batch) * noise_std
            noise_encoder_outputs = self._encode_eeg(noise_inputs, invert_masks_batch)
            noise_embedding = self._project_to_pretrained_space(noise_encoder_outputs)
            noise_sentence_repr = masked_mean_pool(noise_embedding, masks_batch=masks_batch)
            noise_loss = self.noise_contrastive_loss(
                eeg_sentence_repr,
                noise_sentence_repr,
                text_sentence_repr,
            )
            aux_loss = aux_loss + weight * noise_loss
            aux_loss_dict['noise_contrastive_loss'] = float(noise_loss.detach().cpu())

        return aux_loss, aux_loss_dict
    
    def forward(self, inputs_batch, masks_batch, invert_masks_batch, target_ids_batch):
        """
            input_batch: [batch_size, seq_len, feature_dim]
            masks_batch: [batch_size, seq_len] (解码器输入的注意力掩码, 0表示需要mask的位置)
            invert_masks_batch: [batch_size, seq_len] (编码器输入的注意力掩码, 1表示需要mask的位置)
            target_ids_batch: [batch_size, target_seq_len] (解码器目标序列的token ids)
        """
        # 获取编码器输出的特征
        encoder_outputs = self.encoder(inputs_batch, 
                                       invert_masks_batch=invert_masks_batch)
        # 投影编码器输出到解码器输入维度
        encoded_embedding = self._project_to_pretrained_space(encoder_outputs)
        auxiliary_loss, aux_loss_dict = self._compute_auxiliary_loss(
            inputs_batch,
            masks_batch,
            invert_masks_batch,
            target_ids_batch,
            encoded_embedding,
        )
        encoded_embedding, pretrained_masks_batch = self._append_t5_task_embedding(encoded_embedding, masks_batch)
        
        # 解码器生成输出
        output = self.pretrained(inputs_embeds=encoded_embedding, attention_mask=pretrained_masks_batch,
                                 return_dict=True, labels=target_ids_batch)
        output.loss = output.loss + auxiliary_loss
        output.aux_loss_dict = aux_loss_dict
        return output
    
    @torch.no_grad()
    def generate(
            self,
            inputs_batch, 
            masks_batch, 
            invert_masks_batch, 
            target_ids_batch,
            generation_config = None,
            logits_processor = None,
            stopping_criteria = None,
            prefix_allowed_tokens_fn= None,
            synced_gpus= False,
            assistant_model = None,
            streamer= None,
            negative_prompt_ids= None,
            negative_prompt_attention_mask = None,
            **kwargs
    ):
        # 获取编码器输出的特征
        encoder_outputs = self.encoder(inputs_batch,
                                       invert_masks_batch=invert_masks_batch)
        # 投影编码器输出到解码器输入维度
        encoded_embedding = self._project_to_pretrained_space(encoder_outputs)
        encoded_embedding, masks_batch = self._append_t5_task_embedding(encoded_embedding, masks_batch)

        # 解码器生成输出
        output=self.pretrained.generate(
            inputs_embeds = encoded_embedding,
            attention_mask = masks_batch,
            labels = target_ids_batch,
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