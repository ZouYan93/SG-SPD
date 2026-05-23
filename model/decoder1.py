import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def loss_setter(self, loss_type, norm_loss, aux_loss):
        if loss_type == 'norm':
            return norm_loss
        if loss_type == 'aux':
            return aux_loss
        if loss_type == 'combined':
            ce_weight = self.enhancement_config.get('ce_weight', 0.2)
            aux_weight = self.enhancement_config.get('aux_weight', 1.0)
            return ce_weight * norm_loss + aux_weight * aux_loss
        raise ValueError(f'Unsupported loss_type: {loss_type}')

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
        text_hidden, attention_mask = self._get_text_encoder_hidden_states(target_ids_batch)
        return masked_mean_pool(text_hidden, masks_batch=attention_mask)

    def _get_text_encoder_hidden_states(self, target_ids_batch):
        input_ids, attention_mask = self._labels_to_text_encoder_inputs(target_ids_batch)
        encoder = self.pretrained.get_encoder()
        was_training = encoder.training
        encoder.eval()
        with torch.no_grad():
            text_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        if was_training:
            encoder.train()
        return text_outputs.last_hidden_state.detach(), attention_mask

    def _build_decoder_input_ids(self, target_ids_batch):
        if hasattr(self.pretrained, 'prepare_decoder_input_ids_from_labels'):
            return self.pretrained.prepare_decoder_input_ids_from_labels(labels=target_ids_batch)

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        decoder_start_token_id = getattr(self.pretrained.config, 'decoder_start_token_id', None)
        if decoder_start_token_id is None:
            decoder_start_token_id = getattr(self.pretrained.config, 'bos_token_id', None)
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id

        input_ids = target_ids_batch.clone()
        input_ids[input_ids == -100] = pad_token_id
        decoder_input_ids = input_ids.new_full(input_ids.shape, pad_token_id)
        decoder_input_ids[:, 0] = decoder_start_token_id
        decoder_input_ids[:, 1:] = input_ids[:, :-1]
        return decoder_input_ids

    def _apply_decoder_input_dropout(self, decoder_input_ids):
        dropout_config = self.enhancement_config.get('decoder_input_dropout', {})
        rate = dropout_config.get('rate', 0.0)
        if not self.training or rate <= 0.0:
            return decoder_input_ids

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        mask_token_id = getattr(self.tokenizer, 'mask_token_id', None)
        unk_token_id = getattr(self.tokenizer, 'unk_token_id', None)
        replacement_id = mask_token_id
        if replacement_id is None:
            replacement_id = unk_token_id
        if replacement_id is None:
            replacement_id = pad_token_id

        valid_mask = decoder_input_ids != pad_token_id
        valid_mask[:, 0] = False
        random_mask = torch.rand(decoder_input_ids.shape, device=decoder_input_ids.device) < rate
        random_mask = random_mask & valid_mask

        corrupted_decoder_input_ids = decoder_input_ids.clone()
        corrupted_decoder_input_ids[random_mask] = replacement_id
        return corrupted_decoder_input_ids

    def _compute_alignment_loss(self, masks_batch, target_ids_batch, encoded_embedding):
        aux_loss = encoded_embedding.new_tensor(0.0)
        aux_loss_dict = {}
        if self.semantic_alignment_loss is None:
            return aux_loss, aux_loss_dict

        eeg_sentence_repr = masked_mean_pool(encoded_embedding, masks_batch=masks_batch)
        text_sentence_repr = self._get_text_sentence_repr(target_ids_batch).detach()
        alignment_config = self.enhancement_config.get('semantic_alignment', {})
        weight = alignment_config.get('weight', 0.1)
        align_loss = self.semantic_alignment_loss(eeg_sentence_repr, text_sentence_repr)
        aux_loss = aux_loss + weight * align_loss
        aux_loss_dict['semantic_alignment_loss'] = float(align_loss.detach().cpu())
        
        retrieval_metrics = self.semantic_alignment_loss.metrics(eeg_sentence_repr, text_sentence_repr)
        aux_loss_dict.update(retrieval_metrics)
        
        return aux_loss, aux_loss_dict

    def _build_negative_eeg_inputs(self, inputs_batch, masks_batch, noise_config):
        negative_type = noise_config.get('negative_type', 'shuffle')
        valid_mask = masks_batch.to(inputs_batch.device).float().unsqueeze(-1)

        if negative_type == 'gaussian' or inputs_batch.size(0) < 2:
            noise_std = noise_config.get('noise_std', 1.0)
            return torch.randn_like(inputs_batch) * noise_std * valid_mask, 'gaussian'

        if negative_type == 'shuffle':
            batch_size = inputs_batch.size(0)
            perm = torch.randperm(batch_size, device=inputs_batch.device)
            if torch.equal(perm, torch.arange(batch_size, device=inputs_batch.device)):
                perm = torch.roll(perm, shifts=1)
            return inputs_batch[perm] * valid_mask, 'shuffle'

        if negative_type == 'time_shuffle':
            seq_len = inputs_batch.size(1)
            perm = torch.randperm(seq_len, device=inputs_batch.device)
            return inputs_batch[:, perm, :] * valid_mask, 'time_shuffle'

        raise ValueError(f'Unsupported noise_contrast negative_type: {negative_type}')

    def _compute_noise_generation_loss(
            self,
            inputs_batch,
            masks_batch,
            invert_masks_batch,
            target_ids_batch,
            eeg_ce_loss,
            decoder_input_ids=None):
        aux_loss = eeg_ce_loss.new_tensor(0.0)
        aux_loss_dict = {}
        if self.noise_contrastive_loss is None:
            return aux_loss, aux_loss_dict

        noise_config = self.enhancement_config.get('noise_contrast', {})
        weight = noise_config.get('weight', 0.1)
        noise_std = noise_config.get('noise_std', 1.0)
        valid_mask = masks_batch.to(inputs_batch.device).float().unsqueeze(-1)
        noise_inputs, negative_type = self._build_negative_eeg_inputs(inputs_batch, masks_batch, noise_config)
        noise_encoder_outputs = self._encode_eeg(noise_inputs, invert_masks_batch)
        noise_embedding = self._project_to_pretrained_space(noise_encoder_outputs)
        noise_embedding, noise_masks_batch = self._append_t5_task_embedding(noise_embedding, masks_batch)
        noise_output = self.pretrained(
            inputs_embeds=noise_embedding,
            attention_mask=noise_masks_batch,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
            labels=target_ids_batch,
        )
        noise_ce_loss = noise_output.loss
        noise_loss = self.noise_contrastive_loss(eeg_ce_loss, noise_ce_loss)
        aux_loss = aux_loss + weight * noise_loss
        aux_loss_dict['noise_ce_loss'] = float(noise_ce_loss.detach().cpu())
        aux_loss_dict['noise_margin_loss'] = float(noise_loss.detach().cpu())
        aux_loss_dict['eeg_noise_ce_gap'] = float((eeg_ce_loss - noise_ce_loss).detach().cpu())
        aux_loss_dict[f'{negative_type}_ce_loss'] = float(noise_ce_loss.detach().cpu())
        aux_loss_dict[f'eeg_{negative_type}_ce_gap'] = float((eeg_ce_loss - noise_ce_loss).detach().cpu())
        return aux_loss, aux_loss_dict

    def _compute_encoder_distillation_loss(
            self,
            eeg_encoder_hidden,
            pretrained_masks_batch,
            source_masks_batch,
            target_ids_batch):
        distill_config = self.enhancement_config.get('hidden_distillation', {})
        aux_loss = eeg_encoder_hidden.new_tensor(0.0)
        aux_loss_dict = {}
        if not distill_config.get('enabled', False):
            return aux_loss, aux_loss_dict

        text_hidden, text_mask = self._get_text_encoder_hidden_states(target_ids_batch)
        text_hidden = text_hidden.to(eeg_encoder_hidden.device)
        text_mask = text_mask.to(eeg_encoder_hidden.device)

        eeg_hidden = eeg_encoder_hidden
        eeg_mask = pretrained_masks_batch.to(eeg_encoder_hidden.device)
        source_len = source_masks_batch.size(1)
        if eeg_hidden.size(1) > source_len:
            prefix_len = eeg_hidden.size(1) - source_len
            eeg_hidden = eeg_hidden[:, prefix_len:, :]
            eeg_mask = source_masks_batch.to(eeg_encoder_hidden.device)

        min_len = min(eeg_hidden.size(1), text_hidden.size(1))
        eeg_token_hidden = F.normalize(eeg_hidden[:, :min_len, :], dim=-1)
        text_token_hidden = F.normalize(text_hidden[:, :min_len, :], dim=-1)
        eeg_token_mask = eeg_mask[:, :min_len].float()
        text_token_mask = text_mask[:, :min_len].float()
        valid_token_mask = eeg_token_mask * text_token_mask

        token_cosine = (eeg_token_hidden * text_token_hidden).sum(dim=-1)
        token_loss = ((1.0 - token_cosine) * valid_token_mask).sum()
        token_loss = token_loss / valid_token_mask.sum().clamp_min(1.0)

        eeg_pooled = F.normalize(masked_mean_pool(eeg_hidden, masks_batch=eeg_mask), dim=-1)
        text_pooled = F.normalize(masked_mean_pool(text_hidden, masks_batch=text_mask), dim=-1)
        pooled_cosine_each = (eeg_pooled * text_pooled).sum(dim=-1)
        pooled_loss = (1.0 - pooled_cosine_each).mean()

        token_weight = distill_config.get('token_weight', 1.0)
        pooled_weight = distill_config.get('pooled_weight', 0.5)
        weight = distill_config.get('weight', 1.0)
        distill_loss = token_weight * token_loss + pooled_weight * pooled_loss
        aux_loss = aux_loss + weight * distill_loss

        token_cosine_mean = (token_cosine * valid_token_mask).sum() / valid_token_mask.sum().clamp_min(1.0)
        pooled_cosine = pooled_cosine_each.mean()
        aux_loss_dict['hidden_distill_loss'] = float(distill_loss.detach().cpu())
        aux_loss_dict['hidden_distill_token_loss'] = float(token_loss.detach().cpu())
        aux_loss_dict['hidden_distill_pooled_loss'] = float(pooled_loss.detach().cpu())
        aux_loss_dict['hidden_distill_token_cosine'] = float(token_cosine_mean.detach().cpu())
        aux_loss_dict['hidden_distill_pooled_cosine'] = float(pooled_cosine.detach().cpu())
        return aux_loss, aux_loss_dict
    
    def forward(self, inputs_batch, masks_batch, invert_masks_batch, target_ids_batch, loss_type='norm'):
        """
            input_batch: [batch_size, seq_len, feature_dim]
            masks_batch: [batch_size, seq_len] (解码器输入的注意力掩码, 0表示需要mask的位置)
            invert_masks_batch: [batch_size, seq_len] (编码器输入的注意力掩码, 1表示需要mask的位置)
            target_ids_batch: [batch_size, target_seq_len] (解码器目标序列的token ids)
        """
        # 获取编码器输出的特征
        encoder_outputs = self._encode_eeg(inputs_batch, invert_masks_batch)
        # 投影编码器输出到解码器输入维度
        encoded_embedding = self._project_to_pretrained_space(encoder_outputs)

        # 计算辅助对齐损失
        auxiliary_loss, aux_loss_dict = self._compute_alignment_loss(
            masks_batch,
            target_ids_batch,
            encoded_embedding,
        )

        encoded_embedding, pretrained_masks_batch = self._append_t5_task_embedding(encoded_embedding, masks_batch)
        decoder_input_ids = self._build_decoder_input_ids(target_ids_batch)
        decoder_input_ids = self._apply_decoder_input_dropout(decoder_input_ids)
        
        # 解码器生成输出
        output = self.pretrained(
            inputs_embeds=encoded_embedding,
            attention_mask=pretrained_masks_batch,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
            labels=target_ids_batch,
        )
        # 计算编码器蒸馏损失
        distill_auxiliary_loss, distill_aux_loss_dict = self._compute_encoder_distillation_loss(
            output.encoder_last_hidden_state,
            pretrained_masks_batch,
            masks_batch,
            target_ids_batch,
        )
        # 计算噪声对比损失
        noise_auxiliary_loss, noise_aux_loss_dict = self._compute_noise_generation_loss(
            inputs_batch,
            masks_batch,
            invert_masks_batch,
            target_ids_batch,
            output.loss,
            decoder_input_ids=decoder_input_ids,
        )

        auxiliary_loss = auxiliary_loss + distill_auxiliary_loss
        aux_loss_dict.update(distill_aux_loss_dict)
        auxiliary_loss = auxiliary_loss + noise_auxiliary_loss
        aux_loss_dict.update(noise_aux_loss_dict)
        aux_loss_dict['eeg_ce_loss'] = float(output.loss.detach().cpu())
        output.loss = self.loss_setter(loss_type, output.loss, auxiliary_loss)
        output.aux_loss_dict = aux_loss_dict
        return output
    
    @torch.no_grad()
    def generate(
            self,
            inputs_batch, 
            masks_batch, 
            invert_masks_batch, 
            target_ids_batch,
            generation_config=None,
            logits_processor=None,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=False,
            assistant_model=None,
            streamer=None,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
            **kwargs
    ):
        # 获取编码器输出的特征
        encoder_outputs = self._encode_eeg(inputs_batch, invert_masks_batch)
        # 投影编码器输出到解码器输入维度
        encoded_embedding = self._project_to_pretrained_space(encoder_outputs)
        encoded_embedding, masks_batch = self._append_t5_task_embedding(encoded_embedding, masks_batch)

        # 解码器生成输出
        output=self.pretrained.generate(
            inputs_embeds=encoded_embedding,
            attention_mask=masks_batch,
            labels=target_ids_batch,
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