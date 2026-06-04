import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from model.neuroguide import NeuroGuide
from utils.data.semkey_attributes import SENTIMENT_LABELS, TOPIC_LABELS


class Decoder(nn.Module):
    def __init__(self, pretrained_name, pretrained_model, encoder_model, tokenizer=None, enhancement_config=None):
        super(Decoder, self).__init__()
        self.pretrained_name = pretrained_name
        self.pretrained = pretrained_model
        self.encoder = encoder_model
        self.tokenizer = tokenizer
        self.enhancement_config = enhancement_config or {}
        self.feature_dim = self.encoder.output_dim
        self.decoder_embedding_size = self.pretrained.config.d_model

        # W_proj：将 EEG encoder 输出投影到预训练语言模型的 hidden size。
        self.fc1 = nn.Linear(self.feature_dim, self.decoder_embedding_size)

        self.neuro_guide = None
        self.length_regressor = None
        self.sentiment_classifier = None
        self.topic_classifier = None
        self.surprisal_regressor = None
        self.sentiment_labels = SENTIMENT_LABELS
        self.topic_labels = TOPIC_LABELS
        self.register_buffer('length_mean', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('length_std', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('surprisal_mean', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('surprisal_std', torch.tensor(1.0, dtype=torch.float32))
        neuro_config = self.enhancement_config.get('neuro_guide', {})
        if neuro_config.get('enabled', False):
            self.neuro_guide = NeuroGuide(
                hidden_dim=self.decoder_embedding_size,
                num_heads=neuro_config.get('num_heads', 8),
                dropout=neuro_config.get('dropout', 0.1),
                temperature=neuro_config.get('temperature', 0.07),
                commitment_loss=neuro_config.get('commitment_loss', 'mse'),
            )
            self.sentiment_classifier = self._build_attribute_head(self.decoder_embedding_size, len(self.sentiment_labels))
            self.topic_classifier = self._build_attribute_head(self.decoder_embedding_size, len(self.topic_labels))
            self.length_regressor = self._build_attribute_head(self.decoder_embedding_size, 1)
            self.surprisal_regressor = self._build_attribute_head(self.decoder_embedding_size, 1)

    def loss_setter(self, loss_type, norm_loss, aux_loss):
        if loss_type in ('neuro_stage1', 'neuro_stage2', 'norm'):
            return norm_loss
        if loss_type == 'aux':
            return aux_loss
        if loss_type == 'combined':
            ce_weight = self.enhancement_config.get('ce_weight', 1.0)
            aux_weight = self.enhancement_config.get('aux_weight', 1.0)
            return ce_weight * norm_loss + aux_weight * aux_loss
        raise ValueError(f'Unsupported loss_type: {loss_type}')

    def _encode_eeg(self, inputs_batch, invert_masks_batch, prior_ids_batch=None):
        return self.encoder(
            inputs_batch,
            invert_masks_batch=invert_masks_batch,
            prior_ids=prior_ids_batch,
        )

    def _project_to_pretrained_space(self, encoder_outputs):
        return self.fc1(encoder_outputs)

    def _build_encoder_attention_mask(self, encoded_embedding, masks_batch):
        # 普通 encoder 保留原始 padding mask；SemKeyStyle 这类 query-merger encoder
        # 输出的是固定长度 latent tokens，每个 latent token 都应当对 LLM 可见。
        batch_size, encoded_len = encoded_embedding.size(0), encoded_embedding.size(1)
        if masks_batch is None:
            return torch.ones(batch_size, encoded_len, device=encoded_embedding.device)
        if getattr(self.encoder, 'uses_learned_queries', False) or masks_batch.size(1) != encoded_len:
            return torch.ones(
                batch_size,
                encoded_len,
                dtype=masks_batch.dtype,
                device=encoded_embedding.device,
            )
        return masks_batch.to(encoded_embedding.device)

    def _build_attribute_head(self, hidden_dim, output_dim):
        neuro_config = self.enhancement_config.get('neuro_guide', {})
        hidden_dims = neuro_config.get('attribute_hidden_dims', [512, 256])
        dropout = neuro_config.get('attribute_dropout', 0.3)
        layers = []
        in_dim = hidden_dim
        for next_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = next_dim
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def _enabled_attribute_losses(self):
        neuro_config = self.enhancement_config.get('neuro_guide', {})
        enabled = neuro_config.get(
            'enabled_attribute_losses',
            ['sentiment', 'topic', 'length', 'surprisal'],
        )
        return set(enabled)

    def set_semkey_attribute_stats(self, length_mean, length_std, surprisal_mean, surprisal_std, **_):
        self.length_mean.data = torch.tensor(float(length_mean), dtype=torch.float32, device=self.length_mean.device)
        self.length_std.data = torch.tensor(max(float(length_std), 1e-6), dtype=torch.float32, device=self.length_std.device)
        self.surprisal_mean.data = torch.tensor(float(surprisal_mean), dtype=torch.float32, device=self.surprisal_mean.device)
        self.surprisal_std.data = torch.tensor(max(float(surprisal_std), 1e-6), dtype=torch.float32, device=self.surprisal_std.device)

    def _append_t5_task_embedding(self, encoded_embedding, masks_batch):
        # baseline 路径保留原框架的 T5 task prefix，方便做对照消融。
        if self.pretrained_name != 'T5' and self.pretrained_name != 'Flan-T5':
            return encoded_embedding, masks_batch
        input_ids = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(encoded_embedding.device)
        task_embedding = self.pretrained.shared(input_ids).to(encoded_embedding.device)
        task_embedding = task_embedding.repeat(encoded_embedding.size(0), 1, 1)
        encoded_embedding = torch.cat((task_embedding, encoded_embedding), dim=1)
        task_mask = torch.ones(
            encoded_embedding.size(0),
            task_embedding.size(1),
            dtype=masks_batch.dtype,
            device=encoded_embedding.device,
        )
        masks_batch = torch.cat((task_mask, masks_batch), dim=1)
        return encoded_embedding, masks_batch

    def _labels_to_text_encoder_inputs(self, target_ids_batch):
        input_ids = target_ids_batch.clone()
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        input_ids[input_ids == -100] = pad_token_id
        attention_mask = (input_ids != pad_token_id).long()
        return input_ids, attention_mask

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

    def _build_prompt_texts(
        self,
        sentiment_ids=None,
        topic_ids=None,
        length_values=None,
        surprisal_values=None,
        batch_size=None,
    ):
        # 原论文 Stage2 的 Query 来自带属性约束的 X_prompt。
        base_prompt = self.enhancement_config.get('neuro_guide', {}).get(
            'prompt_text',
            'System: Based on the following EEG signals, reconstruct the text.'
        )
        if batch_size is None:
            if sentiment_ids is not None:
                batch_size = sentiment_ids.size(0)
            elif topic_ids is not None:
                batch_size = topic_ids.size(0)
            elif length_values is not None:
                batch_size = length_values.size(0)
            elif surprisal_values is not None:
                batch_size = surprisal_values.size(0)
            else:
                batch_size = 1

        prompts = []
        neuro_config = self.enhancement_config.get('neuro_guide', {})
        prompt_attributes = neuro_config.get(
            'prompt_attributes',
            ['sentiment', 'topic', 'length'],
        )
        prompt_attributes = set(prompt_attributes)
        for i in range(batch_size):
            parts = [base_prompt]
            if length_values is not None and 'length' in prompt_attributes:
                length_value = float(length_values[i].detach().cpu().item())
                parts.append(f'The length of the sentence is {length_value:.1f} words.')
            if surprisal_values is not None and 'surprisal' in prompt_attributes:
                surprisal_value = float(surprisal_values[i].detach().cpu().item())
                parts.append(f'The average surprisal value is {surprisal_value:.2f}.')
            if sentiment_ids is not None and 'sentiment' in prompt_attributes:
                sent_idx = int(sentiment_ids[i].detach().cpu().item())
                sent_idx = max(0, min(sent_idx, len(self.sentiment_labels) - 1))
                parts.append(f'Sentiment: {self.sentiment_labels[sent_idx]}.')
            if topic_ids is not None and 'topic' in prompt_attributes:
                topic_idx = int(topic_ids[i].detach().cpu().item())
                topic_idx = max(0, min(topic_idx, len(self.topic_labels) - 1))
                parts.append(f'Topic: {self.topic_labels[topic_idx]}.')
            parts.append('Target:')
            prompts.append(' '.join(parts))
        return prompts

    def _build_prompt_ids(
        self,
        batch_size,
        device,
        sentiment_ids=None,
        topic_ids=None,
        length_values=None,
        surprisal_values=None,
    ):
        prompts = self._build_prompt_texts(
            sentiment_ids=sentiment_ids,
            topic_ids=topic_ids,
            length_values=length_values,
            surprisal_values=surprisal_values,
            batch_size=batch_size,
        )
        prompt = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        prompt_ids = prompt['input_ids'].to(device)
        prompt_mask = prompt['attention_mask'].to(device)

        # 去掉 prompt 末尾 EOS，避免训练时 prompt 自己提前结束。
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        if eos_token_id is None:
            return prompt_ids, prompt_mask

        keep = prompt_ids != eos_token_id
        lengths = keep.long().sum(dim=1).clamp_min(1)
        max_len = lengths.max().item()
        trimmed = prompt_ids.new_full((batch_size, max_len), self.tokenizer.pad_token_id)
        trimmed_mask = prompt_mask.new_zeros((batch_size, max_len))
        for i in range(batch_size):
            row = prompt_ids[i][keep[i]]
            trimmed[i, :row.numel()] = row
            trimmed_mask[i, :row.numel()] = 1
        return trimmed, trimmed_mask

    def _predict_attributes(self, eeg_vector):
        sentiment_logits = self.sentiment_classifier(eeg_vector)
        topic_logits = self.topic_classifier(eeg_vector)
        length_pred = self.length_regressor(eeg_vector).squeeze(-1)
        surprisal_pred = self.surprisal_regressor(eeg_vector).squeeze(-1)
        return sentiment_logits, topic_logits, length_pred, surprisal_pred

    def _denormalize_regression(self, values, task):
        if task == 'length':
            return values * self.length_std.to(values.device) + self.length_mean.to(values.device)
        if task == 'surprisal':
            return values * self.surprisal_std.to(values.device) + self.surprisal_mean.to(values.device)
        raise ValueError(f'Unsupported regression task: {task}')

    def _resolve_prompt_attributes(
        self,
        states,
        sentiment_labels_batch=None,
        topic_labels_batch=None,
        length_values_batch=None,
        surprisal_values_batch=None,
        use_targets=True,
    ):
        sentiment_logits, topic_logits, length_pred, surprisal_pred = self._predict_attributes(states['eeg_vector'])
        pred_sentiment_ids = sentiment_logits.argmax(dim=-1)
        pred_topic_ids = topic_logits.argmax(dim=-1)
        pred_length_values = self._denormalize_regression(length_pred, 'length')
        pred_surprisal_values = self._denormalize_regression(surprisal_pred, 'surprisal')

        if not use_targets:
            return pred_sentiment_ids, pred_topic_ids, pred_length_values, pred_surprisal_values

        device = states['eeg_vector'].device
        sentiment_ids = pred_sentiment_ids
        if sentiment_labels_batch is not None:
            target_sentiment = sentiment_labels_batch.to(device).long()
            valid = target_sentiment.ge(0)
            sentiment_ids = torch.where(valid, target_sentiment, sentiment_ids)

        topic_ids = pred_topic_ids
        if topic_labels_batch is not None:
            target_topic = topic_labels_batch.to(device).long()
            valid = target_topic.ge(0)
            topic_ids = torch.where(valid, target_topic, topic_ids)

        length_values = pred_length_values
        if length_values_batch is not None:
            target_length = length_values_batch.to(device).float()
            valid = torch.isfinite(target_length)
            length_values = torch.where(valid, target_length, length_values)

        surprisal_values = pred_surprisal_values
        if surprisal_values_batch is not None:
            target_surprisal = surprisal_values_batch.to(device).float()
            valid = torch.isfinite(target_surprisal)
            surprisal_values = torch.where(valid, target_surprisal, surprisal_values)

        return sentiment_ids, topic_ids, length_values, surprisal_values

    def _build_prompted_decoder_inputs(
        self,
        target_ids_batch,
        states,
        sentiment_labels_batch=None,
        topic_labels_batch=None,
        length_values_batch=None,
        surprisal_values_batch=None,
    ):
        batch_size = target_ids_batch.size(0)
        neuro_config = self.enhancement_config.get('neuro_guide', {})
        use_targets = bool(neuro_config.get('use_ground_truth_attributes_in_stage2_train', False))
        sentiment_ids, topic_ids, length_values, surprisal_values = self._resolve_prompt_attributes(
            states,
            sentiment_labels_batch=sentiment_labels_batch,
            topic_labels_batch=topic_labels_batch,
            length_values_batch=length_values_batch,
            surprisal_values_batch=surprisal_values_batch,
            use_targets=use_targets,
        )
        prompt_ids, _ = self._build_prompt_ids(
            batch_size,
            target_ids_batch.device,
            sentiment_ids=sentiment_ids,
            topic_ids=topic_ids,
            length_values=length_values,
            surprisal_values=surprisal_values,
        )
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0

        target_ids = target_ids_batch.clone()
        target_ids[target_ids == -100] = pad_token_id
        merged = torch.cat([prompt_ids, target_ids], dim=1)

        labels = torch.cat([
            target_ids_batch.new_full(prompt_ids.shape, -100),
            target_ids_batch,
        ], dim=1)

        decoder_input_ids = merged.new_full(merged.shape, pad_token_id)
        decoder_start_token_id = getattr(self.pretrained.config, 'decoder_start_token_id', None)
        if decoder_start_token_id is None:
            decoder_start_token_id = getattr(self.pretrained.config, 'bos_token_id', None)
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id
        decoder_input_ids[:, 0] = decoder_start_token_id
        decoder_input_ids[:, 1:] = merged[:, :-1]
        return decoder_input_ids, labels

    def _get_decoder_start_token_id(self):
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        decoder_start_token_id = getattr(self.pretrained.config, 'decoder_start_token_id', None)
        if decoder_start_token_id is None:
            decoder_start_token_id = getattr(self.pretrained.config, 'bos_token_id', None)
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id
        return decoder_start_token_id

    def _build_generation_decoder_prefix(
        self,
        states,
        sentiment_labels_batch=None,
        topic_labels_batch=None,
        length_values_batch=None,
        surprisal_values_batch=None,
    ):
        # 生成时同样把 X_prompt 作为 decoder prefix 参与 self-attention 形成 Q。
        batch_size = states['kv_embeds'].size(0)
        device = states['kv_embeds'].device
        use_targets = self.enhancement_config.get('neuro_guide', {}).get(
            'use_ground_truth_attributes_in_eval',
            False,
        )
        sentiment_ids, topic_ids, length_values, surprisal_values = self._resolve_prompt_attributes(
            states,
            sentiment_labels_batch=sentiment_labels_batch,
            topic_labels_batch=topic_labels_batch,
            length_values_batch=length_values_batch,
            surprisal_values_batch=surprisal_values_batch,
            use_targets=use_targets,
        )
        prompt_ids, _ = self._build_prompt_ids(
            batch_size,
            device,
            sentiment_ids=sentiment_ids,
            topic_ids=topic_ids,
            length_values=length_values,
            surprisal_values=surprisal_values,
        )
        start_ids = prompt_ids.new_full((batch_size, 1), self._get_decoder_start_token_id())
        return torch.cat([start_ids, prompt_ids], dim=1)

    def _build_neuroguide_states(self, inputs_batch, masks_batch, invert_masks_batch, prior_ids_batch=None):
        # 1. EEG encoder 得到原始 EEG latent tokens，记作 E_raw。
        eeg_hidden = self._encode_eeg(
            inputs_batch,
            invert_masks_batch,
            prior_ids_batch=prior_ids_batch,
        )

        # 2. W_proj 后得到 E_eeg，并用 learnable query attention pooling 得到句级向量 v。
        eeg_embeds = self._project_to_pretrained_space(eeg_hidden)
        eeg_mask = torch.ones(
            eeg_embeds.size(0),
            eeg_embeds.size(1),
            dtype=masks_batch.dtype,
            device=masks_batch.device,
        )
        eeg_vector = self.neuro_guide.pool_eeg(eeg_embeds, eeg_mask)

        # 3. Stage2 K/V 来源：[v; E_eeg]。后续 W_K/W_V 由 LLM cross-attention 内部完成。
        kv_embeds = torch.cat([eeg_vector.unsqueeze(1), eeg_embeds], dim=1)
        global_mask = torch.ones(
            masks_batch.size(0),
            1,
            dtype=masks_batch.dtype,
            device=masks_batch.device,
        )
        kv_mask = torch.cat([global_mask, eeg_mask], dim=1)
        return {
            'eeg_hidden': eeg_hidden,
            'eeg_embeds': eeg_embeds,
            'eeg_mask': eeg_mask,
            'eeg_vector': eeg_vector,
            'kv_embeds': kv_embeds,
            'kv_mask': kv_mask,
        }

    def _compute_stage1_alignment(self, states, masks_batch, target_ids_batch):
        text_hidden, text_mask = self._get_text_encoder_hidden_states(target_ids_batch)
        text_hidden = text_hidden.to(states['eeg_embeds'].device)
        text_mask = text_mask.to(states['eeg_embeds'].device)
        text_vector = self.neuro_guide.pool_text(text_hidden, text_mask)

        clip_loss, clip_metrics = self.neuro_guide.contrastive_loss(states['eeg_vector'], text_vector)
        commit_loss, commit_metrics = self.neuro_guide.commitment_alignment(
            states['eeg_embeds'],
            text_hidden,
            eeg_mask=states['eeg_mask'],
            text_mask=text_mask,
        )
        return clip_loss, clip_metrics, commit_loss, commit_metrics

    def _make_noise_inputs(self, inputs_batch, masks_batch, invert_masks_batch=None, prior_ids_batch=None):
        grounding_config = self.enhancement_config.get('neuro_guide', {}).get('signal_grounding', {})
        noise_types = grounding_config.get('noise_types', ['random'])
        if isinstance(noise_types, str):
            noise_types = [noise_types]
        if not noise_types:
            noise_types = ['random']

        noise_items = []
        mask = None
        if masks_batch is not None:
            mask = masks_batch.to(inputs_batch.device).float().unsqueeze(-1)

        for noise_type in noise_types:
            if noise_type == 'random':
                noise = torch.randn_like(inputs_batch)
                noise_masks = torch.ones_like(masks_batch) if masks_batch is not None else None
                noise_invert_masks = torch.zeros_like(invert_masks_batch) if invert_masks_batch is not None else None
                noise_items.append(('random', noise, noise_masks, noise_invert_masks, prior_ids_batch))
            elif noise_type == 'shuffle':
                if inputs_batch.size(0) > 1:
                    perm = torch.randperm(inputs_batch.size(0), device=inputs_batch.device)
                    if torch.equal(perm, torch.arange(inputs_batch.size(0), device=inputs_batch.device)):
                        perm = torch.roll(perm, shifts=1, dims=0)
                    noise = inputs_batch[perm].detach()
                    noise_masks = None if masks_batch is None else masks_batch[perm]
                    noise_invert_masks = None if invert_masks_batch is None else invert_masks_batch[perm]
                    noise_prior_ids = None if prior_ids_batch is None else prior_ids_batch[perm]
                else:
                    noise = torch.randn_like(inputs_batch)
                    if mask is not None:
                        noise = noise * mask
                    noise_masks = masks_batch
                    noise_invert_masks = invert_masks_batch
                    noise_prior_ids = prior_ids_batch
                noise_items.append(('shuffle', noise, noise_masks, noise_invert_masks, noise_prior_ids))
            else:
                raise ValueError(f'Unsupported signal grounding noise_type: {noise_type}')
        return noise_items

    def _attribute_uniform_loss(self, states):
        sentiment_logits, topic_logits, _, _ = self._predict_attributes(states['eeg_vector'])

        def kl_to_uniform(logits):
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            return (probs * (log_probs + torch.log(torch.tensor(logits.size(-1), device=logits.device, dtype=logits.dtype)))).sum(dim=-1).mean()

        return kl_to_uniform(sentiment_logits) + kl_to_uniform(topic_logits)

    def _compute_signal_grounding_loss(
        self,
        states,
        inputs_batch,
        masks_batch,
        invert_masks_batch,
        target_ids_batch,
        prior_ids_batch=None,
        eeg_recon_loss=None,
        decoder_input_ids=None,
        labels=None,
        stage='stage1',
    ):
        neuro_config = self.enhancement_config.get('neuro_guide', {})
        grounding_config = neuro_config.get('signal_grounding', {})
        if not grounding_config.get('enabled', False):
            return states['eeg_vector'].new_tensor(0.0), {}

        total = states['eeg_vector'].new_tensor(0.0)
        metrics = {}

        text_hidden, text_mask = self._get_text_encoder_hidden_states(target_ids_batch)
        text_hidden = text_hidden.to(states['eeg_embeds'].device)
        text_mask = text_mask.to(states['eeg_embeds'].device)
        text_vector = self.neuro_guide.pool_text(text_hidden, text_mask)
        eeg_sim = nn.functional.cosine_similarity(states['eeg_vector'], text_vector, dim=-1)
        noise_items = self._make_noise_inputs(
            inputs_batch,
            masks_batch,
            invert_masks_batch=invert_masks_batch,
            prior_ids_batch=prior_ids_batch,
        )
        align_losses = []
        noise_sims = []
        uniform_losses = []
        recon_margin_losses = []
        noise_recon_losses = []

        align_margin = float(grounding_config.get('align_margin', 0.15))
        align_weight = float(grounding_config.get('align_weight', 1.0))
        entropy_weight = float(grounding_config.get('noise_attribute_uniform_weight', 0.0))
        recon_weight = float(grounding_config.get('recon_weight', 0.0))
        separation_weight = float(grounding_config.get('eeg_noise_separation_weight', 0.0))
        max_eeg_noise_cosine = float(grounding_config.get('max_eeg_noise_cosine', 0.1))
        eeg_noise_sims = []
        separation_losses = []

        recon_noise_names = grounding_config.get('recon_noise_types', None)
        if isinstance(recon_noise_names, str):
            recon_noise_names = [recon_noise_names]
        if recon_noise_names is not None:
            recon_noise_names = set(recon_noise_names)
        elif noise_items:
            recon_noise_names = {noise_items[0][0]}
        else:
            recon_noise_names = set()

        for noise_name, noise_inputs, noise_masks, noise_invert_masks, noise_prior_ids in noise_items:
            noise_states = self._build_neuroguide_states(
                noise_inputs,
                noise_masks,
                noise_invert_masks,
                prior_ids_batch=noise_prior_ids,
            )
            noise_sim = nn.functional.cosine_similarity(noise_states['eeg_vector'], text_vector, dim=-1)
            align_margin_loss = nn.functional.relu(align_margin - eeg_sim + noise_sim).mean()
            total = total + align_weight * align_margin_loss
            align_losses.append(align_margin_loss)
            noise_sims.append(noise_sim.mean())
            metrics[f'signal_grounding_{noise_name}_noise_text_cosine'] = float(noise_sim.mean().detach().cpu())

            if separation_weight > 0:
                eeg_noise_sim = nn.functional.cosine_similarity(
                    states['eeg_vector'],
                    noise_states['eeg_vector'],
                    dim=-1,
                )
                separation_loss = nn.functional.relu(eeg_noise_sim - max_eeg_noise_cosine).mean()
                total = total + separation_weight * separation_loss
                eeg_noise_sims.append(eeg_noise_sim.mean())
                separation_losses.append(separation_loss)
                metrics[f'signal_grounding_{noise_name}_eeg_noise_cosine'] = float(eeg_noise_sim.mean().detach().cpu())

            if entropy_weight > 0:
                uniform_loss = self._attribute_uniform_loss(noise_states)
                total = total + entropy_weight * uniform_loss
                uniform_losses.append(uniform_loss)

            if (
                recon_weight > 0
                and noise_name in recon_noise_names
                and eeg_recon_loss is not None
                and decoder_input_ids is not None
                and labels is not None
            ):
                if stage == 'stage2':
                    noise_output = self.pretrained(
                        encoder_outputs=BaseModelOutput(last_hidden_state=noise_states['kv_embeds']),
                        attention_mask=noise_states['kv_mask'],
                        decoder_input_ids=decoder_input_ids,
                        return_dict=True,
                        labels=labels,
                    )
                else:
                    noise_output = self.pretrained(
                        encoder_outputs=BaseModelOutput(last_hidden_state=noise_states['eeg_embeds']),
                        attention_mask=noise_states['eeg_mask'],
                        decoder_input_ids=decoder_input_ids,
                        return_dict=True,
                        labels=labels,
                    )
                recon_margin = float(grounding_config.get('recon_margin', 0.25))
                recon_margin_loss = nn.functional.relu(recon_margin + eeg_recon_loss - noise_output.loss)
                total = total + recon_weight * recon_margin_loss
                recon_margin_losses.append(recon_margin_loss)
                noise_recon_losses.append(noise_output.loss)
                metrics[f'signal_grounding_{noise_name}_noise_recon_loss'] = float(noise_output.loss.detach().cpu())

        divisor = float(len(noise_items))
        total = total / divisor
        metrics['signal_grounding_align_margin_loss'] = float(torch.stack(align_losses).mean().detach().cpu())
        metrics['signal_grounding_eeg_text_cosine'] = float(eeg_sim.mean().detach().cpu())
        metrics['signal_grounding_noise_text_cosine'] = float(torch.stack(noise_sims).mean().detach().cpu())
        if eeg_noise_sims:
            metrics['signal_grounding_eeg_noise_cosine'] = float(torch.stack(eeg_noise_sims).mean().detach().cpu())
            metrics['signal_grounding_eeg_noise_separation_loss'] = float(torch.stack(separation_losses).mean().detach().cpu())
        if uniform_losses:
            metrics['signal_grounding_noise_attribute_uniform_loss'] = float(torch.stack(uniform_losses).mean().detach().cpu())
        if recon_margin_losses:
            metrics['signal_grounding_recon_margin_loss'] = float(torch.stack(recon_margin_losses).mean().detach().cpu())
            metrics['signal_grounding_noise_recon_loss'] = float(torch.stack(noise_recon_losses).mean().detach().cpu())

        return total, metrics

    def _compute_attribute_losses(
        self,
        states,
        sentiment_labels_batch=None,
        topic_labels_batch=None,
        length_values_batch=None,
        surprisal_values_batch=None,
    ):
        sentiment_logits, topic_logits, length_pred, surprisal_pred = self._predict_attributes(states['eeg_vector'])
        device = states['eeg_vector'].device
        neuro_config = self.enhancement_config.get('neuro_guide', {})
        enabled_losses = self._enabled_attribute_losses()

        total = states['eeg_vector'].new_tensor(0.0)
        metrics = {}

        if 'sentiment' in enabled_losses and sentiment_labels_batch is not None:
            sentiment_targets = sentiment_labels_batch.to(device).long()
            valid = sentiment_targets.ge(0)
            if valid.any():
                sentiment_loss = nn.functional.cross_entropy(sentiment_logits[valid], sentiment_targets[valid])
                total = total + neuro_config.get('sentiment_loss_weight', 0.25) * sentiment_loss
                with torch.no_grad():
                    sentiment_acc = (sentiment_logits[valid].argmax(dim=-1) == sentiment_targets[valid]).float().mean()
                metrics['neuroguide_sentiment_loss'] = float(sentiment_loss.detach().cpu())
                metrics['neuroguide_sentiment_acc'] = float(sentiment_acc.detach().cpu())

        if 'topic' in enabled_losses and topic_labels_batch is not None:
            topic_targets = topic_labels_batch.to(device).long()
            valid = topic_targets.ge(0)
            if valid.any():
                topic_loss = nn.functional.cross_entropy(topic_logits[valid], topic_targets[valid])
                total = total + neuro_config.get('topic_loss_weight', 0.25) * topic_loss
                with torch.no_grad():
                    topic_acc = (topic_logits[valid].argmax(dim=-1) == topic_targets[valid]).float().mean()
                metrics['neuroguide_topic_loss'] = float(topic_loss.detach().cpu())
                metrics['neuroguide_topic_acc'] = float(topic_acc.detach().cpu())

        if 'length' in enabled_losses and length_values_batch is not None:
            length_targets = length_values_batch.to(device).float()
            valid = torch.isfinite(length_targets)
            if valid.any():
                length_norm = (length_targets[valid] - self.length_mean.to(device)) / self.length_std.to(device)
                length_loss = nn.functional.mse_loss(length_pred[valid], length_norm)
                total = total + neuro_config.get('length_loss_weight', 0.25) * length_loss
                with torch.no_grad():
                    length_denorm = self._denormalize_regression(length_pred[valid], 'length')
                    length_mae = nn.functional.l1_loss(length_denorm, length_targets[valid])
                metrics['neuroguide_length_loss'] = float(length_loss.detach().cpu())
                metrics['neuroguide_length_mae'] = float(length_mae.detach().cpu())

        if 'surprisal' in enabled_losses and surprisal_values_batch is not None:
            surprisal_targets = surprisal_values_batch.to(device).float()
            valid = torch.isfinite(surprisal_targets)
            if valid.any():
                surprisal_norm = (
                    (surprisal_targets[valid] - self.surprisal_mean.to(device))
                    / self.surprisal_std.to(device)
                )
                surprisal_loss = nn.functional.mse_loss(surprisal_pred[valid], surprisal_norm)
                total = total + neuro_config.get('surprisal_loss_weight', 0.25) * surprisal_loss
                with torch.no_grad():
                    surprisal_denorm = self._denormalize_regression(surprisal_pred[valid], 'surprisal')
                    surprisal_mae = nn.functional.l1_loss(surprisal_denorm, surprisal_targets[valid])
                metrics['neuroguide_surprisal_loss'] = float(surprisal_loss.detach().cpu())
                metrics['neuroguide_surprisal_mae'] = float(surprisal_mae.detach().cpu())

        return total, metrics

    def _forward_neuroguide_stage1(
        self,
        inputs_batch,
        masks_batch,
        invert_masks_batch,
        target_ids_batch,
        prior_ids_batch=None,
        seq_len_batch=None,
        sentiment_labels_batch=None,
        topic_labels_batch=None,
        length_values_batch=None,
        surprisal_values_batch=None,
        alignment_target_ids_batch=None,
    ):
        states = self._build_neuroguide_states(
            inputs_batch,
            masks_batch,
            invert_masks_batch,
            prior_ids_batch=prior_ids_batch,
        )
        decoder_input_ids = self._build_decoder_input_ids(target_ids_batch)

        # Stage1 重建：E_eeg 作为 LLM encoder memory，训练 EEG 表征支持文本重建。
        recon_output = self.pretrained(
            encoder_outputs=BaseModelOutput(last_hidden_state=states['eeg_embeds']),
            attention_mask=states['eeg_mask'],
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
            labels=target_ids_batch,
        )
        recon_loss = recon_output.loss
        if alignment_target_ids_batch is None:
            alignment_target_ids_batch = target_ids_batch
        clip_loss, clip_metrics, commit_loss, commit_metrics = self._compute_stage1_alignment(
            states,
            masks_batch,
            alignment_target_ids_batch,
        )
        attribute_loss, attribute_metrics = self._compute_attribute_losses(
            states,
            sentiment_labels_batch=sentiment_labels_batch,
            topic_labels_batch=topic_labels_batch,
            length_values_batch=length_values_batch,
            surprisal_values_batch=surprisal_values_batch,
        )
        neuro_config = self.enhancement_config.get('neuro_guide', {})
        stage1_grounding_weight = neuro_config.get('stage1_signal_grounding_weight', 0.0)
        if stage1_grounding_weight > 0:
            grounding_loss, grounding_metrics = self._compute_signal_grounding_loss(
                states,
                inputs_batch,
                masks_batch,
                invert_masks_batch,
                alignment_target_ids_batch,
                prior_ids_batch=prior_ids_batch,
                eeg_recon_loss=recon_loss,
                decoder_input_ids=decoder_input_ids,
                labels=target_ids_batch,
                stage='stage1',
            )
        else:
            grounding_loss = recon_loss.new_tensor(0.0)
            grounding_metrics = {}
        total_loss = (
            neuro_config.get('stage1_recon_weight', 1.0) * recon_loss
            + neuro_config.get('stage1_clip_weight', 0.5) * clip_loss
            + neuro_config.get('stage1_commitment_weight', 0.7) * commit_loss
            + neuro_config.get('stage1_attribute_weight', 0.5) * attribute_loss
            + stage1_grounding_weight * grounding_loss
        )
        recon_output.loss = total_loss
        recon_output.aux_loss_dict = {
            'neuroguide_stage1_total_loss': float(total_loss.detach().cpu()),
            'neuroguide_recon_ce_loss': float(recon_loss.detach().cpu()),
            'neuroguide_attribute_loss': float(attribute_loss.detach().cpu()),
            'signal_grounding_loss': float(grounding_loss.detach().cpu()),
            **clip_metrics,
            **commit_metrics,
            **attribute_metrics,
            **grounding_metrics,
        }
        return recon_output

    def _forward_neuroguide_stage2(
        self,
        inputs_batch,
        masks_batch,
        invert_masks_batch,
        target_ids_batch,
        prior_ids_batch=None,
        sentiment_labels_batch=None,
        topic_labels_batch=None,
        length_values_batch=None,
        surprisal_values_batch=None,
    ):
        states = self._build_neuroguide_states(
            inputs_batch,
            masks_batch,
            invert_masks_batch,
            prior_ids_batch=prior_ids_batch,
        )
        decoder_input_ids, labels = self._build_prompted_decoder_inputs(
            target_ids_batch,
            states,
            sentiment_labels_batch=sentiment_labels_batch,
            topic_labels_batch=topic_labels_batch,
            length_values_batch=length_values_batch,
            surprisal_values_batch=surprisal_values_batch,
        )

        # Stage2 注入：decoder self-attention 处理 [X_prompt; y_<t>] 形成 Q，
        # cross-attention 查询 [v; E_eeg] 形成的 K/V。
        output = self.pretrained(
            encoder_outputs=BaseModelOutput(last_hidden_state=states['kv_embeds']),
            attention_mask=states['kv_mask'],
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
            labels=labels,
        )
        recon_loss = output.loss
        neuro_config = self.enhancement_config.get('neuro_guide', {})
        stage2_attribute_weight = neuro_config.get('stage2_attribute_weight', 0.0)
        use_stage2_attribute_loss = bool(neuro_config.get('use_stage2_attribute_loss', False))
        attribute_metrics = {}
        if use_stage2_attribute_loss and stage2_attribute_weight > 0:
            attribute_loss, attribute_metrics = self._compute_attribute_losses(
                states,
                sentiment_labels_batch=sentiment_labels_batch,
                topic_labels_batch=topic_labels_batch,
                length_values_batch=length_values_batch,
                surprisal_values_batch=surprisal_values_batch,
            )
            output.loss = recon_loss + stage2_attribute_weight * attribute_loss
            attribute_metrics = {
                'neuroguide_stage2_attribute_loss': float(attribute_loss.detach().cpu()),
                **attribute_metrics,
            }
        else:
            output.loss = recon_loss
        stage2_grounding_weight = neuro_config.get('stage2_signal_grounding_weight', 0.0)
        if stage2_grounding_weight > 0:
            grounding_loss, grounding_metrics = self._compute_signal_grounding_loss(
                states,
                inputs_batch,
                masks_batch,
                invert_masks_batch,
                target_ids_batch,
                prior_ids_batch=prior_ids_batch,
                eeg_recon_loss=recon_loss,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                stage='stage2',
            )
            output.loss = output.loss + stage2_grounding_weight * grounding_loss
        else:
            grounding_loss = recon_loss.new_tensor(0.0)
            grounding_metrics = {}
        output.prompt_length = decoder_input_ids.size(1) - target_ids_batch.size(1)
        output.aux_loss_dict = {
            'neuroguide_stage2_total_loss': float(output.loss.detach().cpu()),
            'neuroguide_stage2_recon_loss': float(recon_loss.detach().cpu()),
            'signal_grounding_loss': float(grounding_loss.detach().cpu()),
            **attribute_metrics,
            **grounding_metrics,
        }
        return output

    def forward(
        self,
        inputs_batch,
        masks_batch,
        invert_masks_batch,
        target_ids_batch,
        loss_type='combined',
        prior_ids_batch=None,
        seq_len_batch=None,
        sentiment_labels_batch=None,
        topic_labels_batch=None,
        length_values_batch=None,
        surprisal_values_batch=None,
        alignment_target_ids_batch=None,
    ):
        if self.neuro_guide is not None:
            if loss_type == 'neuro_stage1':
                return self._forward_neuroguide_stage1(
                    inputs_batch,
                    masks_batch,
                    invert_masks_batch,
                    target_ids_batch,
                    prior_ids_batch=prior_ids_batch,
                    seq_len_batch=seq_len_batch,
                    sentiment_labels_batch=sentiment_labels_batch,
                    topic_labels_batch=topic_labels_batch,
                    length_values_batch=length_values_batch,
                    surprisal_values_batch=surprisal_values_batch,
                    alignment_target_ids_batch=alignment_target_ids_batch,
                )

            # 评估脚本通常不传 loss_type。对于 NeuroGuide，默认 teacher-forcing
            # 应该走 Stage2 注入路径，而不是原框架 baseline 路径。
            if loss_type in ('combined', 'norm', 'neuro_stage2'):
                return self._forward_neuroguide_stage2(
                    inputs_batch,
                    masks_batch,
                    invert_masks_batch,
                    target_ids_batch,
                    prior_ids_batch=prior_ids_batch,
                    sentiment_labels_batch=sentiment_labels_batch,
                    topic_labels_batch=topic_labels_batch,
                    length_values_batch=length_values_batch,
                    surprisal_values_batch=surprisal_values_batch,
                )

            raise ValueError(f'Unsupported NeuroGuide loss_type: {loss_type}')

        if loss_type in ('neuro_stage1', 'neuro_stage2'):
            raise ValueError('NeuroGuide loss_type requires enhancement.neuro_guide.enabled = true')

        auxiliary_loss = inputs_batch.new_tensor(0.0)
        encoder_outputs = self._encode_eeg(
                inputs_batch,
                invert_masks_batch,
                prior_ids_batch=prior_ids_batch,
        )
        encoded_embedding = self._project_to_pretrained_space(encoder_outputs)
        decoder_input_ids = self._build_decoder_input_ids(target_ids_batch)
        pretrained_masks_batch = self._build_encoder_attention_mask(encoded_embedding, masks_batch)
        encoded_embedding, pretrained_masks_batch = self._append_t5_task_embedding(encoded_embedding, pretrained_masks_batch)
        output = self.pretrained(
            inputs_embeds=encoded_embedding,
            attention_mask=pretrained_masks_batch,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
            labels=target_ids_batch,
        )
        output.aux_loss_dict = {'eeg_ce_loss': float(output.loss.detach().cpu())}
        output.loss = self.loss_setter(loss_type, output.loss, auxiliary_loss)
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
            prior_ids_batch=None,
            sentiment_labels_batch=None,
            topic_labels_batch=None,
            length_values_batch=None,
            surprisal_values_batch=None,
            **kwargs
    ):
        if self.neuro_guide is not None:
            states = self._build_neuroguide_states(
                inputs_batch,
                masks_batch,
                invert_masks_batch,
                prior_ids_batch=prior_ids_batch,
            )
            decoder_input_ids = self._build_generation_decoder_prefix(
                states,
                sentiment_labels_batch=sentiment_labels_batch,
                topic_labels_batch=topic_labels_batch,
                length_values_batch=length_values_batch,
                surprisal_values_batch=surprisal_values_batch,
            )
            prefix_len = decoder_input_ids.size(1)

            # 调用方传 max_length 时通常指“希望生成的文本长度”，但 HF 会把 prefix
            # 也计入 max_length。这里自动补上 prefix 长度，避免有效生成空间被吃掉。
            if 'max_length' in kwargs and 'max_new_tokens' not in kwargs and kwargs['max_length'] is not None:
                kwargs = dict(kwargs)
                kwargs['max_length'] = kwargs['max_length'] + prefix_len

            generated = self.pretrained.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=states['kv_embeds']),
                attention_mask=states['kv_mask'],
                decoder_input_ids=decoder_input_ids,
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
            if generated.size(1) > prefix_len:
                generated = generated[:, prefix_len:]
            return generated

        encoder_outputs = self._encode_eeg(
            inputs_batch,
            invert_masks_batch,
            prior_ids_batch=prior_ids_batch,
        )
        encoded_embedding = self._project_to_pretrained_space(encoder_outputs)
        masks_batch = self._build_encoder_attention_mask(encoded_embedding, masks_batch)
        encoded_embedding, masks_batch = self._append_t5_task_embedding(encoded_embedding, masks_batch)
        return self.pretrained.generate(
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
