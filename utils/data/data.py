import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from model.encoder import PriorLabelEmbedder
from utils.data.semkey_attributes import (
    SemKeyAttributeProvider,
    matching_text_keys,
    normalize_text as normalize_semkey_text,
)

"""用于构建 ZuCo 数据集的 Dataset 类"""
class ZuCo_dataset(Dataset):
    _global_norm_stats = {}
    _mtv_cache = {}
    DEFAULT_MTV_VARIANT_KEYS = [
        'lexical simplification (v0)',
        'lexical simplification (v1)',
        'semantic clarity (v0)',
        'semantic clarity (v1)',
        'syntax simplification (v0)',
        'syntax simplification (v1)',
        'naive rewritten',
        'naive simplified',
    ]

    def __init__(self, 
                 input_datasets, 
                 phase, 
                 tokenizer, 
                 subject = 'ALL', 
                 eeg_type = 'GD', 
                 bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                 setting = 'unique_sent', 
                 is_add_CLS_token = False, 
                 test_input='noise',
                 task_name=None,
                 use_mtv=False,
                 mtv_path=None,
                 mtv_variant_keys=None,
                 mtv_include_original=False,
                 return_all_target_texts=False,
                 attribute_path=None,
                 require_semkey_attributes=False):
        self.inputs = []
        self.tokenizer = tokenizer
        self.dataset_plan = self._build_dataset_plan(task_name, input_datasets)
        self.use_mtv = bool(use_mtv and phase == 'train')
        self.return_all_target_texts = bool(return_all_target_texts)
        self.mtv_variant_keys = mtv_variant_keys or self.DEFAULT_MTV_VARIANT_KEYS
        self.mtv_include_original = bool(mtv_include_original)
        self.mtv_lookup = self._load_mtv_lookup(mtv_path, self.mtv_variant_keys) if (self.use_mtv or self.return_all_target_texts) else {}
        self.mtv_base_count = 0
        self.mtv_expanded_count = 0
        self.attribute_provider = SemKeyAttributeProvider(
            self._repo_root(),
            attribute_path=attribute_path,
            strict=require_semkey_attributes,
        )

        if not isinstance(input_datasets,list):
            input_datasets = [input_datasets]
        print(f'[信息]加载 {len(input_datasets)} 个数据集')
        for dataset_idx, input_dataset in enumerate(input_datasets):
            task_key, dataset_key = self.dataset_plan[dataset_idx]
            if subject == 'ALL':
                subjects = list(input_dataset.keys())
                print('[信息]被试者: ', subjects)
            else:
                subjects = [subject]
            
            total_num_sentence = len(input_dataset[subjects[0]])
            
            train_divider = int(0.8*total_num_sentence)
            eval_divider = train_divider + int(0.1*total_num_sentence)
            
            print(f'训练集划分位置 {train_divider}')
            print(f'验证集划分位置 {eval_divider}')

            if setting == 'unique_sent':
                # 前 80% 作为训练集，10% 作为验证集，10% 作为测试集
                if phase == 'train':
                    print('[信息]初始化训练集...')
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input, task_key=task_key, dataset_key=dataset_key, subject_key=key, attribute_provider=self.attribute_provider)
                            if input_sample is not None:
                                self._append_input_sample(input_sample)
                elif phase == 'eval':
                    print('[信息]初始化验证集...')
                    for key in subjects:
                        for i in range(train_divider,eval_divider):
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input, task_key=task_key, dataset_key=dataset_key, subject_key=key, attribute_provider=self.attribute_provider)
                            if input_sample is not None:
                                self._append_input_sample(input_sample)
                elif phase == 'test':
                    print('[信息]初始化测试集...')
                    for key in subjects:
                        for i in range(eval_divider,total_num_sentence):
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input, task_key=task_key, dataset_key=dataset_key, subject_key=key, attribute_provider=self.attribute_provider)
                            if input_sample is not None:
                                self._append_input_sample(input_sample)
            
            elif setting == 'unique_subj':
                print('[警告]只能用于v1.0版本SR任务')
                # 训练集受试者：['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW']
                # 验证集受试者：['ZMG']
                # 测试集受试者：['ZPH']
                if phase == 'train':
                    print(f'[信息]使用 {setting} 设置初始化训练集...')
                    for i in range(total_num_sentence):
                        for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH','ZKW']:
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, task_key=task_key, dataset_key=dataset_key, subject_key=key, attribute_provider=self.attribute_provider)
                            if input_sample is not None:
                                self._append_input_sample(input_sample)
                if phase == 'eval':
                    print(f'[信息]使用 {setting} 设置初始化验证集...')
                    for i in range(total_num_sentence):
                        for key in ['ZMG']:
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, task_key=task_key, dataset_key=dataset_key, subject_key=key, attribute_provider=self.attribute_provider)
                            if input_sample is not None:
                                self._append_input_sample(input_sample)
                if phase == 'test':
                    print(f'[信息]使用 {setting} 设置初始化测试集...')
                    for i in range(total_num_sentence):
                        for key in ['ZPH']:
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, task_key=task_key, dataset_key=dataset_key, subject_key=key, attribute_provider=self.attribute_provider)
                            if input_sample is not None:
                                self._append_input_sample(input_sample)
            
            print('[信息]当前样本数:', len(self.inputs))

        print('[信息]输入tensor形状:', self.inputs[0]['input_embeddings'].size())
        if self.use_mtv:
            print(f'[信息] MTV 训练增强: base={self.mtv_base_count}, expanded={self.mtv_expanded_count}')

        if test_input != 'noise':
            stats_key = self._get_norm_stats_key(subject, eeg_type, bands, setting, is_add_CLS_token)
            if phase == 'train' or stats_key not in self._global_norm_stats:
                self._global_norm_stats[stats_key] = self._compute_global_norm_stats(self.inputs)
                print('[信息]已计算训练集 EEG 全局归一化统计量')
            self._apply_global_normalization(self.inputs, self._global_norm_stats[stats_key])

    def __len__(self):
        return len(self.inputs)

    def _append_input_sample(self, input_sample):
        self.mtv_base_count += 1
        self._set_all_target_texts(input_sample)
        if not self.use_mtv:
            self.inputs.append(input_sample)
            self.mtv_expanded_count += 1
            return

        variants = self._get_mtv_variants(input_sample.get('target_string', ''))
        if self.mtv_include_original:
            variants = [input_sample.get('target_string', '')] + variants
        if not variants:
            raise KeyError(f"MTV enabled but no variants matched for target: {input_sample.get('target_string', '')}")

        for variant_text in variants:
            variant_sample = input_sample.copy()
            self._set_target_text(variant_sample, variant_text)
            self._set_all_target_texts(variant_sample)
            self.inputs.append(variant_sample)
        self.mtv_expanded_count += len(variants)

    @classmethod
    def _build_dataset_plan(cls, task_name, input_datasets):
        """根据加载顺序还原 task/dataset 先验标签。"""
        plan = []
        if task_name:
            if 'task1' in task_name:
                plan.append(('<SR>', 'ZuCo1'))
            if 'task2' in task_name:
                plan.append(('<NR>', 'ZuCo1'))
            if 'task3' in task_name:
                plan.append(('<TSR>', 'ZuCo1'))
            if 'taskNRv2' in task_name:
                plan.append(('<NR>', 'ZuCo2'))
        fallback = [('<SR>', 'ZuCo1'), ('<NR>', 'ZuCo1'), ('<TSR>', 'ZuCo1'), ('<NR>', 'ZuCo2')]
        while len(plan) < len(input_datasets):
            plan.append(fallback[len(plan)] if len(plan) < len(fallback) else ('<UNK>', '<UNK>'))
        return plan[:len(input_datasets)]

    @classmethod
    def _repo_root(cls):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    @staticmethod
    def _normalize_text(text):
        return normalize_semkey_text(text)

    @classmethod
    def _default_mtv_path_candidates(cls):
        base_dir = cls._repo_root()
        return [
            os.path.join(base_dir, 'dataset', 'ZuCo', 'zuco_label_8variants.df')
        ]

    @classmethod
    def _resolve_mtv_path(cls, mtv_path):
        if mtv_path:
            path = mtv_path
            if not os.path.isabs(path):
                path = os.path.join(cls._repo_root(), path)
            return os.path.abspath(path)
        for path in cls._default_mtv_path_candidates():
            if os.path.exists(path):
                return path
        return None

    @classmethod
    def _load_mtv_lookup(cls, mtv_path, variant_keys):
        resolved_path = cls._resolve_mtv_path(mtv_path)
        cache_key = (resolved_path, tuple(variant_keys))
        if cache_key in cls._mtv_cache:
            return cls._mtv_cache[cache_key]

        if not resolved_path or not os.path.exists(resolved_path):
            print(f'[警告] 未找到 MTV 文件: {mtv_path or cls._default_mtv_path_candidates()[0]}，训练将回退到原始 target。')
            cls._mtv_cache[cache_key] = {}
            return cls._mtv_cache[cache_key]

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError('启用 MTV 需要安装 pandas，用于读取 zuco_label_8variants.df。') from exc

        df = pd.read_pickle(resolved_path)
        lookup = {}
        key_columns = [col for col in ('input text', 'raw text') if col in df.columns]
        usable_variant_keys = [key for key in variant_keys if key in df.columns]
        missing_keys = [key for key in variant_keys if key not in df.columns]
        if missing_keys:
            print(f'[警告] MTV 文件缺少列: {missing_keys}')

        for _, row in df.iterrows():
            variants = []
            for key in usable_variant_keys:
                value = row.get(key)
                if isinstance(value, str):
                    value = cls._normalize_text(value)
                    if value and value not in variants:
                        variants.append(value)
            if not variants:
                continue
            for key_col in key_columns:
                for source_text in matching_text_keys(row.get(key_col, '')):
                    if source_text and source_text not in lookup:
                        lookup[source_text] = variants

        cls._mtv_cache[cache_key] = lookup
        print(f'[信息] 已加载 MTV 变体: {resolved_path}, matched_texts={len(lookup)}, variant_keys={usable_variant_keys}')
        return cls._mtv_cache[cache_key]

    def _get_mtv_variants(self, target_string):
        for key in matching_text_keys(target_string):
            variants = self.mtv_lookup.get(key)
            if variants:
                return variants
        return []

    def _get_all_target_texts(self, target_string):
        variants = self._get_mtv_variants(target_string)
        return variants or [self._normalize_text(target_string)]

    def _set_all_target_texts(self, input_sample):
        all_target_texts = self._get_all_target_texts(input_sample.get('target_string', ''))
        input_sample['all_target_texts'] = json.dumps(all_target_texts, ensure_ascii=False)

    def _set_target_text(self, input_sample, target_string):
        target_string = self._normalize_text(target_string)
        target_tokenized = self.tokenizer(
            target_string,
            padding='max_length',
            max_length=input_sample.get('max_len', 60),
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
        )
        input_sample['target_string'] = target_string
        input_sample['target_ids'] = target_tokenized['input_ids'][0]
        input_sample['target_mask'] = target_tokenized['attention_mask'][0]

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        output = (
            input_sample['input_embeddings'], 
            input_sample['seq_len'],
            input_sample['mask'], 
            input_sample['invert_mask'],
            input_sample['target_ids'], 
            input_sample['target_mask'], 
            input_sample['sentiment_label'], 
            input_sample['prior_ids'],
            input_sample['original_target_ids'],
            input_sample['original_target_mask'],
            input_sample['topic_label'],
            input_sample['length_value'],
            input_sample['surprisal_value'],
            #input_sample['sent_level_EEG']
        )
        if self.return_all_target_texts:
            output = output + (input_sample['all_target_texts'],)
        return output
        # 字段：input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, 

    @classmethod
    def normalize(cls, input_tensor):
        mean = torch.mean(input_tensor)
        std = torch.std(input_tensor)
        input_tensor = (input_tensor - mean)/std
        return input_tensor 

    @classmethod
    def _get_norm_stats_key(cls, subject, eeg_type, bands, setting, add_CLS_token):
        return (str(subject), str(eeg_type), tuple(bands), str(setting), bool(add_CLS_token))

    @classmethod
    def _compute_global_norm_stats(cls, inputs):
        valid_tokens = []
        for sample in inputs:
            mask = sample['mask'].bool()
            valid_tokens.append(sample['input_embeddings'][mask])
        stacked = torch.cat(valid_tokens, dim=0).float()
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0).clamp_min(1e-6)
        return mean, std

    @classmethod
    def _apply_global_normalization(cls, inputs, stats):
        mean, std = stats
        for sample in inputs:
            x = sample['input_embeddings'].float()
            mask = sample['mask'].float().unsqueeze(-1)
            sample['input_embeddings'] = ((x - mean) / std) * mask

    def get_semkey_regression_stats(self):
        length_values = []
        surprisal_values = []
        for sample in self.inputs:
            length_value = float(sample['length_value'])
            surprisal_value = float(sample['surprisal_value'])
            if np.isfinite(length_value):
                length_values.append(length_value)
            if np.isfinite(surprisal_value):
                surprisal_values.append(surprisal_value)

        def mean_std(values):
            if not values:
                return 0.0, 1.0
            values = np.asarray(values, dtype=np.float32)
            return float(values.mean()), float(max(values.std(), 1e-6))

        length_mean, length_std = mean_std(length_values)
        surprisal_mean, surprisal_std = mean_std(surprisal_values)
        return {
            'length_mean': length_mean,
            'length_std': length_std,
            'surprisal_mean': surprisal_mean,
            'surprisal_std': surprisal_std,
            'has_surprisal': bool(surprisal_values),
        }

    @classmethod
    def get_word_eeg(cls, word_obj, eeg_type, bands):
        word_eeg_features = []

        for band in bands:
            word_eeg_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(word_eeg_features)

        if len(word_eeg_embedding) != 105*len(bands):
            print(f'期望词级 EEG 嵌入维度为 {105*len(bands)}，但实际得到 {len(word_eeg_embedding)}，返回 None')
            return None
        
        return torch.from_numpy(word_eeg_embedding).float()

    @classmethod
    def get_sent_eeg(cls, sent_obj, bands):
        sent_eeg_features = []

        for band in bands:
            sent_eeg_features.append(sent_obj['sentence_level_EEG']['mean'+band])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)

        if len(sent_eeg_embedding) != 105*len(bands):
            print(f'期望句子级 EEG 嵌入维度为 {105*len(bands)}，但实际得到 {len(sent_eeg_embedding)}，返回 None')
            return None
        
        return torch.from_numpy(sent_eeg_embedding).float()

    @classmethod
    def get_input_sample(cls, 
                         sent_obj, 
                         tokenizer, 
                         eeg_type = 'GD', 
                         bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                         max_len = 60, 
                         add_CLS_token = False, 
                         test_input="noise",
                         task_key='<UNK>',
                         dataset_key='<UNK>',
                         subject_key='<UNK>',
                         attribute_provider=None):
    
        # 若sent_obj为None，直接返回None
        if sent_obj is None:
            return None

        input_sample = {}
        input_sample['prior_ids'] = torch.tensor(
            PriorLabelEmbedder.encode_one(task_key, dataset_key, subject_key),
            dtype=torch.long,
        )
        # 获取目标标签
        target_string = cls._normalize_text(sent_obj['content'])
        input_sample['target_string'] = target_string
        input_sample['original_target_string'] = target_string
        input_sample['max_len'] = max_len
        target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask=True)
        input_sample['target_ids'] = target_tokenized['input_ids'][0]
        input_sample['original_target_ids'] = target_tokenized['input_ids'][0].clone()
        input_sample['original_target_mask'] = target_tokenized['attention_mask'][0].clone()
        
        # 获取句子级 EEG 特征
        sent_level_eeg_tensor = cls.get_sent_eeg(sent_obj, bands)
        if torch.isnan(sent_level_eeg_tensor).any():
            return None
        input_sample['sent_level_EEG'] = sent_level_eeg_tensor

        # 处理一些特殊情况
        if 'emp11111ty' in target_string:
            target_string = target_string.replace('emp11111ty','empty')
        if 'film.1' in target_string:
            target_string = target_string.replace('film.1','film.')
        
        if attribute_provider is None:
            attribute_provider = SemKeyAttributeProvider(cls._repo_root(), strict=False)
        attributes = attribute_provider.get(target_string, task_key=task_key)
        input_sample['sentiment_label'] = torch.tensor(attributes['sentiment_label'], dtype=torch.long)
        input_sample['topic_label'] = torch.tensor(attributes['topic_label'], dtype=torch.long)
        input_sample['length_value'] = torch.tensor(attributes['length'], dtype=torch.float32)
        input_sample['surprisal_value'] = torch.tensor(attributes['surprisal'], dtype=torch.float32)

        # 获取输入嵌入
        word_embeddings = []

        """在最前面添加 CLS token 的嵌入"""
        if add_CLS_token:
            word_embeddings.append(torch.ones(105*len(bands)))

        for word in sent_obj['word']:
            word_level_eeg_tensor = cls.get_word_eeg(word, eeg_type, bands = bands)
            if word_level_eeg_tensor is None:
                return None
            if torch.isnan(word_level_eeg_tensor).any():
                return None
            word_embeddings.append(word_level_eeg_tensor)

        # 补齐到 max_len
        while len(word_embeddings) < max_len:
            word_embeddings.append(torch.zeros(105*len(bands)))

        if test_input=='noise':
            rand_eeg= torch.randn(torch.stack(word_embeddings).size())
            input_sample['input_embeddings'] = rand_eeg
        else:
            input_sample['input_embeddings'] = torch.stack(word_embeddings)
        
        # 屏蔽 padding token
        input_sample['mask'] = torch.zeros(max_len) # 0 表示被屏蔽
        if add_CLS_token:
            input_sample['mask'][:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1) # 1 表示不屏蔽
        else:
            input_sample['mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) # 1 表示不屏蔽
        
        # 反向屏蔽 padding token：适配 pytorch transformer
        input_sample['invert_mask'] = torch.ones(max_len) # 1 表示被屏蔽
        if add_CLS_token:
            input_sample['invert_mask'][:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1) # 0 表示不屏蔽
        else:
            input_sample['invert_mask'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word'])) # 0 表示不屏蔽

        if test_input == 'noise':
            input_sample['mask'] = torch.ones(max_len)
            input_sample['invert_mask'] = torch.zeros(max_len)

        # 屏蔽目标序列中的 padding，用于计算交叉熵损失
        input_sample['target_mask'] = target_tokenized['attention_mask'][0]
        input_sample['seq_len'] = len(sent_obj['word'])
        
        # 清理长度为 0 的样本
        if input_sample['seq_len'] == 0:
            return None

        return input_sample

"""用于构建 SST 数据集的 Dataset 类"""
class SST_tenary_dataset(Dataset):
    def __init__(self, ternary_labels_dict, tokenizer, max_len = 60, balance_class = True):
        self.inputs = []
        
        pos_samples = []
        neg_samples = []
        neu_samples = []

        for key, value in ternary_labels_dict.items():
            tokenized_inputs = tokenizer(key, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask=True)
            input_ids = tokenized_inputs['input_ids'][0]
            masks = tokenized_inputs['attention_mask'][0]
            label = torch.tensor(value)
            # 统计类别数量
            if value == 0:
                neg_samples.append((input_ids, masks, label))
            elif value == 1:
                neu_samples.append((input_ids, masks, label))
            elif value == 2:
                pos_samples.append((input_ids, masks, label))
        print(f'[信息]原始类别分布:\n\t非常正向: {len(pos_samples)}\n\t中性: {len(neu_samples)}\n\t非常负向: {len(neg_samples)}')    
        if balance_class:
            print(f'[信息]将各类别样本数均衡到 {min([len(pos_samples),len(neg_samples),len(neu_samples)])}...')
            for i in range(min([len(pos_samples),len(neg_samples),len(neu_samples)])):
                self.inputs.append(pos_samples[i])
                self.inputs.append(neg_samples[i])
                self.inputs.append(neu_samples[i])
        else:
            self.inputs = pos_samples + neg_samples + neu_samples
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return input_sample
