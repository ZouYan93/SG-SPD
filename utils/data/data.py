import numpy as np
import torch
from torch.utils.data import Dataset

"""用于构建 ZuCo 数据集的 Dataset 类"""
class ZuCo_dataset(Dataset):
    def __init__(self, 
                 input_datasets, 
                 phase, 
                 tokenizer, 
                 subject = 'ALL', 
                 eeg_type = 'GD', 
                 bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                 setting = 'unique_sent', 
                 is_add_CLS_token = False, 
                 test_input='noise'):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_datasets,list):
            input_datasets = [input_datasets]
        print(f'[信息]加载 {len(input_datasets)} 个数据集')
        for input_dataset in input_datasets:
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
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'eval':
                    print('[信息]初始化验证集...')
                    for key in subjects:
                        for i in range(train_divider,eval_divider):
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print('[信息]初始化测试集...')
                    for key in subjects:
                        for i in range(eval_divider,total_num_sentence):
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            
            elif setting == 'unique_subj':
                print('[警告]只能用于v1.0版本SR任务')
                # 训练集受试者：['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW']
                # 验证集受试者：['ZMG']
                # 测试集受试者：['ZPH']
                if phase == 'train':
                    print(f'[信息]使用 {setting} 设置初始化训练集...')
                    for i in range(total_num_sentence):
                        for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH','ZKW']:
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'eval':
                    print(f'[信息]使用 {setting} 设置初始化验证集...')
                    for i in range(total_num_sentence):
                        for key in ['ZMG']:
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'test':
                    print(f'[信息]使用 {setting} 设置初始化测试集...')
                    for i in range(total_num_sentence):
                        for key in ['ZPH']:
                            input_sample = self.get_input_sample(input_dataset[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            
            print('[信息]当前样本数:', len(self.inputs))

        print('[信息]输入tensor形状:', self.inputs[0]['input_embeddings'].size())

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'], 
            input_sample['seq_len'],
            input_sample['mask'], 
            input_sample['invert_mask'],
            input_sample['target_ids'], 
            input_sample['target_mask'], 
            input_sample['sentiment_label'], 
            #input_sample['sent_level_EEG']
        )
        # 字段：input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, 

    @classmethod
    def normalize(cls, input_tensor):
        mean = torch.mean(input_tensor)
        std = torch.std(input_tensor)
        input_tensor = (input_tensor - mean)/std
        return input_tensor 

    @classmethod
    def get_word_eeg(cls, word_obj, eeg_type, bands):
        word_eeg_features = []

        for band in bands:
            word_eeg_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(word_eeg_features)

        if len(word_eeg_embedding) != 105*len(bands):
            print(f'期望词级 EEG 嵌入维度为 {105*len(bands)}，但实际得到 {len(word_eeg_embedding)}，返回 None')
            return None
        
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return cls.normalize(return_tensor)

    @classmethod
    def get_sent_eeg(cls, sent_obj, bands):
        sent_eeg_features = []

        for band in bands:
            sent_eeg_features.append(sent_obj['sentence_level_EEG']['mean'+band])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)

        if len(sent_eeg_embedding) != 105*len(bands):
            print(f'期望句子级 EEG 嵌入维度为 {105*len(bands)}，但实际得到 {len(sent_eeg_embedding)}，返回 None')
            return None
        
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return cls.normalize(return_tensor)

    @classmethod
    def get_input_sample(cls, 
                         sent_obj, 
                         tokenizer, 
                         eeg_type = 'GD', 
                         bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                         max_len = 60, 
                         add_CLS_token = False, 
                         test_input="noise"):
    
        # 若sent_obj为None，直接返回None
        if sent_obj is None:
            return None

        input_sample = {}
        # 获取目标标签
        target_string = sent_obj['content']
        target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask=True)
        input_sample['target_ids'] = target_tokenized['input_ids'][0]
        
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
        
        # 获取情感标签
        input_sample['sentiment_label'] = torch.tensor(-100) # 占位值

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