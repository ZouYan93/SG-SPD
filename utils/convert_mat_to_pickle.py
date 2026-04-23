import argparse
import os
import pickle
from glob import glob

import h5py
import numpy as np
import scipy.io as io
from tqdm import tqdm

from data_loading_helpers import DataLoadingHelper as dh


class MatToPickleConverter:
    """将 ZuCo 的 mat 文件转换为统一的 pickle 数据格式。"""

    SUPPORTED_TASKS = ['task1-SR', 'task2-NR', 'task3-TSR']
    SUPPORTED_VERSIONS = ['v1', 'v2']

    def __init__(self, base_dir='.'):
        self.base_dir = base_dir

    def convert(self, task_name, version='v1'):
        """
        执行 mat -> pickle 转换。

        参数:
            task_name: 任务名，仅支持 task1-SR/task2-NR/task3-TSR。
            version: mat 版本，仅支持 v1/v2。

        返回:
            str: 生成的 pickle 文件路径。
        """
        raw_task_name = task_name
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError("version 必须是 'v1' 或 'v2'")
        if task_name not in self.SUPPORTED_TASKS:
            raise ValueError("task_name 必须是 'task1-SR', 'task2-NR' 或 'task3-TSR'")

        task_name = task_name + '-2.0' if version == 'v2' else task_name

        print('#' * 50)
        print(f'开始处理 ZuCo {task_name}...')

        input_mat_files_dir = os.path.join(self.base_dir, f'dataset/ZuCo/{task_name}/Matlab_files')
        output_dir = os.path.join(self.base_dir, f'dataset/ZuCo/{task_name}/pickle')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_name = f'{task_name}-dataset.pickle'
        output_path = os.path.join(output_dir, output_name)

        mat_files = sorted(glob(os.path.join(input_mat_files_dir, '*.mat')))
        if len(mat_files) == 0:
            raise FileNotFoundError(f'未找到 mat 文件: {input_mat_files_dir}')

        dataset_dict = {}
        for mat_file in tqdm(mat_files):
            subject_name = os.path.basename(mat_file).split('_')[0].replace('results', '').strip()
            dataset_dict[subject_name] = []

            if version == 'v1':
                self._process_v1_file(
                    mat_file=mat_file,
                    subject_name=subject_name,
                    dataset_dict=dataset_dict,
                    raw_task_name=raw_task_name,
                )
            else:
                self._process_v2_file(
                    mat_file=mat_file,
                    subject_name=subject_name,
                    dataset_dict=dataset_dict,
                )

        with open(output_path, 'wb') as handle:
            pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('保存到:', output_path)

        with open(output_path, 'rb') as handle:
            dataset = pickle.load(handle)

        print('试验者:', dataset.keys())
        if version == 'v1' and 'ZAB' in dataset:
            print('num of sent:', len(dataset['ZAB']))
        if version == 'v2' and 'YAC' in dataset:
            print('num of sent:', len(dataset['YAC']))

        return output_path

    def _process_v1_file(self, mat_file, subject_name, dataset_dict, raw_task_name):
        """处理 v1 mat 文件。"""
        matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']
        for sent in matdata:
            word_data = sent.word
            if isinstance(word_data, float):
                print(f'缺失句子: 试验者:{subject_name} 内容:{sent.content}, return None')
                dataset_dict[subject_name].append(None)
                continue

            sent_obj = {'content': sent.content}
            sent_obj['sentence_level_EEG'] = {
                'mean_t1': sent.mean_t1,
                'mean_t2': sent.mean_t2,
                'mean_a1': sent.mean_a1,
                'mean_a2': sent.mean_a2,
                'mean_b1': sent.mean_b1,
                'mean_b2': sent.mean_b2,
                'mean_g1': sent.mean_g1,
                'mean_g2': sent.mean_g2,
            }

            if raw_task_name == 'task1-SR':
                sent_obj['answer_EEG'] = {
                    'answer_mean_t1': sent.answer_mean_t1,
                    'answer_mean_t2': sent.answer_mean_t2,
                    'answer_mean_a1': sent.answer_mean_a1,
                    'answer_mean_a2': sent.answer_mean_a2,
                    'answer_mean_b1': sent.answer_mean_b1,
                    'answer_mean_b2': sent.answer_mean_b2,
                    'answer_mean_g1': sent.answer_mean_g1,
                    'answer_mean_g2': sent.answer_mean_g2,
                }

            sent_obj['word'] = []
            word_tokens_has_fixation = []
            word_tokens_with_mask = []
            word_tokens_all = []

            for word in word_data:
                word_obj = {'content': word.content}
                word_tokens_all.append(word.content)
                word_obj['nFixations'] = word.nFixations

                if word.nFixations > 0:
                    word_obj['word_level_EEG'] = {
                        'FFD': {
                            'FFD_t1': word.FFD_t1,
                            'FFD_t2': word.FFD_t2,
                            'FFD_a1': word.FFD_a1,
                            'FFD_a2': word.FFD_a2,
                            'FFD_b1': word.FFD_b1,
                            'FFD_b2': word.FFD_b2,
                            'FFD_g1': word.FFD_g1,
                            'FFD_g2': word.FFD_g2,
                        },
                        'TRT': {
                            'TRT_t1': word.TRT_t1,
                            'TRT_t2': word.TRT_t2,
                            'TRT_a1': word.TRT_a1,
                            'TRT_a2': word.TRT_a2,
                            'TRT_b1': word.TRT_b1,
                            'TRT_b2': word.TRT_b2,
                            'TRT_g1': word.TRT_g1,
                            'TRT_g2': word.TRT_g2,
                        },
                        'GD': {
                            'GD_t1': word.GD_t1,
                            'GD_t2': word.GD_t2,
                            'GD_a1': word.GD_a1,
                            'GD_a2': word.GD_a2,
                            'GD_b1': word.GD_b1,
                            'GD_b2': word.GD_b2,
                            'GD_g1': word.GD_g1,
                            'GD_g2': word.GD_g2,
                        },
                    }
                    sent_obj['word'].append(word_obj)
                    word_tokens_has_fixation.append(word.content)
                    word_tokens_with_mask.append(word.content)
                else:
                    word_tokens_with_mask.append('[MASK]')

            sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
            sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
            sent_obj['word_tokens_all'] = word_tokens_all
            dataset_dict[subject_name].append(sent_obj)

    def _process_v2_file(self, mat_file, subject_name, dataset_dict):
        """处理 v2 mat 文件。"""
        with h5py.File(mat_file, 'r') as file_obj:
            matdata = file_obj['sentenceData']

            if subject_name == 'YMH':
                return

            mean_t1_objs = matdata['mean_t1']
            mean_t2_objs = matdata['mean_t2']
            mean_a1_objs = matdata['mean_a1']
            mean_a2_objs = matdata['mean_a2']
            mean_b1_objs = matdata['mean_b1']
            mean_b2_objs = matdata['mean_b2']
            mean_g1_objs = matdata['mean_g1']
            mean_g2_objs = matdata['mean_g2']

            raw_data = matdata['rawData']
            content_data = matdata['content']
            word_data = matdata['word']

            for idx in range(len(raw_data)):
                reference_content = content_data[idx][0]
                sent_string = dh.load_matlab_string(file_obj[reference_content])
                sent_obj = {'content': sent_string}

                sent_obj['sentence_level_EEG'] = {
                    'mean_t1': np.squeeze(file_obj[mean_t1_objs[idx][0]][()]),
                    'mean_t2': np.squeeze(file_obj[mean_t2_objs[idx][0]][()]),
                    'mean_a1': np.squeeze(file_obj[mean_a1_objs[idx][0]][()]),
                    'mean_a2': np.squeeze(file_obj[mean_a2_objs[idx][0]][()]),
                    'mean_b1': np.squeeze(file_obj[mean_b1_objs[idx][0]][()]),
                    'mean_b2': np.squeeze(file_obj[mean_b2_objs[idx][0]][()]),
                    'mean_g1': np.squeeze(file_obj[mean_g1_objs[idx][0]][()]),
                    'mean_g2': np.squeeze(file_obj[mean_g2_objs[idx][0]][()]),
                }

                sent_obj['word'] = []
                parsed_word_data = dh.extract_word_level_data(file_obj, file_obj[word_data[idx][0]])
                word_level_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = parsed_word_data

                if word_level_data == {}:
                    print(f'句子数据缺失：被试:{subject_name} 内容:{sent_string}，追加 None')
                    dataset_dict[subject_name].append(None)
                    continue
                if len(word_tokens_all) == 0:
                    print(f'无词级特征：被试:{subject_name} 内容:{sent_string}，追加 None')
                    dataset_dict[subject_name].append(None)
                    continue

                for widx in range(len(word_level_data)):
                    data_dict = word_level_data[widx]
                    word_obj = {'content': data_dict['content'], 'nFixations': data_dict['nFix']}
                    if 'GD_EEG' not in data_dict:
                        continue

                    ffd = data_dict['FFD_EEG']
                    trt = data_dict['TRT_EEG']
                    gd = data_dict['GD_EEG']
                    assert len(gd) == len(trt) == len(ffd) == 8

                    word_obj['word_level_EEG'] = {
                        'FFD': {
                            'FFD_t1': ffd[0],
                            'FFD_t2': ffd[1],
                            'FFD_a1': ffd[2],
                            'FFD_a2': ffd[3],
                            'FFD_b1': ffd[4],
                            'FFD_b2': ffd[5],
                            'FFD_g1': ffd[6],
                            'FFD_g2': ffd[7],
                        },
                        'TRT': {
                            'TRT_t1': trt[0],
                            'TRT_t2': trt[1],
                            'TRT_a1': trt[2],
                            'TRT_a2': trt[3],
                            'TRT_b1': trt[4],
                            'TRT_b2': trt[5],
                            'TRT_g1': trt[6],
                            'TRT_g2': trt[7],
                        },
                        'GD': {
                            'GD_t1': gd[0],
                            'GD_t2': gd[1],
                            'GD_a1': gd[2],
                            'GD_a2': gd[3],
                            'GD_b1': gd[4],
                            'GD_b2': gd[5],
                            'GD_g1': gd[6],
                            'GD_g2': gd[7],
                        },
                    }
                    sent_obj['word'].append(word_obj)

                sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
                sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
                sent_obj['word_tokens_all'] = word_tokens_all
                dataset_dict[subject_name].append(sent_obj)