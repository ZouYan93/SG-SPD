import os
import re
import json
import pickle
from glob import glob

def remove_text_after_token(text, token='</s>'):
    token_index = text.find(token)
    if token_index != -1:
        return text[:token_index]
    return text


def parse_run_and_experiment_from_ckpt(ckpt_path):
    """
    从检查点路径中解析出 run_dir、run_name 和 experiment_dir
    例如路径: ./experiments/decoder/task1_task2_task3_20260425-195643/Bart_Transformer_EEG/best/Bart_Transformer_EEG.pt
    - run_dir: ./experiments/decoder/task1_task2_task3_20260425-195643
    - run_name: task1_task2_task3_20260425-195643
    - experiment_dir: ./experiments/decoder/task1_task2_task3_20260425-195643/Bart_Transformer_EEG
    """
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'找不到检查点: {ckpt_path}')

    parent_dir = os.path.dirname(ckpt_path)
    assert os.path.basename(parent_dir) in ('best', 'last'), f"非法路径: {parent_dir}"
    experiment_dir = os.path.dirname(parent_dir)

    run_dir = None
    cur = experiment_dir
    pattern = re.compile(r'.*_\d{8}-\d{6}$')
    while True:
        base = os.path.basename(cur)
        if pattern.match(base):
            run_dir = cur
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    if run_dir is None:
        raise ValueError(f'无法从检查点路径解析 task_timestamp: {ckpt_path}')

    run_name = os.path.basename(run_dir)
    return run_dir, run_name, experiment_dir


def load_training_and_experiment_configs(run_dir, experiment_dir):
    train_config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(train_config_path):
        raise FileNotFoundError(f'找不到训练配置文件: {train_config_path}')
    with open(train_config_path, 'r', encoding='utf-8') as f:
        training_config = json.load(f)

    config_candidates = sorted(glob(os.path.join(experiment_dir, 'config', '*.json')))
    if not config_candidates:
        raise FileNotFoundError(f'找不到实验配置文件: {os.path.join(experiment_dir, "config")}')
    with open(config_candidates[0], 'r', encoding='utf-8') as f:
        experiment_config = json.load(f)

    return training_config, experiment_config


def load_dataset_dicts(task_name):
    whole_dataset_dicts = []
    if 'task1' in task_name:
        with open('./dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        with open('./dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        with open('./dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        with open('./dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle', 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    return whole_dataset_dicts
