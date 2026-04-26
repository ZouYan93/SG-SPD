import os
import re
import gc
import json
import pickle
import torch
from glob import glob
from evaluate import load


def ensure_torch_generate_compat():
    """为旧版 torch 提供 generate 相关兼容能力。"""
    if hasattr(torch, 'isin'):
        return

    def _torch_isin(elements, test_elements):
        if not torch.is_tensor(elements):
            elements = torch.as_tensor(elements)
        if not torch.is_tensor(test_elements):
            test_elements = torch.as_tensor(test_elements, device=elements.device)
        else:
            test_elements = test_elements.to(elements.device)

        if test_elements.numel() == 0:
            return torch.zeros_like(elements, dtype=torch.bool)

        view_shape = [1] * elements.dim() + [-1]
        return (elements.unsqueeze(-1) == test_elements.reshape(view_shape)).any(dim=-1)

    torch.isin = _torch_isin


def remove_text_after_token(text, token='</s>'):
    """从文本中移除指定 token 及其后面的内容，常用于清理生成文本中的特殊结束标记"""
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
    """从训练目录和实验目录加载训练配置和实验配置"""
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
    """根据任务名称加载对应的整个数据集字典列表"""
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


def collect_ckpt_paths(search_paths, include_regex=r'.*\.pt$', exclude_regex_list=None):
    """
    根据输入路径（支持单个路径或路径列表）递归收集 checkpoint 路径。

    参数:
    - search_paths: str | list[str]，可传目录或具体文件路径
    - include_regex: str，包含规则，默认匹配 .pt 文件
    - exclude_regex_list: list[str]，剔除规则列表（支持正则），例如 [r'/last/']

    返回:
    - list[str]，按字典序排序且去重后的绝对路径列表
    """
    if isinstance(search_paths, str):
        path_list = [search_paths]
    elif isinstance(search_paths, list):
        path_list = search_paths
    else:
        raise TypeError('search_paths 必须是字符串或字符串列表')

    include_pattern = re.compile(include_regex)
    exclude_patterns = [re.compile(p) for p in (exclude_regex_list or [])]

    collected = []
    for raw_path in path_list:
        abs_path = os.path.abspath(raw_path)
        if not os.path.exists(abs_path):
            continue

        if os.path.isfile(abs_path):
            candidates = [abs_path]
        else:
            candidates = glob(os.path.join(abs_path, '**', '*.pt'), recursive=True)

        for ckpt_path in candidates:
            norm_path = os.path.abspath(ckpt_path)
            if not include_pattern.search(norm_path):
                continue
            if any(pattern.search(norm_path) for pattern in exclude_patterns):
                continue
            collected.append(norm_path)

    # 去重并排序，保证输出稳定
    return sorted(set(collected))


def export_eval_decoder_experiments_json(search_paths,
                                         output_json_path='./results/eval_decoder_experiments.json',
                                         include_regex=r'.*\.pt$',
                                         exclude_regex_list=None):
    """
    收集 checkpoint 路径并导出到 JSON 文件。

    输出格式:
    {
      "checkpoint_paths": ["/abs/path/a.pt", "/abs/path/b.pt", ...]
    }
    """
    checkpoint_paths = collect_ckpt_paths(
        search_paths=search_paths,
        include_regex=include_regex,
        exclude_regex_list=exclude_regex_list,
    )

    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump({'checkpoint_paths': checkpoint_paths}, f, ensure_ascii=False, indent=2)

    return checkpoint_paths


def _load_evaluate_metric(metric_name, metrics_dir=None):
    """加载 evaluate 库中的指标，优先尝试从本地目录加载，如果未找到则在线加载（或使用缓存）"""
    if metrics_dir:
        local_metric_path = os.path.join(metrics_dir, metric_name)
        if os.path.isdir(local_metric_path):
            print(f"[信息]从本地加载指标 {metric_name}: {local_metric_path}")
            return load(local_metric_path)

    print(f"[信息]在线/缓存加载指标 {metric_name}")
    return load(metric_name)


def load_metrics(metrics_dir=None):
    """加载 eval_decoder.py 评测用到的所有指标，返回一个包含各指标实例的元组"""
    metric = _load_evaluate_metric("sacrebleu", metrics_dir)
    cer_metric = _load_evaluate_metric("cer", metrics_dir)
    wer_metric = _load_evaluate_metric("wer", metrics_dir)
    return metric, cer_metric, wer_metric


def state_dict_to_cpu(state_dict):
    """将模型状态字典中的所有张量移动到 CPU 上，返回新的字典"""
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def release_cuda_memory(*objs):
    """删除对象并清理 CUDA 内存，适用于训练或评估过程中需要手动释放显存的场景"""
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()