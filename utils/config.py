import argparse
import copy
import json

DEFAULT_TRAIN_DECODER_CONFIG = {
    # 全局配置
    'task_name': 'task1',
    'save_dir': './experiments/decoder',
    'cuda': 'cuda:0',
    'subjects': 'ALL',
    'eeg_type': 'GD',
    'eeg_bands': ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
    'batch_size': 32,
    'num_epoch_step1': 20,
    'num_epoch_step2': 30,
    'learning_rate_step1': 0.00002,
    'learning_rate_step2': 0.00002,
    'skip_step_one': False,
    'load_step1_checkpoint': False,
    'stepone_checkpoint': '',
    'use_random_init': False,
    'seed': 12,
    'experiments': [
        {'pretrained': 'Bart', 'encoder': 'Transformer', 'train_input': 'noise'},
    ],
}

DEFAULT_EVAL_DECODER_CONFIG = {
    'search_paths_json_path': './results/eval_decoder_search_paths.json',
    'include_regex': r'.*\.pt$',
    'exclude_regex_list': [r'/last/'],
    'ckpt_list_output_json_path': './results/eval_decoder_checkpoints.json',
    'test_input': 'EEG',
    'cuda': 'cuda:0',
    'seed': 12,
    'batch_size': 1,
}


def _build_train_decoder_parser():
    parser = argparse.ArgumentParser(
        description='训练 EEG-To-Text 解码器配置'
    )

    # 全局配置
    parser.add_argument('--task_name', type=str, default=None, help='任务名称，用于区分不同实验, 只能为{task1, task1_task2, task1_task2_task3, task1_task2_taskNRv2}之一')
    parser.add_argument('--save_dir', type=str, default=None, help='模型保存目录')
    parser.add_argument('--cuda', type=str, default=None, help='CUDA 设备')
    parser.add_argument('--subjects', type=str, default=None, help='受试者列表')
    parser.add_argument('--eeg_type', type=str, default=None, help='EEG 类型')
    parser.add_argument('--eeg_bands', nargs='+', default=None, help='EEG 频段列表')
    parser.add_argument('--batch_size', type=int, default=None, help='训练批次大小')
    parser.add_argument('--num_epoch_step1', type=int, default=None, help='Step1 训练轮数')
    parser.add_argument('--num_epoch_step2', type=int, default=None, help='Step2 训练轮数')
    parser.add_argument('--learning_rate_step1', type=float, default=None, help='Step1 学习率')
    parser.add_argument('--learning_rate_step2', type=float, default=None, help='Step2 学习率')
    parser.add_argument('--stepone_checkpoint', type=str, default=None, help='Step1 检查点路径')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')

    # 开关项
    parser.add_argument('--skip_step_one', dest='skip_step_one', action='store_true', help='是否跳过 Step1 训练')
    parser.add_argument('--no_skip_step_one', dest='skip_step_one', action='store_false', help='不跳过 Step1 训练')
    parser.set_defaults(skip_step_one=None)

    parser.add_argument('--load_step1_checkpoint', dest='load_step1_checkpoint', action='store_true', help='是否加载 Step1 检查点')
    parser.add_argument('--no_load_step1_checkpoint', dest='load_step1_checkpoint', action='store_false', help='不加载 Step1 检查点')
    parser.set_defaults(load_step1_checkpoint=None)

    parser.add_argument('--use_random_init', dest='use_random_init', action='store_true', help='是否使用随机初始化')
    parser.add_argument('--use_pretrained_init', dest='use_random_init', action='store_false', help='使用预训练初始化')
    parser.set_defaults(use_random_init=None)

    # 支持直接传 JSON 字符串，或通过文件读取
    parser.add_argument(
        '--experiments_json',
        type=str,
        default=None,
        help='实验列表 JSON 字符串，例如: [{"pretrained":"Bart","encoder":"Transformer","train_input":"noise"}]',
    )
    parser.add_argument(
        '--experiments_json_path',
        type=str,
        default=None,
        help='实验列表 JSON 文件路径，文件内容应为 list[dict]',
    )

    return parser


def _build_eval_decoder_parser():
    parser = argparse.ArgumentParser(
        description='评估 EEG-To-Text 解码器配置'
    )
    parser.add_argument(
        '--search_paths_json_path',
        type=str,
        default=None,
        help='JSON 文件路径，文件中存放待搜索的路径列表（用于自动收集 ckpt）',
    )
    parser.add_argument(
        '--include_regex',
        type=str,
        default=None,
        help='ckpt 包含正则，默认匹配 .pt 文件',
    )
    parser.add_argument(
        '--exclude_regex_list',
        nargs='*',
        default=None,
        help='ckpt 排除正则列表，例如 --exclude_regex_list /last/ /Bart_MLP_EEG/',
    )
    parser.add_argument(
        '--ckpt_list_output_json_path',
        type=str,
        default=None,
        help='自动收集后的 ckpt 列表导出路径',
    )
    parser.add_argument('--test_input', type=str, default=None, help='评估输入类型，如 noise/clean')
    parser.add_argument('--cuda', type=str, default=None, help='CUDA 设备')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=None, help='评估批次大小')
    return parser


def _load_experiments_from_args(parsed_args):
    if parsed_args.experiments_json_path:
        with open(parsed_args.experiments_json_path, 'r', encoding='utf-8') as f:
            experiments = json.load(f)
    elif parsed_args.experiments_json:
        experiments = json.loads(parsed_args.experiments_json)
    else:
        experiments = None

    if experiments is None:
        return None

    if not isinstance(experiments, list) or not experiments:
        raise ValueError('experiments 必须是非空列表')

    for i, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            raise ValueError(f'experiments[{i}] 必须是字典')
        for key in ('pretrained', 'encoder', 'train_input'):
            if key not in exp:
                raise ValueError(f'experiments[{i}] 缺少字段: {key}')

    return experiments


def get_config(case):
    if case == 'train_decoder':
        parser = _build_train_decoder_parser()
        parsed_args = parser.parse_args()
        config = copy.deepcopy(DEFAULT_TRAIN_DECODER_CONFIG)
        for key in (
            'task_name',
            'save_dir',
            'cuda',
            'subjects',
            'eeg_type',
            'eeg_bands',
            'batch_size',
            'num_epoch_step1',
            'num_epoch_step2',
            'learning_rate_step1',
            'learning_rate_step2',
            'skip_step_one',
            'load_step1_checkpoint',
            'stepone_checkpoint',
            'use_random_init',
            'seed'
        ):
            value = getattr(parsed_args, key)
            if value is not None:
                config[key] = value
        experiments = _load_experiments_from_args(parsed_args)
        if experiments is not None:
            config['experiments'] = experiments

        return config

    if case == 'eval_decoder':
        parser = _build_eval_decoder_parser()
        parsed_args = parser.parse_args()
        config = copy.deepcopy(DEFAULT_EVAL_DECODER_CONFIG)

        for key in (
            'search_paths_json_path',
            'include_regex',
            'ckpt_list_output_json_path',
            'test_input',
            'cuda',
            'seed',
            'batch_size',
        ):
            value = getattr(parsed_args, key)
            if value is not None:
                config[key] = value

        if parsed_args.exclude_regex_list is not None:
            config['exclude_regex_list'] = parsed_args.exclude_regex_list

        if not config['search_paths_json_path']:
            raise ValueError('eval_decoder 需要提供 --search_paths_json_path')

        return config

    raise ValueError(f'不支持的配置类型: {case}')