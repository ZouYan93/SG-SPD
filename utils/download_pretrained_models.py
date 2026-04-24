import argparse
import json
import os
import sys
from typing import Dict, List
from huggingface_hub import snapshot_download

# 确保项目根目录在sys.path中，以便registry模块中导入model时能正确找到
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from registry import MODEL_REGISTRY


def load_target_pretrained_names(experiments_json_path: str = None) -> List[str]:
    if not experiments_json_path:
        return list(MODEL_REGISTRY.keys())

    with open(experiments_json_path, 'r', encoding='utf-8') as f:
        experiments = json.load(f)

    names = []
    for i, exp in enumerate(experiments):
        if not isinstance(exp, dict) or 'pretrained' not in exp:
            raise ValueError(f'experiments[{i}] 缺少 pretrained 字段')
        names.append(exp['pretrained'])

    # 去重并保持顺序
    seen = set()
    uniq_names = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq_names.append(n)
    return uniq_names


def download_models(pretrained_names: List[str], output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    local_path_map = {}
    for name in pretrained_names:
        if name not in MODEL_REGISTRY:
            raise ValueError(f'预训练模型 {name} 不在 MODEL_REGISTRY 中')

        model_id = MODEL_REGISTRY[name]['model_id']
        save_dir = os.path.join(output_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        print(f'[下载] {name}: {model_id}')
        local_snapshot_dir = snapshot_download(
            repo_id=model_id,
            local_dir=save_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        local_path_map[name] = local_snapshot_dir
        print(f'[完成] {name} -> {local_snapshot_dir}')

    return local_path_map


def save_manifest(local_path_map: Dict[str, str], output_dir: str) -> str:
    manifest = {
        'note': '将 utils/registry.py 里的 model_id 改为 local_model_path 即可离线加载',
        'models': [],
    }

    for name, local_path in local_path_map.items():
        manifest['models'].append(
            {
                'pretrained': name,
                'local_model_path': local_path,
            }
        )

    manifest_path = os.path.join(output_dir, 'local_model_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description='预下载当前项目用到的 Hugging Face 预训练模型到指定目录')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./pretrained_models',
        help='模型下载目录',
    )
    parser.add_argument(
        '--experiments_json_path',
        type=str,
        default=None,
        help='实验配置 JSON 路径；传入后仅下载配置里实际用到的 pretrained',
    )
    args = parser.parse_args()

    pretrained_names = load_target_pretrained_names(args.experiments_json_path)
    print(f'[信息]待下载模型: {pretrained_names}')

    local_path_map = download_models(pretrained_names, args.output_dir)
    manifest_path = save_manifest(local_path_map, args.output_dir)
    print(f'[信息]已生成本地映射清单: {manifest_path}')


if __name__ == '__main__':
    main()
