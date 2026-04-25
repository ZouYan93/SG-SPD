import argparse
import json
import os
from typing import Dict

from huggingface_hub import snapshot_download

METRIC_REPOS = {
    "sacrebleu": "evaluate-metric/sacrebleu",
    "cer": "evaluate-metric/cer",
    "wer": "evaluate-metric/wer",
}


def download_metrics(output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    local_path_map: Dict[str, str] = {}
    for metric_name, repo_id in METRIC_REPOS.items():
        save_dir = os.path.join(output_dir, metric_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[下载] {metric_name}: {repo_id}")
        local_snapshot_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="space",
            local_dir=save_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        local_path_map[metric_name] = local_snapshot_dir
        print(f"[完成] {metric_name} -> {local_snapshot_dir}")

    return local_path_map


def save_manifest(local_path_map: Dict[str, str], output_dir: str) -> str:
    manifest = {
        "metrics": [],
    }

    for name, local_path in local_path_map.items():
        manifest["metrics"].append(
            {
                "name": name,
                "local_metric_path": local_path,
            }
        )

    manifest_path = os.path.join(output_dir, "local_metrics_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="预下载 eval_decoder 用到的 evaluate 指标")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./metrics",
        help="指标下载目录",
    )
    args = parser.parse_args()

    local_path_map = download_metrics(args.output_dir)
    manifest_path = save_manifest(local_path_map, args.output_dir)
    print(f"[信息]已生成本地映射清单: {manifest_path}")


if __name__ == "__main__":
    main()
