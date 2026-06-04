# SG-SPD 中文说明

[English README](README_en.md)

SG-SPD 是一个面向 ZuCo 数据集的 EEG-to-Text 解码框架。项目支持普通 encoder-decoder baseline，也支持完整的 SG-SPD 增强分支，包括 语义引导的EEG encoder、语义属性 prompt、两阶段训练、LoRA 阶段二微调，以及 EEG/noise 对照评估。

同一套 `train_decoder.py` 和 `eval_decoder.py` 可以通过修改实验 JSON 来运行简单 baseline 或完整 SG-SPD 配置。

## 目录结构

```text
.
+-- dataset/
|   +-- ZuCo/
|       +-- task_materials/              # sentiment / relation 等元信息
|       +-- zuco_label_8variants.df      # 多文本变体标签，可选
|       +-- zuco_semkey_labels.df        # SG-SPD 使用的语义属性标签
+-- experiments/
|   +-- train_decoder_experiments.json   # 训练实验配置
+-- metrics/                             # 本地 evaluate 指标，脚本下载生成
+-- model/                               # encoder / decoder / SG-SPD 模块
+-- pretrained_models/                   # 本地预训练模型，脚本下载生成
+-- results/
|   +-- eval_decoder_search_paths.json   # 评估 checkpoint 搜索路径
+-- scripts/                             # 常用入口脚本
+-- train_decoder.py
+-- eval_decoder.py
```

## 1. 创建环境

```bash
conda env create -f environment.yml
conda activate EEGToText
```

如果你的 CUDA 或 PyTorch 版本不同，可以先安装适配自己机器的 PyTorch，再安装 `environment.yml` 中其余依赖。

项目脚本默认是 Linux / WSL / Git Bash 风格。在 Windows PowerShell 中，可以直接运行脚本里的 `python3 ...` 命令。

## 2. 下载预训练模型

预训练模型注册在 `utils/registry.py` 中，当前支持：

```text
T5, Flan-T5, Bart, Pegasus
```

下载全部注册的主干模型和语义属性模型：

```bash
bash scripts/download_pretrained.sh
```

模型会保存到：

```text
./pretrained_models/
```

只下载当前实验配置实际用到的模型：

```bash
python3 utils/download_pretrained_models.py \
  --output_dir ./pretrained_models \
  --experiments_json_path ./experiments/train_decoder_experiments.json
```

如果不需要下载语义属性生成模型：

```bash
python3 utils/download_pretrained_models.py \
  --output_dir ./pretrained_models \
  --experiments_json_path ./experiments/train_decoder_experiments.json \
  --no_include_attribute_models
```

脚本默认在未设置 `HF_ENDPOINT` 时使用：

```text
https://hf-mirror.com
```

## 3. 下载评估指标

`eval_decoder.py` 会优先从本地加载 evaluate 指标。首次评估前运行：

```bash
bash scripts/download_eval_metrics.sh
```

指标会保存到：

```text
./metrics/
```

## 4. 准备 ZuCo 数据

本仓库不自动下载 ZuCo 原始数据。请根据 ZuCo 官方许可自行下载 `.mat` 文件，并放到以下目录：

```text
dataset/ZuCo/task1-SR/Matlab_files/*.mat
dataset/ZuCo/task2-NR/Matlab_files/*.mat
dataset/ZuCo/task3-TSR/Matlab_files/*.mat
dataset/ZuCo/task2-NR-2.0/Matlab_files/*.mat
```

然后运行：

```bash
bash scripts/prepare_dataset.sh
```

该脚本会把 `.mat` 转成训练使用的 `.pickle`：

```text
dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle
dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle
dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle
dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle
```

同时会生成 task1-SR 的情感标签：

```text
dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json
```

当前完整 SG-SPD 配置需要语义属性文件：

```text
dataset/ZuCo/zuco_semkey_labels.df
```

该文件已放在仓库中。如果你想重新生成它，可以运行：

```bash
bash scripts/prepare_labels.sh
```

其输入文件为：

```text
dataset/ZuCo/zuco_label_8variants.df
```

注意：`prepare_dataset.sh` 中还包含 Stanford Sentiment Treebank 三分类数据生成步骤。如果只运行当前 SG-SPD 配置，核心需要的是 ZuCo pickle 和 `zuco_semkey_labels.df`；SST 主要用于额外的情感数据工具。

## 5. 训练

推荐直接运行：

```bash
bash scripts/train_decoder.sh
```

等价的主要命令如下：

```bash
python3 train_decoder.py \
  --task_name task1_task2_task3 \
  --save_dir ./experiments/decoder \
  --cuda cuda:0 \
  --subjects ALL \
  --eeg_type GD \
  --batch_size 16 \
  --num_epoch_step1 30 \
  --num_epoch_step2 15 \
  --learning_rate_step1 0.00004 \
  --learning_rate_step2 0.00001 \
  --seed 12 \
  --experiments_json_path ./experiments/train_decoder_experiments.json
```

常用参数：

```text
--task_name      task1 / task1_task2 / task1_task2_task3 / task1_task2_taskNRv2
--subjects       ALL 或单个被试，如 ZAB
--eeg_type       GD / FFD / TRT
--eeg_bands      _t1 _t2 _a1 _a2 _b1 _b2 _g1 _g2
```

训练输出目录：

```text
experiments/decoder/{task_name}_{timestamp}/
+-- config.json
+-- experiment_metrics.xlsx
+-- {pretrained}_{encoder}_{train_input}_{enhancement}/
    +-- best/*.pt
    +-- last/*.pt
    +-- config/*.json
```

## 6. 配置训练实验

训练实验由 `experiments/train_decoder_experiments.json` 控制。每个实验至少包含：

```json
{
  "pretrained": "T5",
  "encoder": "Transformer",
  "train_input": "EEG"
}
```

### 简单 baseline 配置

```json
[
  {
    "pretrained": "T5",
    "encoder": "Transformer",
    "train_input": "EEG",
    "enhancement": {
      "name": "baseline"
    }
  }
]
```

可选 encoder 注册在 `utils/registry.py`：

```text
Transformer, MLP, CNN, LSTM, SemKeyStyle
```

例如单独测试 SemKeyStyle encoder：

```json
[
  {
    "pretrained": "T5",
    "encoder": "SemKeyStyle",
    "train_input": "EEG",
    "enhancement": {
      "name": "baseline"
    }
  }
]
```

### 完整 SG-SPD 配置

仓库默认的 `experiments/train_decoder_experiments.json` 已经写好复杂配置。核心字段如下：

```json
{
  "pretrained": "T5",
  "encoder": "SemKeyStyle",
  "train_input": "EEG",
  "enhancement": {
    "name": "neuroguide",
    "neuro_guide": {
      "enabled": true,
      "prompt_text": "System: Based on the following EEG signals, reconstruct the text.",
      "prompt_attributes": ["sentiment", "topic", "length"],
      "enabled_attribute_losses": ["sentiment", "topic", "length"],
      "attribute_path": "./dataset/ZuCo/zuco_semkey_labels.df",
      "require_semkey_attributes": true,
      "stage1_recon_weight": 0.2,
      "stage1_clip_weight": 1.0,
      "stage1_commitment_weight": 0.7,
      "stage1_attribute_weight": 1.0,
      "stage2_signal_grounding_weight": 0.5,
      "signal_grounding": {
        "enabled": true,
        "noise_types": ["random", "shuffle"]
      },
      "freeze_after_step1": true
    },
    "freeze_pretrained_step2": true,
    "stage2_freeze_strategy": "lora",
    "stage2_train_non_pretrained": true
  }
}
```

说明：

- `attribute_path`：语义属性标签路径。
- `prompt_attributes`：哪些属性会写入 Stage2 decoder prompt。
- `enabled_attribute_losses`：Stage1 训练哪些属性头。
- `stage2_freeze_strategy: "lora"`：冻结 LLM 主体，训练 LoRA 与非预训练模块。
- `signal_grounding`：开启 EEG/noise 对照约束。

## 7. 多文本变体 MTV

训练脚本支持 Multiple Text Variants。默认配置中关闭：

```json
"mtv": {
  "enabled": false,
  "path": "./dataset/ZuCo/zuco_label_8variants.df",
  "include_original": false
}
```

开启示例：

```json
"mtv": {
  "enabled": true,
  "path": "./dataset/ZuCo/zuco_label_8variants.df",
  "include_original": true
}
```

也可以通过命令行开启：

```bash
python3 train_decoder.py \
  --use_mtv \
  --mtv_path ./dataset/ZuCo/zuco_label_8variants.df \
  --mtv_include_original \
  --experiments_json_path ./experiments/train_decoder_experiments.json
```

## 8. 评估

先编辑：

```text
results/eval_decoder_search_paths.json
```

填入需要搜索 checkpoint 的实验目录，例如：

```json
[
  "./experiments/decoder/task1_task2_task3_20260604-120000"
]
```

然后运行：

```bash
bash scripts/eval_decoder.sh
```

该脚本会分别评估：

```text
--test_input EEG
--test_input noise
```

默认 checkpoint 过滤规则：

```text
include: .*\.pt$
exclude: /last/
```

即通常只评估 `best/*.pt`，跳过 `last/*.pt`。

手动评估命令：

```bash
python3 eval_decoder.py \
  --search_paths_json_path ./results/eval_decoder_search_paths.json \
  --include_regex ".*\\.pt$" \
  --exclude_regex_list "/last/" \
  --ckpt_list_output_json_path ./results/eval_decoder_checkpoints.json \
  --test_input EEG \
  --cuda cuda:0 \
  --batch_size 1 \
  --seed 12
```

评估结果输出到：

```text
results/eval_decoder/{run_name}/results.xlsx
```

Excel 中包含：

- checkpoint 信息
- BLEU / SacreBLEU / ROUGE-1 / WER / CER
- 非 teacher-forcing 生成结果
- teacher-forcing 解码结果

## 9. 推荐完整流程

```bash
# 1. 创建环境
conda env create -f environment.yml
conda activate EEGToText

# 2. 下载预训练模型和评估指标
bash scripts/download_pretrained.sh
bash scripts/download_eval_metrics.sh

# 3. 手动放置 ZuCo 原始 .mat 文件到 dataset/ZuCo/*/Matlab_files/

# 4. 转换数据
bash scripts/prepare_dataset.sh

# 5. 可选：重新生成语义属性标签
bash scripts/prepare_labels.sh

# 6. 训练
bash scripts/train_decoder.sh

# 7. 将实验目录写入 results/eval_decoder_search_paths.json

# 8. 评估 EEG 和 noise
bash scripts/eval_decoder.sh
```

## 10. 常见问题

### 找不到 `Matlab_files`

说明 ZuCo 原始 `.mat` 文件没有放到指定位置。请检查：

```text
dataset/ZuCo/task1-SR/Matlab_files/
dataset/ZuCo/task2-NR/Matlab_files/
dataset/ZuCo/task3-TSR/Matlab_files/
dataset/ZuCo/task2-NR-2.0/Matlab_files/
```

### 找不到 `./pretrained_models/T5`

运行：

```bash
bash scripts/download_pretrained.sh
```

或者手动把 Hugging Face 模型文件放到 `utils/registry.py` 中定义的 `local_model_path`。

### 评估指标无法加载

运行：

```bash
bash scripts/download_eval_metrics.sh
```

### SG-SPD 训练提示缺少属性标签

检查：

```text
dataset/ZuCo/zuco_semkey_labels.df
```

缺失时运行：

```bash
bash scripts/prepare_labels.sh
```

### 快速 sanity test

```bash
python3 train_decoder.py \
  --task_name task1 \
  --cuda cuda:0 \
  --batch_size 4 \
  --num_epoch_step1 1 \
  --num_epoch_step2 1 \
  --experiments_json '[{"pretrained":"T5","encoder":"Transformer","train_input":"EEG","enhancement":{"name":"baseline"}}]'
```
