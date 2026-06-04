# SG-SPD

[中文说明](README.md)

SG-SPD is an EEG-to-text decoding framework for the ZuCo dataset. It supports standard encoder-decoder baselines and a full SG-SPD enhancement branch with a Signal-Guided EEG encoder, semantic attribute prompts, two-stage training, LoRA-based stage-2 tuning, and EEG/noise grounding evaluation.

The same `train_decoder.py` and `eval_decoder.py` scripts can run either simple baselines or the full SG-SPD configuration by changing the experiment JSON.

## Project Structure

```text
.
+-- dataset/
|   +-- ZuCo/
|       +-- task_materials/              # sentiment/relation metadata
|       +-- zuco_label_8variants.df      # optional multi-text-variant labels
|       +-- zuco_semkey_labels.df        # semantic attributes for SG-SPD
+-- experiments/
|   +-- train_decoder_experiments.json   # training experiment list
+-- metrics/                             # local evaluate metrics
+-- model/                               # encoder, decoder, SG-SPD modules
+-- pretrained_models/                   # local Hugging Face models
+-- results/
|   +-- eval_decoder_search_paths.json   # checkpoint search paths
+-- scripts/
+-- train_decoder.py
+-- eval_decoder.py
```

## Environment

```bash
conda env create -f environment.yml
conda activate EEGToText
```

The shell scripts are intended for Linux, WSL, or Git Bash. On Windows PowerShell, run the underlying `python3 ...` commands directly.

## Download Models and Metrics

Download pretrained models:

```bash
bash scripts/download_pretrained.sh
```

Download local evaluation metrics:

```bash
bash scripts/download_eval_metrics.sh
```

Models are saved in `./pretrained_models/`; metrics are saved in `./metrics/`.

To download only models required by one experiment JSON:

```bash
python3 utils/download_pretrained_models.py \
  --output_dir ./pretrained_models \
  --experiments_json_path ./experiments/train_decoder_experiments.json
```

To skip semantic attribute generation models:

```bash
python3 utils/download_pretrained_models.py \
  --output_dir ./pretrained_models \
  --experiments_json_path ./experiments/train_decoder_experiments.json \
  --no_include_attribute_models
```

## Prepare ZuCo Data

ZuCo raw `.mat` files are not downloaded automatically. Download them from the official source according to the ZuCo license and place them as:

```text
dataset/ZuCo/task1-SR/Matlab_files/*.mat
dataset/ZuCo/task2-NR/Matlab_files/*.mat
dataset/ZuCo/task3-TSR/Matlab_files/*.mat
dataset/ZuCo/task2-NR-2.0/Matlab_files/*.mat
```

Then convert data:

```bash
bash scripts/prepare_dataset.sh
```

This generates:

```text
dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle
dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle
dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle
dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle
dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json
```

The default SG-SPD configuration expects:

```text
dataset/ZuCo/zuco_semkey_labels.df
```

It is included in the repository. To regenerate it:

```bash
bash scripts/prepare_labels.sh
```

## Training

Recommended:

```bash
bash scripts/train_decoder.sh
```

Main command:

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

Common task and EEG options:

```text
--task_name      task1 / task1_task2 / task1_task2_task3 / task1_task2_taskNRv2
--subjects       ALL or one subject ID, e.g. ZAB
--eeg_type       GD / FFD / TRT
--eeg_bands      _t1 _t2 _a1 _a2 _b1 _b2 _g1 _g2
```

Training outputs:

```text
experiments/decoder/{task_name}_{timestamp}/
+-- config.json
+-- experiment_metrics.xlsx
+-- {pretrained}_{encoder}_{train_input}_{enhancement}/
    +-- best/*.pt
    +-- last/*.pt
    +-- config/*.json
```

## Configure Experiments

Experiments are configured in:

```text
experiments/train_decoder_experiments.json
```

A minimal baseline:

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

Available encoders are registered in `utils/registry.py`:

```text
Transformer, MLP, CNN, LSTM, SemKeyStyle
```

Full SG-SPD settings are already provided in `experiments/train_decoder_experiments.json`. Key options:

- `attribute_path`: semantic attribute label file.
- `prompt_attributes`: attributes inserted into the decoder prompt.
- `enabled_attribute_losses`: attribute heads trained in stage 1.
- `stage2_freeze_strategy: "lora"`: freeze the LLM backbone and train LoRA plus non-pretrained modules.
- `signal_grounding`: enable EEG/noise grounding losses.

## Multiple Text Variants

The training script supports optional Multiple Text Variants:

```json
"mtv": {
  "enabled": true,
  "path": "./dataset/ZuCo/zuco_label_8variants.df",
  "include_original": true
}
```

Command-line example:

```bash
python3 train_decoder.py \
  --use_mtv \
  --mtv_path ./dataset/ZuCo/zuco_label_8variants.df \
  --mtv_include_original \
  --experiments_json_path ./experiments/train_decoder_experiments.json
```

## Evaluation

Edit:

```text
results/eval_decoder_search_paths.json
```

Example:

```json
[
  "./experiments/decoder/task1_task2_task3_20260604-120000"
]
```

Run:

```bash
bash scripts/eval_decoder.sh
```

Manual command:

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

Results are written to:

```text
results/eval_decoder/{run_name}/results.xlsx
```

The evaluation workbook includes checkpoint summaries, BLEU, SacreBLEU, ROUGE-1, WER, CER, generated text, and teacher-forcing decoded text.

## Recommended Workflow

```bash
conda env create -f environment.yml
conda activate EEGToText

bash scripts/download_pretrained.sh
bash scripts/download_eval_metrics.sh

# Place ZuCo raw .mat files under dataset/ZuCo/*/Matlab_files/
bash scripts/prepare_dataset.sh

# Optional
bash scripts/prepare_labels.sh

bash scripts/train_decoder.sh

# Edit results/eval_decoder_search_paths.json
bash scripts/eval_decoder.sh
```

## Troubleshooting

### Missing `Matlab_files`

Place ZuCo raw `.mat` files under:

```text
dataset/ZuCo/task1-SR/Matlab_files/
dataset/ZuCo/task2-NR/Matlab_files/
dataset/ZuCo/task3-TSR/Matlab_files/
dataset/ZuCo/task2-NR-2.0/Matlab_files/
```

### Missing `./pretrained_models/T5`

Run:

```bash
bash scripts/download_pretrained.sh
```

### Evaluation metrics cannot be loaded

Run:

```bash
bash scripts/download_eval_metrics.sh
```

### SG-SPD training complains about missing attributes

Check:

```text
dataset/ZuCo/zuco_semkey_labels.df
```

If it is missing, regenerate it:

```bash
bash scripts/prepare_labels.sh
```

## Citation

If this repository is used for research, please cite the related EEG-to-text and ZuCo resources, and add SG-SPD citation information when available.
