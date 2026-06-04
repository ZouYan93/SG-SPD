echo "本脚本为每条数据准备对应标签"
echo "请确保运行本脚本前，准备了相应的预训练模型"

set -euo pipefail

cd "$(dirname "$0")/.."

python3 ./utils/data/generate_semkey_attributes.py \
  --attribute_model_profile SemKey \
  --input_df ./dataset/ZuCo/zuco_label_8variants.df \
  --output_df ./dataset/ZuCo/zuco_semkey_labels.df \
  --cache_dir ./pretrained_models/hf_cache
