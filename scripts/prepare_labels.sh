set -euo pipefail

cd "$(dirname "$0")/.."

python3 ./utils/data/generate_semkey_attributes.py \
  --attribute_model_profile SemKey \
  --input_df ./dataset/ZuCo/zuco_label_8variants.df \
  --output_df ./dataset/ZuCo/zuco_semkey_labels.df \
  --cache_dir ./pretrained_models/hf_cache
