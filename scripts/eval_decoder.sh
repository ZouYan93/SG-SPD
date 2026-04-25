echo "本脚本用于评估解码器模型。支持从 JSON 读取搜索路径并结合正则自动收集 ckpt。"

python3 eval_decoder.py \
    --search_paths_json_path ./results/eval_decoder_search_paths.json \
    --include_regex ".*\\.pt$" \
    --exclude_regex_list "/last/" \
    --ckpt_list_output_json_path ./results/eval_decoder_checkpoints.json \
    --test_input EEG \
    --cuda cuda:0 \
    --batch_size 1 \
    --seed 12