echo "本脚本将下载 eval_decoder 依赖的 evaluate 指标到本地目录。"

if [ -z "$HF_ENDPOINT" ]; then
	export HF_ENDPOINT="https://hf-mirror.com"
fi
echo "[信息]当前 HF_ENDPOINT=$HF_ENDPOINT"

python3 utils/download_eval_metrics.py --output_dir ./metrics
