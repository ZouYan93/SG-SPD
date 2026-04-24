echo "本脚本将对预训练模型进行下载。"
echo "请确保已经安装了 huggingface_hub 包，并且有稳定的网络连接。"
echo "下载完成后，预训练模型将保存在 ./pretrained_models 目录下。"

if [ -z "$HF_ENDPOINT" ]; then
	export HF_ENDPOINT="https://hf-mirror.com"
fi
echo "[信息]当前 HF_ENDPOINT=$HF_ENDPOINT"

python3 utils/download_pretrained_models.py --output_dir ./pretrained_models