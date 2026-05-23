echo "本脚本将使用 ZuCo 数据集训练解码器模型。"
echo "训练分为两个阶段：第一阶段训练编码器，第二阶段训练解码器。"
echo "请确保已经运行 scripts/prepare_dataset.sh 来准备数据集，并且已经安装了所需的 Python 包。"
echo "注意：训练过程可能耗时较长，尤其是在使用较大模型和完整数据集时，请耐心等待..."

python3 train_decoder4.py --task_name task1_task2_task3 \
    --save_dir ./experiments/decoder \
    --cuda cuda:1 \
    --subjects ALL \
    --eeg_type GD \
    --batch_size 16 \
    --num_epoch_step1 30 \
    --num_epoch_step2 15 \
    --learning_rate_step1 0.00004 \
    --learning_rate_step2 0.00001 \
    --seed 12 \
    --experiments_json_path ./experiments/train_enh_decoder_experiments.json