echo "本脚本将 ZuCo 数据集中的 .mat 文件转换为 .pickle 文件。"
echo "本脚本还会生成 ZuCo task1-SR v1.0 的三分类 sentiment_labels.json，以及过滤后的 StanfordSentimentTreebank 三分类数据 ternary_dataset.json。"
echo "注意：ZuCo task1-SR 中的句子与过滤后的 StanfordSentimentTreebank 句子不重叠。"
echo "注意：该过程可能耗时较长，请耐心等待..."

python3 ./utils/prepare_dataset.py