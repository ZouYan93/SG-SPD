import json
from convert_mat_to_pickle import MatToPickleConverter
from sentiment_dataset_builder import SentimentDatasetBuilder

if __name__ == '__main__':
    """将 ZuCo mat 数据转换为 pickle。"""
    print('#' * 50)
    print('开始执行 mat -> pickle 转换...')

    converter = MatToPickleConverter(base_dir='.')
    converter.convert(task_name='task1-SR', version='v1')
    converter.convert(task_name='task2-NR', version='v1')
    converter.convert(task_name='task3-TSR', version='v1')
    converter.convert(task_name='task2-NR', version='v2')
    

    """使用SentimentDatasetBuilder类生成 ZuCo task1-SR 情感标签。"""
    print('#' * 50)
    print('开始生成 ZuCo task1-SR 情感标签...')

    # 初始化SentimentDatasetBuilder实例
    builder = SentimentDatasetBuilder(base_dir='.')
    
    # 路径
    sentiment_labels_task1_csv_path = './dataset/ZuCo/task_materials/sentiment_labels_task1.csv'
    output_path = './dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'
    
    # 生成 ZuCo task1-SR 情感标签
    sentiment_labels = builder.build_zuco_sentiment_labels(
        sentiment_labels_task1_csv_path=sentiment_labels_task1_csv_path,
        output_path=output_path,
    )

    print(f'标签总数: {len(sentiment_labels)}')


    """生成 StanfordSentimentTreebank 三分类情感数据集。"""
    print('#' * 50)
    print('开始生成 StanfordSentimentTreebank 三分类情感数据集...')

    # 路径
    SST_dir_path = './dataset/stanfordsentiment/stanfordSentimentTreebank'
    ZUCO_SENTIMENT_LABELS = json.load(open('./dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
    output_path = './dataset/stanfordsentiment/ternary_dataset.json'

    # 生成 StanfordSentimentTreebank 三分类情感数据集
    builder.build_sst_ternary_dataset(SST_dir_path, ZUCO_SENTIMENT_LABELS, output_path=output_path)