import os
import json
from fuzzy_match import algorithims


class SentimentDatasetBuilder:
    """用于构建 ZuCo 情感标签与 SST 三分类数据集。"""

    def __init__(self, base_dir='.'):
        self.base_dir = base_dir

    def build_zuco_sentiment_labels(self, sentiment_labels_task1_csv_path=None, output_path=None):
        """
        从 ZuCo task1 的 csv 构建句子到情感标签的映射。

        返回:
            dict: {sentence_text: label}
        """
        if sentiment_labels_task1_csv_path is None:
            sentiment_labels_task1_csv_path = os.path.join(
                self.base_dir,
                'dataset/ZuCo/task_materials/sentiment_labels_task1.csv',
            )
        if output_path is None:
            output_path = os.path.join(
                self.base_dir,
                'dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json',
            )

        print('#' * 50)
        print('开始生成 ZuCo task1-SR 情感标签...')

        sentiment_labels = {}
        with open(sentiment_labels_task1_csv_path, 'r') as file_obj:
            for line in file_obj:
                if line.startswith('sentence_id') or line.startswith('#'):
                    continue

                parsed_line = line.split(';')
                # 处理句子中包含分号且被引号包裹的情况。
                if '";' in line:
                    sent_text = line.split('";')[0].split('"')[1]
                else:
                    sent_text = parsed_line[1]

                label = int(parsed_line[-1].strip())
                sentiment_labels[sent_text] = label

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w') as out:
            json.dump(sentiment_labels, out, indent=4)

        print(f'写入文件: {output_path}')
        return sentiment_labels

    @staticmethod
    def get_sentiment_label_dict(sst_labels_file_path):
        """读取 SST 标签文件，返回 {phrase_id: sentiment_score}。"""
        ret_dict = {}
        with open(sst_labels_file_path) as file_obj:
            for line in file_obj:
                if line.startswith('phrase'):
                    continue

                phrase_id = int(line.split('|')[0])
                label = float(line.split('|')[1].strip())
                assert phrase_id not in ret_dict
                ret_dict[phrase_id] = label
        return ret_dict

    @staticmethod
    def get_phrasestr_phrase_dict(sst_dictionary_file_path):
        """读取 SST 字典文件，返回 {phrase_str: phrase_id}。"""
        ret_dict = {}
        with open(sst_dictionary_file_path) as file_obj:
            for line in file_obj:
                phrase_str = line.split('|')[0]
                phrase_id = int(line.split('|')[1].strip())
                assert phrase_str not in ret_dict
                ret_dict[phrase_str] = phrase_id
        return ret_dict

    @classmethod
    def get_sentence_label_dict(cls, sst_sentences_file_path, sst_labels_file_path, sst_dictionary_file_path):
        """
        构建 SST 句子标签字典。

        返回:
            tuple:
                sentence_2_label_all: {sentence: 原始分数(0-1)}
                sentence_2_label_ternary: {sentence: 三分类标签(0/1/2)}
        """
        phrase_id_to_label = cls.get_sentiment_label_dict(sst_labels_file_path)
        phrase_str_to_phrase_id = cls.get_phrasestr_phrase_dict(sst_dictionary_file_path)

        sentence_2_label_all = {}
        sentence_2_label_ternary = {}

        with open(sst_sentences_file_path) as file_obj:
            for line in file_obj:
                if line.startswith('sentence_index'):
                    continue

                parsed_line = line.split('\t')
                assert len(parsed_line) == 2
                sentence = parsed_line[1].strip()

                # 修复 SST 文本中的特殊标记与编码问题。
                sentence = sentence.replace('-LRB-', '(').replace('-RRB-', ')').replace('Ã©', 'é')
                if sentence not in phrase_str_to_phrase_id:
                    continue

                sent_phrase_id = phrase_str_to_phrase_id[sentence]
                label = phrase_id_to_label[sent_phrase_id]

                if sentence not in sentence_2_label_all:
                    sentence_2_label_all[sentence] = label

                if sentence not in sentence_2_label_ternary:
                    if label <= 0.2:
                        sentence_2_label_ternary[sentence] = 0
                    elif (label > 0.4) and (label <= 0.6):
                        sentence_2_label_ternary[sentence] = 1
                    elif label > 0.8:
                        sentence_2_label_ternary[sentence] = 2

        return sentence_2_label_all, sentence_2_label_ternary

    def build_sst_ternary_dataset(self, sst_dir_path, zuco_sentiment_labels, output_path=None, match_dump_path=None):
        """
        过滤掉与 ZuCo 已使用句子相近的样本，生成 SST 三分类数据集。

        参数:
            sst_dir_path: SST 数据目录。
            zuco_sentiment_labels: ZuCo 的句子标签字典，键为句子文本。
            output_path: 过滤后数据集输出路径。
            match_dump_path: 匹配样本调试输出路径。

        返回:
            dict: 过滤后的三分类数据集。
        """
        sst_sentences_file_path = os.path.join(sst_dir_path, 'datasetSentences.txt')
        if not os.path.isfile(sst_sentences_file_path):
            print(f'未找到文件: {sst_sentences_file_path}')

        sst_labels_file_path = os.path.join(sst_dir_path, 'sentiment_labels.txt')
        if not os.path.isfile(sst_labels_file_path):
            print(f'未找到文件: {sst_labels_file_path}')

        sst_dictionary_file_path = os.path.join(sst_dir_path, 'dictionary.txt')
        if not os.path.isfile(sst_dictionary_file_path):
            print(f'未找到文件: {sst_dictionary_file_path}')

        if output_path is None:
            output_path = os.path.join(self.base_dir, 'dataset/stanfordsentiment/ternary_dataset.json')
        if match_dump_path is None:
            match_dump_path = os.path.join(self.base_dir, 'temp.txt')

        _, sentence_2_label_ternary = self.get_sentence_label_dict(
            sst_sentences_file_path,
            sst_labels_file_path,
            sst_dictionary_file_path,
        )
        print('原始三分类数据量:', len(sentence_2_label_ternary))

        zuco_used_sentences = list(zuco_sentiment_labels)
        filtered_ternary_dataset = {}
        filtered_pairs = []

        for key, value in sentence_2_label_ternary.items():
            add_instance = True
            for used_sent in zuco_used_sentences:
                if algorithims.trigram(used_sent, key) > 0.7:
                    filtered_pairs.append((used_sent, key))
                    zuco_used_sentences.remove(used_sent)
                    add_instance = False
                    break
            if add_instance:
                filtered_ternary_dataset[key] = value

        print('被过滤样本数:', len(filtered_pairs))
        print('过滤后三分类数据量:', len(filtered_ternary_dataset))
        print('未匹配剩余句子:', zuco_used_sentences)
        print('未匹配剩余句子数量:', len(zuco_used_sentences))

        with open(match_dump_path, 'w') as temp:
            for matched_pair in filtered_pairs:
                temp.write('#######\n')
                temp.write('\t' + matched_pair[0] + '\n')
                temp.write('\t' + matched_pair[1] + '\n')
                temp.write('\n')

        output_dir = os.path.dirname(output_path)
        if output_dir and (not os.path.exists(output_dir)):
            os.makedirs(output_dir)

        with open(output_path, 'w') as out:
            json.dump(filtered_ternary_dataset, out, indent=4)
        print(f'写入文件: {output_path}')

        return filtered_ternary_dataset