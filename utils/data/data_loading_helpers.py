import numpy as np
import re

class DataLoadingHelper:
    """用于从 Matlab/h5py 数据结构中提取词级别 EEG 与眼动特征。"""

    eeg_float_resolution = np.float16

    Alpha_ffd_names = ['FFD_a1', 'FFD_a1_diff', 'FFD_a2', 'FFD_a2_diff']
    Beta_ffd_names  = ['FFD_b1', 'FFD_b1_diff', 'FFD_b2', 'FFD_b2_diff']
    Gamma_ffd_names = ['FFD_g1', 'FFD_g1_diff', 'FFD_g2', 'FFD_g2_diff']
    Theta_ffd_names = ['FFD_t1', 'FFD_t1_diff', 'FFD_t2', 'FFD_t2_diff']

    Alpha_gd_names  = ['GD_a1', 'GD_a1_diff', 'GD_a2', 'GD_a2_diff']
    Beta_gd_names   = ['GD_b1', 'GD_b1_diff', 'GD_b2', 'GD_b2_diff']
    Gamma_gd_names  = ['GD_g1', 'GD_g1_diff', 'GD_g2', 'GD_g2_diff']
    Theta_gd_names  = ['GD_t1', 'GD_t1_diff', 'GD_t2', 'GD_t2_diff']

    Alpha_gpt_names = ['GPT_a1', 'GPT_a1_diff', 'GPT_a2', 'GPT_a2_diff']
    Beta_gpt_names  = ['GPT_b1', 'GPT_b1_diff', 'GPT_b2', 'GPT_b2_diff']
    Gamma_gpt_names = ['GPT_g1', 'GPT_g1_diff', 'GPT_g2', 'GPT_g2_diff']
    Theta_gpt_names = ['GPT_t1', 'GPT_t1_diff', 'GPT_t2', 'GPT_t2_diff']

    Alpha_sfd_names = ['SFD_a1', 'SFD_a1_diff', 'SFD_a2', 'SFD_a2_diff']
    Beta_sfd_names  = ['SFD_b1', 'SFD_b1_diff', 'SFD_b2', 'SFD_b2_diff']
    Gamma_sfd_names = ['SFD_g1', 'SFD_g1_diff', 'SFD_g2', 'SFD_g2_diff']
    Theta_sfd_names = ['SFD_t1', 'SFD_t1_diff', 'SFD_t2', 'SFD_t2_diff']

    Alpha_trt_names = ['TRT_a1', 'TRT_a1_diff', 'TRT_a2', 'TRT_a2_diff']
    Beta_trt_names  = ['TRT_b1', 'TRT_b1_diff', 'TRT_b2', 'TRT_b2_diff']
    Gamma_trt_names = ['TRT_g1', 'TRT_g1_diff', 'TRT_g2', 'TRT_g2_diff']
    Theta_trt_names = ['TRT_t1', 'TRT_t1_diff', 'TRT_t2', 'TRT_t2_diff']

    Alpha_features  = Alpha_ffd_names + Alpha_gd_names + Alpha_gpt_names + Alpha_trt_names
    Beta_features   = Beta_ffd_names  + Beta_gd_names  + Beta_gpt_names  + Beta_trt_names
    Gamma_features  = Gamma_ffd_names + Gamma_gd_names + Gamma_gpt_names + Gamma_trt_names
    Theta_features  = Theta_ffd_names + Theta_gd_names + Theta_gpt_names + Theta_trt_names

    @staticmethod
    def extract_all_fixations(data_container, word_data_object, float_resolution=np.float16):
        """
        从单词对象中提取所有 fixation 数据。

        参数:
            data_container: h5py 总数据容器。
            word_data_object: 当前单词对应的 fixation 对象。
            float_resolution: 转换后的浮点精度，用于压缩存储。

        返回:
            list: 每次 fixation 的数组列表。
        """
        word_data = data_container[word_data_object]
        fixations_data = []
        if len(word_data.shape) > 1:
            for fixation_idx in range(word_data.shape[0]):
                fixations_data.append(
                    np.array(data_container[word_data[fixation_idx][0]]).astype(float_resolution)
                )
        return fixations_data

    @staticmethod
    def is_real_word(word):
        """
        判断字符串是否是有效词（包含字母或数字）。

        参数:
            word: 输入字符串。

        返回:
            bool: 是有效词返回 True，否则返回 False。
        """
        is_word = re.search('[a-zA-Z0-9]', word)
        return is_word

    @staticmethod
    def load_matlab_string(matlab_extracted_object):
        """
        将 h5py 读取出的 Matlab 字符串对象转换为 Python 字符串。

        参数:
            matlab_extracted_object: Matlab 字符串对象。

        返回:
            str: 转换后的 Python 字符串。
        """
        extracted_string = u''.join(chr(c[0]) for c in matlab_extracted_object)
        return extracted_string

    @classmethod
    def extract_word_level_data(cls, data_container, word_objects, eeg_float_resolution=np.float16):
        """
        提取某个句子的词级别数据。

        参数:
            data_container: h5py 总数据容器。
            word_objects: 该句子内所有单词的数据容器。
            eeg_float_resolution: EEG 保存精度，用于压缩存储。

        返回:
            tuple:
                word_level_data (dict): 词索引到词级数据的映射。
                word_tokens_all (list): 句子中所有有效词。
                word_tokens_has_fixation (list): 有 fixation 的词。
                word_tokens_with_mask (list): 无 fixation 的词替换为 [MASK]。
        """
        available_objects = list(word_objects)

        if isinstance(available_objects[0], str):
            contentData = word_objects['content']

            if "rawEEG" in available_objects:
                rawData = word_objects['rawEEG']
                etData = word_objects['rawET']

                ffdData = word_objects['FFD']
                gdData = word_objects['GD']
                gptData = word_objects['GPT']
                trtData = word_objects['TRT']

                try:
                    sfdData = word_objects['SFD']
                except KeyError:
                    print("未找到 SFD 字段。")
                    sfdData = []

                nFixData = word_objects['nFixations']
                fixPositions = word_objects["fixPositions"]

                Alpha_features_data = [word_objects[feature] for feature in cls.Alpha_features]
                Beta_features_data  = [word_objects[feature] for feature in cls.Beta_features]
                Gamma_features_data = [word_objects[feature] for feature in cls.Gamma_features]
                Theta_features_data = [word_objects[feature] for feature in cls.Theta_features]

                GD_EEG_features = [
                    word_objects[feature]
                    for feature in ['GD_t1', 'GD_t2', 'GD_a1', 'GD_a2', 'GD_b1', 'GD_b2', 'GD_g1', 'GD_g2']
                ]
                FFD_EEG_features = [
                    word_objects[feature]
                    for feature in ['FFD_t1', 'FFD_t2', 'FFD_a1', 'FFD_a2', 'FFD_b1', 'FFD_b2', 'FFD_g1', 'FFD_g2']
                ]
                TRT_EEG_features = [
                    word_objects[feature]
                    for feature in ['TRT_t1', 'TRT_t2', 'TRT_a1', 'TRT_a2', 'TRT_b1', 'TRT_b2', 'TRT_g1', 'TRT_g2']
                ]

                assert len(contentData) == len(etData) == len(rawData), "不同数据项的长度不一致。"

                zipped_data = zip(
                    rawData,
                    etData,
                    contentData,
                    ffdData,
                    gdData,
                    gptData,
                    trtData,
                    sfdData,
                    nFixData,
                    fixPositions,
                )

                word_level_data = {}
                word_idx = 0

                word_tokens_has_fixation = []
                word_tokens_with_mask = []
                word_tokens_all = []

                for raw_eegs_obj, ets_obj, word_obj, ffd, gd, gpt, trt, sfd, nFix, fixPos in zipped_data:
                    word_string = cls.load_matlab_string(data_container[word_obj[0]])
                    if cls.is_real_word(word_string):
                        data_dict = {}
                        data_dict["RAW_EEG"] = cls.extract_all_fixations(
                            data_container,
                            raw_eegs_obj[0],
                            eeg_float_resolution,
                        )
                        data_dict["RAW_ET"] = cls.extract_all_fixations(data_container, ets_obj[0], np.float32)

                        data_dict["FFD"]  = data_container[ffd[0]][()][0, 0] if len(data_container[ffd[0]][()].shape) == 2 else None
                        data_dict["GD"]   = data_container[gd[0]][()][0, 0] if len(data_container[gd[0]][()].shape) == 2 else None
                        data_dict["GPT"]  = data_container[gpt[0]][()][0, 0] if len(data_container[gpt[0]][()].shape) == 2 else None
                        data_dict["TRT"]  = data_container[trt[0]][()][0, 0] if len(data_container[trt[0]][()].shape) == 2 else None
                        data_dict["SFD"]  = data_container[sfd[0]][()][0, 0] if len(data_container[sfd[0]][()].shape) == 2 else None
                        data_dict["nFix"] = data_container[nFix[0]][()][0, 0] if len(data_container[nFix[0]][()].shape) == 2 else None

                        data_dict["ALPHA_EEG"] = np.concatenate(
                            [
                                data_container[obj[word_idx][0]][()]
                                if len(data_container[obj[word_idx][0]][()].shape) == 2 else []
                                for obj in Alpha_features_data
                            ],
                            0,
                        )

                        data_dict["BETA_EEG"] = np.concatenate(
                            [
                                data_container[obj[word_idx][0]][()]
                                if len(data_container[obj[word_idx][0]][()].shape) == 2 else []
                                for obj in Beta_features_data
                            ],
                            0,
                        )

                        data_dict["GAMMA_EEG"] = np.concatenate(
                            [
                                data_container[obj[word_idx][0]][()]
                                if len(data_container[obj[word_idx][0]][()].shape) == 2 else []
                                for obj in Gamma_features_data
                            ],
                            0,
                        )

                        data_dict["THETA_EEG"] = np.concatenate(
                            [
                                data_container[obj[word_idx][0]][()]
                                if len(data_container[obj[word_idx][0]][()].shape) == 2 else []
                                for obj in Theta_features_data
                            ],
                            0,
                        )

                        data_dict["word_idx"] = word_idx
                        data_dict["content"] = word_string

                        word_tokens_all.append(word_string)
                        if data_dict["nFix"] is not None:
                            data_dict["GD_EEG"] = [
                                np.squeeze(data_container[obj[word_idx][0]][()])
                                if len(data_container[obj[word_idx][0]][()].shape) == 2 else []
                                for obj in GD_EEG_features
                            ]
                            data_dict["FFD_EEG"] = [
                                np.squeeze(data_container[obj[word_idx][0]][()])
                                if len(data_container[obj[word_idx][0]][()].shape) == 2 else []
                                for obj in FFD_EEG_features
                            ]
                            data_dict["TRT_EEG"] = [
                                np.squeeze(data_container[obj[word_idx][0]][()])
                                if len(data_container[obj[word_idx][0]][()].shape) == 2 else []
                                for obj in TRT_EEG_features
                            ]
                            word_tokens_has_fixation.append(word_string)
                            word_tokens_with_mask.append(word_string)
                        else:
                            word_tokens_with_mask.append('[MASK]')

                        word_level_data[word_idx] = data_dict
                        word_idx += 1
                    else:
                        print(word_string + " 不是有效词。")
            else:
                # 如果没有词级数据，则只返回词嵌入相关结构。
                word_level_data = {}
                word_idx = 0
                word_tokens_has_fixation = []
                word_tokens_with_mask = []
                word_tokens_all = []

                for word_obj in contentData:
                    word_string = cls.load_matlab_string(data_container[word_obj[0]])
                    if cls.is_real_word(word_string):
                        data_dict = {}
                        data_dict["RAW_EEG"] = []
                        data_dict["ICA_EEG"] = []
                        data_dict["RAW_ET"] = []
                        data_dict["FFD"] = None
                        data_dict["GD"] = None
                        data_dict["GPT"] = None
                        data_dict["TRT"] = None
                        data_dict["SFD"] = None
                        data_dict["nFix"] = None
                        data_dict["ALPHA_EEG"] = []
                        data_dict["BETA_EEG"] = []
                        data_dict["GAMMA_EEG"] = []
                        data_dict["THETA_EEG"] = []

                        data_dict["word_idx"] = word_idx
                        data_dict["content"] = word_string
                        word_level_data[word_idx] = data_dict
                        word_idx += 1
                    else:
                        print(word_string + " 不是有效词。")

        else:
            word_tokens_has_fixation = []
            word_tokens_with_mask = []
            word_tokens_all = []
            word_level_data = {}

        return word_level_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask