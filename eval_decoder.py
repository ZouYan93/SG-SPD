import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json
import time
from tqdm import tqdm
import time
from utils.data.data import ZuCo_dataset
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from evaluate import load
from model.decoder import Decoder
from utils.config import get_config
from utils.registry import MODEL_REGISTRY, ENCODER_REGISTRY, CLASS_MAP

metric = load("sacrebleu")
cer_metric = load("cer")
wer_metric = load("wer")

def remove_text_after_token(text, token='</s>'):
    token_index = text.find(token)
    if token_index != -1:
        return text[:token_index]
    return text

def eval_model(dataloaders, 
               device, 
               tokenizer, 
               criterion, 
               model, 
               output_all_results_path = './results/temp.txt' , 
               score_results='./score_results/task.txt'):
    start_time = time.time()
    model.eval()
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    pred_tokens_list_previous = []
    pred_string_list_previous = []
    with open(output_all_results_path,'w') as f:
        for inputs, seq_len, masks, invert_masks, target_ids, \
            target_mask, sentiment_labels in tqdm(dataloaders['test']):
            inputs_batch = inputs.to(device).float()
            masks_batch = masks.to(device)
            invert_masks_batch = invert_masks.to(device)
            target_ids_batch = target_ids.to(device)
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens = True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens = True)
            
            f.write(f'target string: {target_string}\n')

            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

            """Teacher-forcing 解码推理"""
            output = model(inputs_batch, masks_batch, invert_masks_batch, target_ids_batch)
            logits_previous = output.logits
            probs_previous = logits_previous[0].softmax(dim = 1)
            values_previous, predictions_previous = probs_previous.topk(1)
            predictions_previous = torch.squeeze(predictions_previous)
            predicted_string_previous = remove_text_after_token(tokenizer.decode(predictions_previous).split('</s></s>')[0].replace('<s>',''))
            f.write(f'predicted string with tf: {predicted_string_previous}\n')
            predictions_previous = predictions_previous.tolist()
            truncated_prediction_previous = []
            for t in predictions_previous:
                if t != tokenizer.eos_token_id:
                    truncated_prediction_previous.append(t)
                else:
                    break
            pred_tokens_previous = tokenizer.convert_ids_to_tokens(truncated_prediction_previous, skip_special_tokens = True)
            pred_tokens_list_previous.append(pred_tokens_previous)
            pred_string_list_previous.append(predicted_string_previous)
            

            """ Non-teacher-forcing 解码推理"""
            predictions=model.generate(inputs_batch, masks_batch, invert_masks_batch, target_ids_batch,
                                       max_length=60,
                                       num_beams=5,
                                       do_sample=True,
                                       repetition_penalty= 5.0,
                                       no_repeat_ngram_size = 2
                                       )
            
            predicted_string=tokenizer.batch_decode(predictions, skip_special_tokens=True)[0]
            predictions=tokenizer.encode(predicted_string)
            f.write(f'predicted string: {predicted_string}\n')
            f.write(f'#' * 50 + '\n\n\n')
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)

    """ 计算 BLEU-N 分数 """
    weights_list = [(1.0,),(0.5, 0.5),(1/3, 1/3, 1/3),(0.25, 0.25, 0.25, 0.25)]
    corpus_bleu_scores = []
    corpus_bleu_scores_previous = []
    for weight in weights_list:
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights = weight)
        corpus_bleu_score_previous = corpus_bleu(target_tokens_list, pred_tokens_list_previous, weights = weight)
        corpus_bleu_scores.append(corpus_bleu_score)
        corpus_bleu_scores_previous.append(corpus_bleu_score_previous)
        print(f'[信息]corpus BLEU-{len(list(weight))} 分数:', corpus_bleu_score)
        print(f'[信息]corpus BLEU-{len(list(weight))} 分数(teacher-forcing):', corpus_bleu_score_previous)


    """ 计算 SacreBLEU 分数 """
    reference_list = [[item] for item in target_string_list]
    sacre_blue = metric.compute(predictions=pred_string_list, references=reference_list)
    sacre_blue_previous = metric.compute(predictions=pred_string_list_previous, references=reference_list)
    print("[信息]SacreBLEU 分数: ", sacre_blue, '\n')
    print("[信息]SacreBLEU 分数(teacher-forcing): ", sacre_blue_previous)

    """ 计算 ROUGE 分数 """
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg = True, ignore_empty=True)
    except ValueError as e:
        rouge_scores = None

    try:
        rouge_scores_previous = rouge.get_scores(pred_string_list_previous, target_string_list, avg = True, ignore_empty=True)
    except ValueError as e:
        rouge_scores_previous = None

    """ 计算 WER 分数 """
    #wer = WordErrorRate()
    wer_scores = wer_metric.compute(predictions=pred_string_list, references=target_string_list)
    wer_scores_previous = wer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    print("[信息]WER 分数: ", wer_scores)
    print("[信息]WER 分数(teacher-forcing): ", wer_scores_previous)
    

    """ 计算 CER 分数 """
    cer_scores = cer_metric.compute(predictions=pred_string_list, references=target_string_list)
    cer_scores_previous = cer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    print("[信息]CER 分数: ", cer_scores)
    print("[信息]CER 分数(teacher-forcing): ", cer_scores_previous)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[信息]评估耗时 {total_time / 60} 分钟.")

     # score_results (only fix teacher-forcing)
    file_content = [
    f'corpus_bleu_score = {corpus_bleu_scores}',
    f'sacre_blue_score = {sacre_blue}',
    f'rouge_scores = {rouge_scores}',
    f'wer_scores = {wer_scores}',
    f'cer_scores = {cer_scores}',
    f'corpus_bleu_score_with_tf = {corpus_bleu_scores_previous}',
    f'sacre_blue_score_with_tf = {sacre_blue_previous}',
    f'rouge_scores_with_tf = {rouge_scores_previous}',
    f'wer_scores_with_tf = {wer_scores_previous}',
    f'cer_scores_with_tf = {cer_scores_previous}',
    f'total_time_minutes = {total_time / 60}'
    ]
    
    with open(score_results, "a") as file_results:
        for line in file_content:
            if isinstance(line, list):
                for item in line:
                    file_results.write(str(item) + "\n")
            else:
                file_results.write(str(line) + "\n")

if __name__ == '__main__':
    args = get_config('eval_decoder')

    """基本参数设置"""
    batch_size = 1
    dataset_setting = 'unique_sent'
    checkpoint_path = args['checkpoint_path']
    test_input = args['test_input']
    print("测试输入为:", test_input)

    """加载训练配置"""
    training_config = json.load(open(args['train_config_path']))
    subject_choice = training_config['subjects']
    eeg_type_choice = training_config['eeg_type']
    bands_choice = training_config['eeg_bands']
    task_name = training_config['task_name']

    """加载实验配置"""
    experiment_config = json.load(open(args['experiment_config_path']))
    train_input = experiment_config['train_input']
    pretrained_name = experiment_config['pretrained']
    encoder_name = experiment_config['encoder']
    pretrained_dict = MODEL_REGISTRY[pretrained_name]
    encoder_dict = ENCODER_REGISTRY[encoder_name]
    
    """设置随机种子以确保结果可复现"""
    seed = args['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    """设置设备"""
    if torch.cuda.is_available():  
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    device = torch.device(dev)
    print(f'[信息]使用设备 {dev}')

    """设置输出路径"""
    output_all_results_path = f'./results/decoder_eval/{task_name}-{pretrained_name}-{encoder_name}-{train_input}_{test_input}/all_decoding_results.xlsx'
    score_results = f'./results/decoder_eval/{task_name}-{pretrained_name}-{encoder_name}-{train_input}_{test_input}/score_results.xlsx'

    """加载数据集"""
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = '/data/johj/ZuCo_data/task1-SR/task1_source.pkl' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = '/data/johj/ZuCo_data/task2-NR/task2_source.pkl' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = '/data/johj/ZuCo_data/task3-TSR/task3_source.pkl' 
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = '/data/johj/ZuCo_data/task2-NR-2.0/taskNRv2_source.pkl' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    """加载分词器"""
    tokenizer = CLASS_MAP[pretrained_dict['tokenizer']].from_pretrained(pretrained_dict['local_model_path'])

    """创建数据集和数据加载器"""
    test_set = ZuCo_dataset(whole_dataset_dicts, 
                            'test', 
                            tokenizer, 
                            subject = subject_choice, 
                            eeg_type = eeg_type_choice, 
                            bands = bands_choice, 
                            setting = dataset_setting, 
                            test_input=test_input)

    dataset_sizes = {"test_set":len(test_set)}
    print('[信息]测试集大小: ', len(test_set))
    test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=4)
    dataloaders = {'test':test_dataloader}

    """加载模型"""
    pretrained = CLASS_MAP[pretrained_dict['generation']].from_pretrained(pretrained_dict['local_model_path'])
    encoder = CLASS_MAP[encoder_dict['model']](**encoder_dict['args'])
    model = Decoder(pretrained_name, pretrained, encoder, tokenizer)
    state_dict = torch.load(checkpoint_path)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)

    """定义损失函数"""
    criterion = nn.CrossEntropyLoss()
    
    ''' eval '''
    eval_model(dataloaders, 
               device, 
               tokenizer, 
               criterion, 
               model, 
               output_all_results_path = output_all_results_path, 
               score_results=score_results)
