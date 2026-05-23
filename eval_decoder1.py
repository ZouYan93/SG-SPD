import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import os
import json
from tqdm import tqdm
from utils.data.data import ZuCo_dataset
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from evaluate import load
from model.decoder import Decoder
from utils.config import get_config
from utils.registry import MODEL_REGISTRY, ENCODER_REGISTRY, CLASS_MAP
from utils.experiment_metrics import write_sections_to_excel
from utils.utils import remove_text_after_token, parse_run_and_experiment_from_ckpt, \
                        load_training_and_experiment_configs, load_dataset_dicts, \
                        export_eval_decoder_experiments_json, load_metrics, \
                        ensure_torch_generate_compat, get_enhancement_name


def eval_model(dataloaders,
               device,
               tokenizer,
               model,
               metric,
               cer_metric,
               wer_metric):
    start_time = time.time()
    model.eval()
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    pred_tokens_list_previous = []
    pred_string_list_previous = []
    output_rows = []
    with torch.no_grad():
        for idx, (inputs, seq_len, masks, invert_masks, target_ids,
                  target_mask, sentiment_labels) in enumerate(tqdm(dataloaders['test'])):
            inputs_batch = inputs.to(device).float()
            masks_batch = masks.to(device)
            invert_masks_batch = invert_masks.to(device)
            target_ids_batch = target_ids.to(device)
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)

            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

            """Teacher-forcing 解码推理"""
            output = model(inputs_batch, masks_batch, invert_masks_batch, target_ids_batch)
            logits_previous = output.logits
            probs_previous = logits_previous[0].softmax(dim=1)
            values_previous, predictions_previous = probs_previous.topk(1)
            predictions_previous = torch.squeeze(predictions_previous)
            predicted_string_previous = remove_text_after_token(tokenizer.decode(predictions_previous).split('</s></s>')[0].replace('<s>',''))
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
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            output_rows.append({
                'sample_id': idx,
                'target_string': target_string,
                'predicted_string': predicted_string,
                'predicted_string_with_tf': predicted_string_previous,
            })

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

    rouge1 = rouge_scores['rouge-1'] if rouge_scores else {'p': None, 'r': None, 'f': None}
    rouge1_previous = rouge_scores_previous['rouge-1'] if rouge_scores_previous else {'p': None, 'r': None, 'f': None}

    summary = {
        'bleu_1': corpus_bleu_scores[0],
        'bleu_2': corpus_bleu_scores[1],
        'bleu_3': corpus_bleu_scores[2],
        'bleu_4': corpus_bleu_scores[3],
        'bleu_1_with_tf': corpus_bleu_scores_previous[0],
        'bleu_2_with_tf': corpus_bleu_scores_previous[1],
        'bleu_3_with_tf': corpus_bleu_scores_previous[2],
        'bleu_4_with_tf': corpus_bleu_scores_previous[3],
        'sacrebleu': sacre_blue.get('score') if isinstance(sacre_blue, dict) else sacre_blue,
        'sacrebleu_with_tf': sacre_blue_previous.get('score') if isinstance(sacre_blue_previous, dict) else sacre_blue_previous,
        'rouge1_p': rouge1.get('p'),
        'rouge1_r': rouge1.get('r'),
        'rouge1_f': rouge1.get('f'),
        'rouge1_p_with_tf': rouge1_previous.get('p'),
        'rouge1_r_with_tf': rouge1_previous.get('r'),
        'rouge1_f_with_tf': rouge1_previous.get('f'),
        'wer': wer_scores,
        'wer_with_tf': wer_scores_previous,
        'cer': cer_scores,
        'cer_with_tf': cer_scores_previous,
        'time_minutes': total_time / 60,
        'sample_count': len(output_rows),
    }

    return summary, output_rows

if __name__ == '__main__':
    args = get_config('eval_decoder')
    ensure_torch_generate_compat()

    metrics_dir = './metrics'
    metric, cer_metric, wer_metric = load_metrics(metrics_dir)

    """基本参数设置"""
    batch_size = args['batch_size']
    dataset_setting = 'unique_sent'
    test_input = args['test_input']
    search_paths_json_path = args['search_paths_json_path']
    include_regex = args['include_regex']
    exclude_regex_list = args['exclude_regex_list']
    ckpt_list_output_json_path = args['ckpt_list_output_json_path']

    if search_paths_json_path:
        with open(search_paths_json_path, 'r', encoding='utf-8') as f:
            json_obj = json.load(f)

        if isinstance(json_obj, list):
            search_paths = json_obj
        elif isinstance(json_obj, dict):
            if 'search_paths' in json_obj:
                search_paths = json_obj['search_paths']
            elif 'paths' in json_obj:
                search_paths = json_obj['paths']
            else:
                raise ValueError(f'无法识别 JSON 键，需包含 search_paths 或 paths: {search_paths_json_path}')
        else:
            raise ValueError(f'路径 JSON 格式错误，需为 list 或 dict: {search_paths_json_path}')

        checkpoint_paths = export_eval_decoder_experiments_json(
            search_paths=search_paths,
            output_json_path=ckpt_list_output_json_path,
            include_regex=include_regex,
            exclude_regex_list=exclude_regex_list,
        )

    if not checkpoint_paths:
        raise ValueError('未找到任何 checkpoint，请检查输入路径和正则过滤条件')

    print("测试输入为:", test_input)
    print(f"[信息]待评估 ckpt 数量: {len(checkpoint_paths)}")
    
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

    for ckpt_path in checkpoint_paths:
        run_dir, run_name, experiment_dir = parse_run_and_experiment_from_ckpt(ckpt_path)
        training_config, experiment_config = load_training_and_experiment_configs(run_dir, experiment_dir)

        subject_choice = training_config['subjects']
        eeg_type_choice = training_config['eeg_type']
        bands_choice = training_config['eeg_bands']
        task_name = training_config['task_name']

        train_input = experiment_config['train_input']
        pretrained_name = experiment_config['pretrained']
        encoder_name = experiment_config['encoder']
        enhancement_config = experiment_config.get('enhancement', None)
        enhancement_name = get_enhancement_name(enhancement_config)
        pretrained_dict = MODEL_REGISTRY[pretrained_name]
        encoder_dict = ENCODER_REGISTRY[encoder_name]

        output_root = os.path.join('./results/eval_decoder', run_name)
        os.makedirs(output_root, exist_ok=True)
        results_excel_path = os.path.join(output_root, 'results.xlsx')
        sheet_name = f'{pretrained_name}_{encoder_name}_{train_input}_{test_input}'
        if enhancement_name != 'baseline':
            sheet_name = f'{enhancement_name}_{test_input}'

        print('#' * 50)
        print(f'[信息]正在评估: {ckpt_path}')
        print(f'[信息]结果文件: {results_excel_path}')
        print(f'[信息]sheet: {sheet_name}')

        whole_dataset_dicts = load_dataset_dicts(task_name)
        tokenizer = CLASS_MAP[pretrained_dict['tokenizer']].from_pretrained(pretrained_dict['local_model_path'])

        test_set = ZuCo_dataset(
            whole_dataset_dicts,
            'test',
            tokenizer,
            subject=subject_choice,
            eeg_type=eeg_type_choice,
            bands=bands_choice,
            setting=dataset_setting,
            test_input=test_input,
        )

        print('[信息]测试集大小: ', len(test_set))
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        dataloaders = {'test': test_dataloader}

        pretrained = CLASS_MAP[pretrained_dict['generation']].from_pretrained(
            pretrained_dict['local_model_path'],
            use_safetensors=False,
        )
        encoder = CLASS_MAP[encoder_dict['model']](**encoder_dict['args'])
        model = Decoder(pretrained_name, pretrained, encoder, tokenizer, enhancement_config=enhancement_config)
        state_dict = torch.load(ckpt_path, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.to(device)

        summary, output_rows = eval_model(
            dataloaders,
            device,
            tokenizer,
            model,
            metric,
            cer_metric,
            wer_metric,
        )

        checkpoint_type = os.path.basename(os.path.dirname(ckpt_path))
        sections = [
            {
                'type': 'kv',
                'title': f'checkpoint_summary_{checkpoint_type}',
                'data': {
                    'checkpoint_path': ckpt_path,
                    'checkpoint_type': checkpoint_type,
                    'task_name': task_name,
                    'test_input': test_input,
                    **summary,
                },
            },
            {
                'type': 'table',
                'title': f'outputs_{checkpoint_type}',
                'headers': ['sample_id', 'target_string', 'predicted_string', 'predicted_string_with_tf'],
                'rows': output_rows,
            },
        ]

        write_sections_to_excel(results_excel_path, sheet_name, sections, mode='replace')
        print(f'[信息]评估结果已写入: {results_excel_path}')
