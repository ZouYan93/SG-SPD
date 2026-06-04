import time
import torch
import os
import numpy as np
import json
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from model.decoder import Decoder
from utils.data.data import ZuCo_dataset
from utils.config import get_config
from utils.registry import MODEL_REGISTRY, ENCODER_REGISTRY, CLASS_MAP
from utils.experiment_metrics import count_model_parameters, get_file_size_mb, write_sections_to_excel
from utils.utils import load_dataset_dicts, state_dict_to_cpu, release_cuda_memory, get_enhancement_name, \
                        should_freeze_pretrained_step2, freeze_pretrained_parameters, apply_step2_lora_if_needed

def set_step2_train_scope(model, enhancement_config):
    strategy = (enhancement_config or {}).get('stage2_freeze_strategy', None)
    if strategy is None:
        strategy = 'full_freeze_llm' if should_freeze_pretrained_step2(enhancement_config) else 'full_trainable_llm'

    if strategy in ('full_freeze_llm', 'lora', 'full_trainable_llm'):
        trainable = []
        frozen_pretrained = 0
        freeze_stage1_step2 = bool((enhancement_config or {}).get('freeze_stage1_step2', False))
        train_non_pretrained_step2 = bool((enhancement_config or {}).get('stage2_train_non_pretrained', True))
        neuro_config = (enhancement_config or {}).get('neuro_guide', {})
        train_lora_embeddings = bool(neuro_config.get('train_lora_embeddings', True))
        for name, param in model.named_parameters():
            if strategy == 'full_trainable_llm':
                param.requires_grad = True
            elif strategy == 'full_freeze_llm':
                if 'pretrained' in name:
                    param.requires_grad = False
                elif freeze_stage1_step2:
                    param.requires_grad = name.startswith('fc1.')
                else:
                    param.requires_grad = True
            else:
                # 源码 Stage2 默认 LoRA：LLM 主体冻结，只训练 LoRA/embedding 小参数；
                # EEG encoder、projector(fc1)、prompt-code 等非 pretrained 模块保持可训练。
                if 'pretrained' not in name:
                    if not train_non_pretrained_step2:
                        param.requires_grad = False
                    else:
                        param.requires_grad = name.startswith('fc1.') if freeze_stage1_step2 else True
                else:
                    param.requires_grad = (
                        'lora_' in name
                        or '.lora_' in name
                        or (train_lora_embeddings and ('embed_tokens' in name or 'shared' in name))
                    )

            if param.requires_grad:
                trainable.append(name)
            elif 'pretrained' in name:
                frozen_pretrained += param.numel()

        print(f'[信息]阶段二冻结策略: {strategy}, 冻结 pretrained 参数: {frozen_pretrained}, 可训练张量数: {len(trainable)}')
        for name in trainable[:30]:
            print(f'[信息]  trainable: {name}')
        if len(trainable) > 30:
            print(f'[信息]  ... 其余 {len(trainable) - 30} 个可训练张量省略')
        return

    scope = (enhancement_config or {}).get('step2_train_scope', 'default')
    if scope == 'default':
        for name, param in model.named_parameters():
            param.requires_grad = True
        if should_freeze_pretrained_step2(enhancement_config):
            freeze_pretrained_parameters(model)
        return

    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable = []
    for name, param in model.named_parameters():
        if scope == 'decoder_cross_attention_norm':
            is_bart_cross_attn = (
                'pretrained' in name
                and '.decoder.layers.' in name
                and ('encoder_attn' in name or 'encoder_attn_layer_norm' in name)
            )
            is_t5_cross_attn = (
                'pretrained' in name
                and '.decoder.block.' in name
                and ('EncDecAttention' in name or 'layer.1.layer_norm' in name)
            )
            is_decoder_norm = (
                'pretrained' in name
                and '.decoder.' in name
                and ('layer_norm' in name or 'LayerNorm' in name or 'final_layer_norm' in name)
            )
            param.requires_grad = is_bart_cross_attn or is_t5_cross_attn or is_decoder_norm
        elif scope == 'pretrained_decoder':
            param.requires_grad = 'pretrained' in name and '.decoder.' in name
        else:
            raise ValueError(f'Unsupported step2_train_scope: {scope}')

        if param.requires_grad:
            trainable.append(name)

    print(f'[信息]阶段二训练范围 {scope}, 可训练参数张量数量: {len(trainable)}')
    for name in trainable[:30]:
        print(f'[信息]  trainable: {name}')
    if len(trainable) > 30:
        print(f'[信息]  ... 其余 {len(trainable) - 30} 个参数张量省略')

def has_auxiliary_loss(enhancement_config):
    if not enhancement_config:
        return False
    return (
        enhancement_config.get('semantic_alignment', {}).get('enabled', False)
        or enhancement_config.get('hidden_distillation', {}).get('enabled', False)
        or enhancement_config.get('noise_contrast', {}).get('enabled', False)
    )

def has_neuroguide(enhancement_config):
    return bool(enhancement_config and enhancement_config.get('neuro_guide', {}).get('enabled', False))

def freeze_neuroguide_after_step1(model, enhancement_config):
    if not has_neuroguide(enhancement_config):
        return
    neuro_config = enhancement_config.get('neuro_guide', {})
    if not neuro_config.get('freeze_after_step1', True):
        return

    frozen = 0
    for name, param in model.named_parameters():
        if 'neuro_guide' in name:
            param.requires_grad = False
            frozen += param.numel()
    print(f'[信息] Stage2 冻结 NeuroGuide 参数: {frozen}')

def freeze_attribute_heads_step2(model, enhancement_config):
    if not has_neuroguide(enhancement_config):
        return
    neuro_config = enhancement_config.get('neuro_guide', {})
    if not neuro_config.get('freeze_attribute_heads_step2', False):
        return

    head_names = (
        'sentiment_classifier',
        'topic_classifier',
        'length_regressor',
        'surprisal_regressor',
    )
    frozen = 0
    for name, param in model.named_parameters():
        if any(head_name in name for head_name in head_names):
            param.requires_grad = False
            frozen += param.numel()
    print(f'[信息] Stage2 冻结属性头参数: {frozen}')

def train_model(dataloaders, 
                device, 
                model, 
                tokenizer, 
                dataset_sizes,
                criterion, 
                optimizer, 
                scheduler, 
                num_epochs=20, 
                checkpoint_path_best = './checkpoints/decoder/best/decoder.pt', 
                checkpoint_path_last = './checkpoints/decoder/last/decoder.pt',
                loss_type='combined'):
    since = time.time()
    best_model_weights = state_dict_to_cpu(model.state_dict())
    best_loss = 1e12
    total_train_time = 0.0
    train_loss_history = []
    eval_loss_history = []
    aux_loss_history = {'train': [], 'eval': []}

    for epoch in range(num_epochs):
        print(f'进度 {epoch + 1}/{num_epochs}')
        print('-' * 50)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
                epoch_train_start = time.time()
            else:
                model.eval()

            epoch_loss = 0.0
            aux_metric_sums = {}
            aux_metric_counts = {}

            for batch in tqdm(dataloaders[phase]):
                alignment_target_ids = None
                topic_labels = None
                length_values = None
                surprisal_values = None
                if len(batch) >= 13:
                    (
                        inputs,
                        seq_len,
                        masks,
                        invert_masks,
                        target_ids,
                        target_mask,
                        sentiment_labels,
                        prior_ids,
                        alignment_target_ids,
                        _,
                        topic_labels,
                        length_values,
                        surprisal_values,
                    ) = batch[:13]
                    prior_ids_batch = prior_ids.to(device)
                elif len(batch) >= 10:
                    inputs, seq_len, masks, invert_masks, target_ids, target_mask, sentiment_labels, prior_ids, alignment_target_ids, _ = batch[:10]
                    prior_ids_batch = prior_ids.to(device)
                elif len(batch) == 8:
                    inputs, seq_len, masks, invert_masks, target_ids, target_mask, sentiment_labels, prior_ids = batch
                    prior_ids_batch = prior_ids.to(device)
                else:
                    inputs, seq_len, masks, invert_masks, target_ids, target_mask, sentiment_labels = batch
                    prior_ids_batch = None
                inputs_batch = inputs.to(device).float()
                seq_len_batch = seq_len.to(device)
                masks_batch = masks.to(device)
                invert_masks_batch = invert_masks.to(device)
                target_ids_batch = target_ids.to(device)
                alignment_target_ids_batch = None if alignment_target_ids is None else alignment_target_ids.to(device)
                sentiment_labels_batch = sentiment_labels.to(device)
                topic_labels_batch = None if topic_labels is None else topic_labels.to(device)
                length_values_batch = None if length_values is None else length_values.to(device)
                surprisal_values_batch = None if surprisal_values is None else surprisal_values.to(device)
                # torch会自动忽略标签为-100的token
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100
                if alignment_target_ids_batch is not None:
                    alignment_target_ids_batch[alignment_target_ids_batch == tokenizer.pad_token_id] = -100

                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(
                        inputs_batch,
                        masks_batch,
                        invert_masks_batch,
                        target_ids_batch,
                        loss_type=loss_type,
                        prior_ids_batch=prior_ids_batch,
                        seq_len_batch=seq_len_batch,
                        sentiment_labels_batch=sentiment_labels_batch,
                        topic_labels_batch=topic_labels_batch,
                        length_values_batch=length_values_batch,
                        surprisal_values_batch=surprisal_values_batch,
                        alignment_target_ids_batch=alignment_target_ids_batch,
                    )
                    loss = output.loss
                    aux_loss_dict = getattr(output, 'aux_loss_dict', {})
                    for aux_name, aux_value in aux_loss_dict.items():
                        aux_metric_sums[aux_name] = aux_metric_sums.get(aux_name, 0.0) + float(aux_value) * inputs_batch.size()[0]
                        aux_metric_counts[aux_name] = aux_metric_counts.get(aux_name, 0) + inputs_batch.size()[0]
                    if phase == 'train':
                        loss.sum().backward()
                        optimizer.step()

                epoch_loss += loss.sum().item() * inputs_batch.size()[0]
                
            if phase == 'train':
                epoch_train_time = time.time() - epoch_train_start
                total_train_time += epoch_train_time
                scheduler.step()

            epoch_loss = epoch_loss / dataset_sizes[phase]
            if phase == 'train':
                train_loss_history.append(float(epoch_loss))
            else:
                eval_loss_history.append(float(epoch_loss))
            aux_epoch_metrics = {
                aux_name: aux_metric_sums[aux_name] / max(aux_metric_counts.get(aux_name, 0), 1)
                for aux_name in aux_metric_sums
            }
            aux_loss_history[phase].append(aux_epoch_metrics)

            print(f'{phase} Loss: {epoch_loss:.6f}')
            if aux_epoch_metrics:
                aux_text = ', '.join([f'{k}: {v:.6f}' for k, v in aux_epoch_metrics.items()])
                print(f'{phase} Aux: {aux_text}')

            if phase == 'eval' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = state_dict_to_cpu(model.state_dict())
                # 保存当前最优模型权重
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'更新最优检查点: {checkpoint_path_best}')

    total_time = time.time() - since
    print(f'总耗时: {total_time // 60:.0f}分{total_time % 60:.0f}秒')
    print(f'训练耗时: {total_train_time // 60:.0f}分{total_train_time % 60:.0f}秒')
    print(f'最优验证损失: {best_loss:.4f}')
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'更新最后检查点: {checkpoint_path_last}')

    train_samples = dataset_sizes['train'] * num_epochs
    train_samples_per_sec = train_samples / total_train_time if total_train_time > 0 else 0.0

    # load best model weights
    model.load_state_dict(best_model_weights)
    metrics = {
        'train_time_sec': total_train_time,
        'train_samples_per_sec': train_samples_per_sec,
        'best_eval_loss': best_loss,
        'train_loss_history': train_loss_history,
        'eval_loss_history': eval_loss_history,
        'aux_loss_history': aux_loss_history,
    }
    return model, metrics

if __name__ == '__main__':
    args = get_config('train_decoder')
    args_experiments = args['experiments']

    """基本训练配置"""
    setting = 'unique_sent'
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']
    task_name = args['task_name']
    save_dir = args['save_dir']
    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    stepone_checkpoint = args['stepone_checkpoint']
    use_random_init = args['use_random_init']
    global_use_mtv = args.get('use_mtv', False)
    global_mtv_path = args.get('mtv_path', None)
    global_mtv_variant_keys = args.get('mtv_variant_keys', None)
    global_mtv_include_original = args.get('mtv_include_original', False)
    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4

    """路径设置"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(save_dir, f'{task_name}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    metrics_excel_path = os.path.join(save_dir, 'experiment_metrics.xlsx')

    """设置设备"""
    if torch.cuda.is_available():  
        dev = args['cuda'] 
        device_ids = [int(dev.split(':')[1])] if ':' in dev else [0]
    else:
        dev = "cpu"
        device_ids = None
    device = torch.device(dev)
    print(f'[信息]使用设备 {dev}')

    """设置随机种子"""
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    """数据相关设置"""
    subject_choice = args['subjects']
    print(f'[信息]被试选择 {subject_choice}')
    eeg_type_choice = args['eeg_type']
    print(f'[信息]EEG类型 {eeg_type_choice}')
    bands_choice = args['eeg_bands']
    print(f'[信息]使用通道 {bands_choice}')

    """加载数据"""
    whole_dataset_dicts = load_dataset_dicts(task_name)
    print(f'[信息]数据加载完成')

    """写入训练基本信息"""
    with open(os.path.join(save_dir, 'config.json'), 'w') as out_config:
        args_without_experiments = {k: v for k, v in args.items() if k != 'experiments'}
        json.dump(args_without_experiments, out_config, indent = 4)

    """多次实验训练"""
    for i, arg_experiment in enumerate(args_experiments):
        experiment_start_time = time.time()
        pretrained_name = arg_experiment['pretrained']
        encoder_name = arg_experiment['encoder']
        train_input = arg_experiment['train_input']
        has_enhancement_config = 'enhancement' in arg_experiment
        enhancement_config = arg_experiment.get('enhancement', None)
        enhancement_name = get_enhancement_name(enhancement_config)
        mtv_config = arg_experiment.get('mtv', {})
        if isinstance(mtv_config, bool):
            mtv_enabled = mtv_config
            mtv_path = global_mtv_path
            mtv_variant_keys = global_mtv_variant_keys
            mtv_include_original = global_mtv_include_original
        else:
            mtv_enabled = bool(mtv_config.get('enabled', global_use_mtv))
            mtv_path = mtv_config.get('path', global_mtv_path)
            mtv_variant_keys = mtv_config.get('variant_keys', global_mtv_variant_keys)
            mtv_include_original = bool(mtv_config.get('include_original', global_mtv_include_original))
        neuro_config = (enhancement_config or {}).get('neuro_guide', {})
        attribute_path = neuro_config.get('attribute_path', None)
        require_semkey_attributes = bool(neuro_config.get('require_semkey_attributes', False))
        pretrained_dict = MODEL_REGISTRY[pretrained_name]
        encoder_dict = ENCODER_REGISTRY[encoder_name]
        print('#' * 50)
        print(f'[信息]实验 {i+1} / {len(args_experiments)}')
        print(f'[信息]该次训练信息：预训练模型 {pretrained_name}, 编码器 {encoder_name}, 训练输入类型 {train_input}, 增强配置 {enhancement_name}')
        print('#' * 50)

        """独立实验结果保存路径设置"""
        print(f'[信息] MTV: enabled={mtv_enabled}, path={mtv_path}, include_original={mtv_include_original}')

        save_suffix = f'_{enhancement_name}' if has_enhancement_config else ''
        if mtv_enabled:
            save_suffix = f'{save_suffix}_mtv'
        save_path = os.path.join(save_dir, f'{pretrained_name}_{encoder_name}_{train_input}{save_suffix}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if skip_step_one:
            save_name = f'{pretrained_name}_{encoder_name}_{train_input}{save_suffix}_skip'
        else:
            save_name = f'{pretrained_name}_{encoder_name}_{train_input}{save_suffix}'

        if use_random_init:
            save_name =  save_name + '_randinit'

        save_path_best = os.path.join(save_path, 'best')
        if not os.path.exists(save_path_best):
            os.makedirs(save_path_best)
        output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')

        save_path_last = os.path.join(save_path, 'last')
        if not os.path.exists(save_path_last):
            os.makedirs(save_path_last)
        output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

        cfg_dir = os.path.join(save_path, 'config')
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        with open(os.path.join(cfg_dir, f'{save_name}.json'), 'w') as out_config:
            arg_experiment = {**arg_experiment, **pretrained_dict, **encoder_dict}
            json.dump(arg_experiment, out_config, indent = 4)

        tokenizer = CLASS_MAP[pretrained_dict['tokenizer']].from_pretrained(pretrained_dict['local_model_path'])

        """数据加载器设置"""
        # train dataset
        train_set = ZuCo_dataset(whole_dataset_dicts, 
                                 'train', 
                                 tokenizer, 
                                 subject=subject_choice, 
                                 eeg_type=eeg_type_choice, 
                                 bands=bands_choice, 
                                 setting=setting, 
                                 test_input=train_input,
                                 task_name=task_name,
                                 use_mtv=mtv_enabled,
                                 mtv_path=mtv_path,
                                 mtv_variant_keys=mtv_variant_keys,
                                 mtv_include_original=mtv_include_original,
                                 attribute_path=attribute_path,
                                 require_semkey_attributes=require_semkey_attributes)
        # dev dataset
        dev_set = ZuCo_dataset(whole_dataset_dicts, 
                               'eval', 
                               tokenizer, 
                               subject=subject_choice, 
                               eeg_type=eeg_type_choice, 
                               bands=bands_choice, 
                               setting=setting, 
                               test_input=train_input,
                               task_name=task_name,
                               attribute_path=attribute_path,
                               require_semkey_attributes=require_semkey_attributes)
    
        dataset_sizes = {'train': len(train_set), 'eval': len(dev_set)}
        print('[信息]训练集数量: ', len(train_set))
        print('[信息]验证集数量: ', len(dev_set))
    
        train_dataloader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=has_neuroguide(enhancement_config),
        )
        needs_batched_eval = enhancement_config and (
            enhancement_config.get('noise_contrast', {}).get('enabled', False)
            or enhancement_config.get('semantic_alignment', {}).get('enabled', False)
            or enhancement_config.get('hidden_distillation', {}).get('enabled', False)
            or has_neuroguide(enhancement_config)
        )
        eval_batch_size = batch_size if needs_batched_eval else 1
        val_dataloader = DataLoader(dev_set, batch_size=eval_batch_size, shuffle=False, num_workers=4)
        dataloaders = {'train':train_dataloader, 'eval':val_dataloader}

        '''模型设置'''
        # 预训练模型加载设置
        if use_random_init:
            config = CLASS_MAP[pretrained_dict['config']].from_pretrained(pretrained_dict['local_model_path'])
            pretrained = CLASS_MAP[pretrained_dict['generation']](config)
        else:
            pretrained = CLASS_MAP[pretrained_dict['generation']].from_pretrained(
                pretrained_dict['local_model_path'],
                use_safetensors=False,
            )

        # 编码器设置
        encoder = CLASS_MAP[encoder_dict['model']](**encoder_dict['args'])
        model = Decoder(pretrained_name, pretrained, encoder, tokenizer, enhancement_config=enhancement_config)
        if has_neuroguide(enhancement_config):
            stats = train_set.get_semkey_regression_stats()
            model.set_semkey_attribute_stats(**stats)
            print(
                '[信息] SemKey regression stats: '
                f"length={stats['length_mean']:.4f}±{stats['length_std']:.4f}, "
                f"surprisal={stats['surprisal_mean']:.4f}±{stats['surprisal_std']:.4f}, "
                f"has_surprisal={stats['has_surprisal']}"
            )
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=device_ids)

        """阶段一训练: 先冻结预训练模型部分权重，单独训练编码器和线性层"""
        if pretrained_name == 'Bart' or pretrained_name == 'T5' or pretrained_name == 'Pegasus':
            for name, param in model.named_parameters():
                if param.requires_grad and 'pretrained' in name:
                    if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                        continue
                    else:
                        param.requires_grad = False
        elif pretrained_name == 'Bert':
            for name, param in model.named_parameters():
                if param.requires_grad and 'pretrained' in name:
                    if ('embeddings' in name) or ('encoder.layer.0' in name):
                        continue
                    else:
                        param.requires_grad = False

        if has_neuroguide(enhancement_config):
            frozen_pretrained = 0
            for name, param in model.named_parameters():
                if 'pretrained' in name:
                    param.requires_grad = False
                    frozen_pretrained += param.numel()
            print(f'[信息] NeuroGuide Stage1 冻结 pretrained LLM 参数: {frozen_pretrained}')
    
        if skip_step_one:
            step1_metrics = {
                'train_time_sec': 0.0,
                'train_samples_per_sec': 0.0,
                'best_eval_loss': None,
                'train_loss_history': [],
                'eval_loss_history': [],
                'aux_loss_history': {'train': [], 'eval': []},
            }
            if load_step1_checkpoint:
                print(f'加载阶段一检查点: {stepone_checkpoint}')
                model.load_state_dict(torch.load(stepone_checkpoint))
            else:
                print('跳过阶段一训练，直接进入阶段二训练 ...')
        else:
            optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)
            exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=10, gamma=0.8)
            criterion = nn.CrossEntropyLoss()
            print('[信息]开始阶段一训练 ...')
            model, step1_metrics = train_model(dataloaders, 
                                               device, 
                                               model,
                                               tokenizer,
                                               dataset_sizes,
                                               criterion, 
                                               optimizer_step1, 
                                               exp_lr_scheduler_step1, 
                                               num_epochs=num_epochs_step1, 
                                               checkpoint_path_best = output_checkpoint_name_best, 
                                               checkpoint_path_last = output_checkpoint_name_last,
                                               loss_type='neuro_stage1' if has_neuroguide(enhancement_config) else ('aux' if has_auxiliary_loss(enhancement_config) else 'combined'))
            release_cuda_memory(optimizer_step1, exp_lr_scheduler_step1)

        """阶段二训练: 更新整个模型进行几次迭代"""
        model = apply_step2_lora_if_needed(model, enhancement_config, pretrained_name)
        set_step2_train_scope(model, enhancement_config)
        freeze_neuroguide_after_step1(model, enhancement_config)
        freeze_attribute_heads_step2(model, enhancement_config)
        optimizer_name_step2 = (enhancement_config or {}).get('stage2_optimizer', 'SGD')
        if optimizer_name_step2 == 'AdamW':
            optimizer_step2 = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=step2_lr,
                weight_decay=(enhancement_config or {}).get('stage2_weight_decay', 0.01),
            )
        else:
            optimizer_step2 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step2_lr, momentum=0.9)
        exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=5, gamma=0.85)
        criterion = nn.CrossEntropyLoss()
        print('[信息]开始阶段二训练 ...')
        trained_model, step2_metrics = train_model(dataloaders, 
                                                   device, 
                                                   model,
                                                   tokenizer,
                                                   dataset_sizes,
                                                   criterion, 
                                                   optimizer_step2, 
                                                   exp_lr_scheduler_step2, 
                                                   num_epochs=num_epochs_step2, 
                                                   checkpoint_path_best = output_checkpoint_name_best, 
                                                   checkpoint_path_last = output_checkpoint_name_last,
                                                   loss_type='neuro_stage2' if has_neuroguide(enhancement_config) else (enhancement_config.get('step2_loss_type', 'combined') if enhancement_config else 'combined'))

        """统计模型参数量、检查点文件大小等信息，并写入Excel表格"""
        total_params, trainable_params = count_model_parameters(trained_model)
        best_checkpoint_size_mb = get_file_size_mb(output_checkpoint_name_best)
        last_checkpoint_size_mb = get_file_size_mb(output_checkpoint_name_last)
        total_time_sec = time.time() - experiment_start_time
        final_best_eval_loss = step2_metrics['best_eval_loss']
        sheet_name = f'{pretrained_name}_{encoder_name}_{train_input}'
        if enhancement_name != 'baseline':
            sheet_name = f'{sheet_name}_{enhancement_name}'
        if mtv_enabled:
            sheet_name = f'{sheet_name}_mtv'

        summary_metrics = {
            'experiment_name': save_name,
            'pretrained': pretrained_name,
            'encoder': encoder_name,
            'train_input': train_input,
            'enhancement': enhancement_name,
            'use_mtv': mtv_enabled,
            'mtv_path': mtv_path,
            'mtv_include_original': mtv_include_original,
            'freeze_pretrained_step2': should_freeze_pretrained_step2(enhancement_config),
            'contrast_negative_type': None if not enhancement_config else enhancement_config.get('noise_contrast', {}).get('negative_type'),
            'step1_time_sec': round(step1_metrics['train_time_sec'], 4),
            'step2_time_sec': round(step2_metrics['train_time_sec'], 4),
            'total_time_sec': round(total_time_sec, 4),
            'step1_train_samples_per_sec': round(step1_metrics['train_samples_per_sec'], 4),
            'step2_train_samples_per_sec': round(step2_metrics['train_samples_per_sec'], 4),
            'step1_best_eval_loss': None if step1_metrics['best_eval_loss'] is None else round(step1_metrics['best_eval_loss'], 6),
            'step2_best_eval_loss': None if step2_metrics['best_eval_loss'] is None else round(step2_metrics['best_eval_loss'], 6),
            'final_best_eval_loss': None if final_best_eval_loss is None else round(final_best_eval_loss, 6),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'best_checkpoint_size_mb': None if best_checkpoint_size_mb is None else round(best_checkpoint_size_mb, 4),
            'last_checkpoint_size_mb': None if last_checkpoint_size_mb is None else round(last_checkpoint_size_mb, 4),
        }

        step1_loss_rows = []
        for idx in range(max(len(step1_metrics['train_loss_history']), len(step1_metrics['eval_loss_history']))):
            step1_loss_rows.append({
                'epoch': idx + 1,
                'train_loss': step1_metrics['train_loss_history'][idx] if idx < len(step1_metrics['train_loss_history']) else None,
                'eval_loss': step1_metrics['eval_loss_history'][idx] if idx < len(step1_metrics['eval_loss_history']) else None,
            })

        step2_loss_rows = []
        for idx in range(max(len(step2_metrics['train_loss_history']), len(step2_metrics['eval_loss_history']))):
            step2_loss_rows.append({
                'epoch': idx + 1,
                'train_loss': step2_metrics['train_loss_history'][idx] if idx < len(step2_metrics['train_loss_history']) else None,
                'eval_loss': step2_metrics['eval_loss_history'][idx] if idx < len(step2_metrics['eval_loss_history']) else None,
            })

        step1_aux_rows = []
        step1_aux_train = step1_metrics.get('aux_loss_history', {}).get('train', [])
        step1_aux_eval = step1_metrics.get('aux_loss_history', {}).get('eval', [])
        step1_aux_keys = sorted({
            key
            for row in step1_aux_train + step1_aux_eval
            for key in row.keys()
        })
        for idx in range(max(len(step1_aux_train), len(step1_aux_eval))):
            row = {'epoch': idx + 1}
            for key in step1_aux_keys:
                row[f'train_{key}'] = step1_aux_train[idx].get(key) if idx < len(step1_aux_train) else None
                row[f'eval_{key}'] = step1_aux_eval[idx].get(key) if idx < len(step1_aux_eval) else None
            step1_aux_rows.append(row)

        step2_aux_rows = []
        step2_aux_train = step2_metrics.get('aux_loss_history', {}).get('train', [])
        step2_aux_eval = step2_metrics.get('aux_loss_history', {}).get('eval', [])
        step2_aux_keys = sorted({
            key
            for row in step2_aux_train + step2_aux_eval
            for key in row.keys()
        })
        for idx in range(max(len(step2_aux_train), len(step2_aux_eval))):
            row = {'epoch': idx + 1}
            for key in step2_aux_keys:
                row[f'train_{key}'] = step2_aux_train[idx].get(key) if idx < len(step2_aux_train) else None
                row[f'eval_{key}'] = step2_aux_eval[idx].get(key) if idx < len(step2_aux_eval) else None
            step2_aux_rows.append(row)

        sections = [
            {
                'type': 'kv',
                'title': 'summary_metrics',
                'data': summary_metrics,
            },
            {
                'type': 'table',
                'title': 'step1_loss_curve',
                'headers': ['epoch', 'train_loss', 'eval_loss'],
                'rows': step1_loss_rows,
            },
            {
                'type': 'table',
                'title': 'step2_loss_curve',
                'headers': ['epoch', 'train_loss', 'eval_loss'],
                'rows': step2_loss_rows,
            },
        ]
        if step1_aux_rows:
            sections.append({
                'type': 'table',
                'title': 'step1_auxiliary_metrics',
                'headers': list(step1_aux_rows[0].keys()),
                'rows': step1_aux_rows,
            })
        if step2_aux_rows:
            sections.append({
                'type': 'table',
                'title': 'step2_auxiliary_metrics',
                'headers': list(step2_aux_rows[0].keys()),
                'rows': step2_aux_rows,
            })
        write_sections_to_excel(metrics_excel_path, sheet_name, sections, mode='replace')
        print(f'[信息]该次实验指标已写入: {metrics_excel_path}')
        release_cuda_memory(
            trained_model,
            model,
            pretrained,
            encoder,
            tokenizer,
            train_set,
            dev_set,
            train_dataloader,
            val_dataloader,
            dataloaders,
            optimizer_step2,
            exp_lr_scheduler_step2,
            criterion,
            step1_metrics,
            step2_metrics,
            sections,
            step1_loss_rows,
            step2_loss_rows,
            summary_metrics,
        )