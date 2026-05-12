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
from utils.utils import load_dataset_dicts, state_dict_to_cpu, release_cuda_memory, get_enhancement_name


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
                checkpoint_path_last = './checkpoints/decoder/last/decoder.pt'):
    since = time.time()
    best_model_weights = state_dict_to_cpu(model.state_dict())
    best_loss = 1e12
    total_train_time = 0.0
    train_loss_history = []
    eval_loss_history = []

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

            for inputs, seq_len, masks, invert_masks, target_ids, target_mask, sentiment_labels in tqdm(dataloaders[phase]):
                inputs_batch = inputs.to(device).float()
                masks_batch = masks.to(device)
                invert_masks_batch = invert_masks.to(device)
                target_ids_batch = target_ids.to(device)
                # torch会自动忽略标签为-100的token
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs_batch, masks_batch, invert_masks_batch, target_ids_batch)
                    loss = output.loss
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

            print(f'{phase} Loss: {epoch_loss:.6f}')

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
    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4

    """路径设置"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(save_dir, f'{task_name}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    metrics_excel_path = os.path.join(save_dir, 'experiment_metrics.xlsx')

    """设置设备"""
    device_ids = [0]
    if torch.cuda.is_available():  
        dev = args['cuda'] 
    else:  
        dev = "cpu"
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
        pretrained_dict = MODEL_REGISTRY[pretrained_name]
        encoder_dict = ENCODER_REGISTRY[encoder_name]
        print('#' * 50)
        print(f'[信息]实验 {i+1} / {len(args_experiments)}')
        print(f'[信息]该次训练信息：预训练模型 {pretrained_name}, 编码器 {encoder_name}, 训练输入类型 {train_input}, 增强配置 {enhancement_name}')
        print('#' * 50)

        """独立实验结果保存路径设置"""
        save_suffix = f'_{enhancement_name}' if has_enhancement_config else ''
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
                                 test_input=train_input)
        # dev dataset
        dev_set = ZuCo_dataset(whole_dataset_dicts, 
                               'eval', 
                               tokenizer, 
                               subject=subject_choice, 
                               eeg_type=eeg_type_choice, 
                               bands=bands_choice, 
                               setting=setting, 
                               test_input=train_input)
    
        dataset_sizes = {'train': len(train_set), 'eval': len(dev_set)}
        print('[信息]训练集数量: ', len(train_set))
        print('[信息]验证集数量: ', len(dev_set))
    
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=4)
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
    
        if skip_step_one:
            step1_metrics = {
                'train_time_sec': 0.0,
                'train_samples_per_sec': 0.0,
                'best_eval_loss': None,
                'train_loss_history': [],
                'eval_loss_history': [],
            }
            if load_step1_checkpoint:
                print(f'加载阶段一检查点: {stepone_checkpoint}')
                model.load_state_dict(torch.load(stepone_checkpoint))
            else:
                print('跳过阶段一训练，直接进入阶段二训练 ...')
        else:
            optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)
            exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)
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
                                               checkpoint_path_last = output_checkpoint_name_last)
            release_cuda_memory(optimizer_step1, exp_lr_scheduler_step1)

        """阶段二训练: 更新整个模型进行几次迭代"""
        for name, param in model.named_parameters():
            param.requires_grad = True
        optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
        exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)
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
                                                   checkpoint_path_last = output_checkpoint_name_last)

        """统计模型参数量、检查点文件大小等信息，并写入Excel表格"""
        total_params, trainable_params = count_model_parameters(trained_model)
        best_checkpoint_size_mb = get_file_size_mb(output_checkpoint_name_best)
        last_checkpoint_size_mb = get_file_size_mb(output_checkpoint_name_last)
        total_time_sec = time.time() - experiment_start_time
        final_best_eval_loss = step2_metrics['best_eval_loss']
        sheet_name = f'{pretrained_name}_{encoder_name}_{train_input}'
        if enhancement_name != 'baseline':
            sheet_name = f'{sheet_name}_{enhancement_name}'

        summary_metrics = {
            'experiment_name': save_name,
            'pretrained': pretrained_name,
            'encoder': encoder_name,
            'train_input': train_input,
            'enhancement': enhancement_name,
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