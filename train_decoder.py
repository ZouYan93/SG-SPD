import time
import copy
import torch
import tqdm
import os
import numpy as np
import json
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from model.decoder import Decoder
from utils.data.data import ZuCo_dataset
from utils.config import get_config
from utils.registry import MODEL_REGISTRY, ENCODER_REGISTRY, CLASS_MAP

def train_model(dataloaders, 
                device, 
                model, 
                tokenizer, 
                criterion, 
                optimizer, 
                scheduler, 
                num_epochs=20, 
                checkpoint_path_best = './checkpoints/decoder/best/decoder.pt', 
                checkpoint_path_last = './checkpoints/decoder/last/decoder.pt'):
    since = time.time()
      
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = 1e12

    for epoch in range(num_epochs):
        print(f'进度 {epoch + 1}/{num_epochs}')
        print('-' * 50)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
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

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs_batch, masks_batch, invert_masks_batch, target_ids_batch)
                    
                    # TODO：可以增加损失，如跨模态对齐等, 后续即插即用
                    loss = output.loss
                    if phase == 'train':
                        loss.sum().backward()
                        optimizer.step()

                epoch_loss += loss.sum().item() * inputs_batch.size()[0]
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = epoch_loss / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.6f}')

            if phase == 'eval' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                # 保存当前最优模型权重
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'更新最优检查点: {checkpoint_path_best}')

    train_time = time.time() - since
    print(f'训练时间: {train_time:.0f}分{train_time % 60:.0f}秒')
    print(f'最优验证损失: {best_loss:.4f}')
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'更新最后检查点: {checkpoint_path_last}')

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model

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

    """设置设备"""
    device_ids = [0]
    if torch.cuda.is_available():  
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    device = torch.device(dev)
    print(f'[信息]使用设备 {dev}')

    """设置随机种子"""
    seed = args['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    """数据相关设置"""
    subject_choice = args['subjects']
    print(f'[信息]被试选择 {subject_choice}')
    eeg_type_choice = args['eeg_type']
    print(f'[信息]EEG类型 {eeg_type_choice}')
    bands_choice = args['eeg_bands']
    print(f'[信息]使用通道 {bands_choice}')

    """加载数据"""
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = '/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = '/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = '/dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle' 
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = '/dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    print(f'[信息]数据加载完成')

    """写入训练基本信息"""
    with open(os.path.join(save_dir, 'config.json'), 'w') as out_config:
        args_without_experiments = {k: v for k, v in args.items() if k != 'experiments'}
        json.dump(args_without_experiments, out_config, indent = 4)

    """多次实验训练"""
    for i, arg_experiment in enumerate(args_experiments):
        print('#' * 50)
        pretrained_name = arg_experiment['pretrained']
        encoder_name = arg_experiment['encoder']
        train_input = arg_experiment['train_input']
        pretrained_dict = MODEL_REGISTRY[pretrained_name]
        encoder_dict = ENCODER_REGISTRY[encoder_name]
        print(f'[信息]实验 {i+1}/{len(args_experiments)}')
        print(f'[信息]该次训练信息： 预训练模型 {pretrained_name}，编码器 {encoder_name}，训练输入类型 {train_input}')
        print('#' * 50)

        """独立实验结果保存路径设置"""
        save_path = os.path.join(save_dir, f'{pretrained_name}_{encoder_name}_{train_input}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if skip_step_one:
            save_name = f'{pretrained_name}_{encoder_name}_{train_input}_skip'
        else:
            save_name = f'{pretrained_name}_{encoder_name}_{train_input}'

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

        cfg_path = os.path.join(save_path, f'config.json')
        if not os.path.exists(cfg_path):
            os.makedirs(cfg_path)
        with open(os.path.join(cfg_path,f'{save_name}.json'), 'w') as out_config:
            arg_experiment = {**arg_experiment, **pretrained_dict, **encoder_dict}
            json.dump(arg_experiment, out_config, indent = 4)

        tokenizer = CLASS_MAP[pretrained_dict['tokenizer']].from_pretrained(pretrained_dict['model_id'])

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
            config = CLASS_MAP[pretrained_dict['config']].from_pretrained(pretrained_dict['model_id'])
            pretrained = CLASS_MAP[pretrained_dict['generation']](config)
        else:
            pretrained = CLASS_MAP[pretrained_dict['generation']].from_pretrained(pretrained_dict['model_id'])

        # 编码器设置
        encoder = CLASS_MAP[encoder_dict['model']](**encoder_dict['args'])
        model = Decoder(pretrained, encoder, tokenizer)
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
            model = train_model(dataloaders, 
                                device, 
                                model, 
                                criterion, 
                                optimizer_step1, 
                                exp_lr_scheduler_step1, 
                                num_epochs=num_epochs_step1, 
                                checkpoint_path_best = output_checkpoint_name_best, 
                                checkpoint_path_last = output_checkpoint_name_last)

        """阶段二训练: 更新整个模型进行几次迭代"""
        for name, param in model.named_parameters():
            param.requires_grad = True
        optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
        exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        print('[信息]开始阶段二训练 ...')
        trained_model = train_model(dataloaders, 
                                    device, 
                                    model, 
                                    criterion, 
                                    optimizer_step2, 
                                    exp_lr_scheduler_step2, 
                                    num_epochs=num_epochs_step2, 
                                    checkpoint_path_best = output_checkpoint_name_best, 
                                    checkpoint_path_last = output_checkpoint_name_last)