import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from pyvirtualdisplay import Display
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

from data.build import get_dataset, get_split_dataloaders
from logs import logger
from logs.metrics import EpochMetric, SimpleMetric
from model.encoder import SynthTR
from model.loss import QuantizedNumericalParamsLoss, CategoricalParamsAccuracy, PresetProcessor, SynthParamsLoss, calculate_rewards
from utils.audio import AudioRenderer, Spectrogram_Processor
from utils.scheduler import linear_scheduler
from utils.hparams import LinearDynamicParam
from utils.distrib import get_parallel_devices
from utils.buffer import Replaybuffer


if __name__ == '__main__':
    # Finetune config
    ft_config = OmegaConf.load('config/finetune.yaml')
    logs_root_dir = Path(ft_config.logs_root_dir)
    model_path = logs_root_dir.joinpath(ft_config.model.saved_path)

    # Load model
    config = OmegaConf.load(model_path.joinpath('config.yaml'))
    device, device_ids = get_parallel_devices(main_cuda_device_idx=0)
    dexed_dataset = get_dataset('dexed', config)
    ft_dataset = get_dataset(ft_config.train.dataset, config)
    dataloader = get_split_dataloaders(config, ft_dataset)
    preset_idx_helper = dexed_dataset.preset_indexes_helper
    model = SynthTR(preset_idx_helper, **config.model.encoder_kwargs)
    model = model.to(device).train()
    model_parallel = nn.DataParallel(model, device_ids=device_ids, output_device=device)
    model_parallel.train()

    # Replay buffer
    buffer = Replaybuffer(len(ft_dataset), preset_idx_helper.full_preset_size, device)
    
    # Parameter loss
    if ft_config.loss.param:
        controls_criterion = SynthParamsLoss(
            preset_idx_helper,
            normalize_losses=ft_config.loss.normalize_losses,
            cat_softmax_t=ft_config.loss.cat_softmax_t,
            label_smoothing=ft_config.loss.label_smoothing,
        )

    # Policy Gradient loss
    if ft_config.loss.pg:
        preset_processor = PresetProcessor(dexed_dataset, preset_idx_helper, device)
        audio_renderer = AudioRenderer(
            dexed_dataset,
            config.model.write_sr,
            ft_config.loss.audio_eval_n_workers,
            device
        )
        spec_processor = Spectrogram_Processor(
            config.train.pg_nfft,
            config.train.pg_hop,
            config.model.write_sr,
        ).to(device)
            
    # Monitoring loss
    controls_num_eval_criterion = QuantizedNumericalParamsLoss(preset_idx_helper, numerical_loss=nn.MSELoss(reduction='mean'))
    controls_accuracy_criterion = CategoricalParamsAccuracy(preset_idx_helper, reduce=True, percentage_output=True)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=ft_config.optim.initial_lr,
            weight_decay=ft_config.optim.weight_decay,
            betas=ft_config.optim.betas
        )
    
    checkpoint = logger.get_model_last_checkpoint(logs_root_dir, config, device=device)
    model.load_state_dict(checkpoint['ae_model_state_dict'])
    
    # if ft_config.train.start_epoch > 0:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Logger
    logger = logger.RunLogger(logs_root_dir, ft_config)
    config.model.name = ft_config.model.name
    config.model.run_name = ft_config.model.run_name
    OmegaConf.save(config, logger.log_dir.joinpath('config.yaml'))
    OmegaConf.save(ft_config, logger.log_dir.joinpath('ft_config.yaml'))
    scalars_train, scalars_valid = dict(), dict()

    if ft_config.loss.pg:
        scalars_train['Specs/LogProb/Train'] = EpochMetric()
        scalars_train['Specs/PGLoss/Train'] = EpochMetric()
        scalars_train['Specs/SC/Train'] = EpochMetric()
        scalars_train['Specs/logMAE/Train'] = EpochMetric()
        scalars_train['Specs/mfccMAE/Train'] = EpochMetric()
        scalars_train['Specs/Rewards/Train'] = EpochMetric()
        scalars_valid['Specs/SC/Valid'] = EpochMetric()
        scalars_valid['Specs/logMAE/Valid'] = EpochMetric()
        scalars_valid['Specs/mfccMAE/Valid'] = EpochMetric()
        scalars_valid['Specs/Rewards/Valid'] = EpochMetric()

    if ft_config.loss.param:
        scalars_train['Controls/ParamLoss/Train'] = EpochMetric()
        scalars_valid['Controls/ParamLoss/Valid'] = EpochMetric()

    if ft_config.train.dataset == 'dexed':
        scalars_train['Controls/Accuracy/Train'] = EpochMetric()
        scalars_train['Controls/QLoss/Train'] = EpochMetric()
        scalars_valid['Controls/Accuracy/Valid'] = EpochMetric()
        scalars_valid['Controls/QLoss/Valid'] = EpochMetric()
        
    scalars_train['Sched/LR'] = SimpleMetric(ft_config.optim.initial_lr)

    if ft_config.scheduler.load:
        # scheduler = ExponentialLR(optimizer, ft_config.scheduler.gamma)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=0.000001)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:        
        scalars_train['Sched/LRwarmup'] = LinearDynamicParam(
            start_value=ft_config.scheduler.warmup_start_factor,
            end_value=1.0,
            end_epoch=ft_config.scheduler.warmup_epochs,
            current_epoch=0,
        )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = ft_config.optim.initial_lr      
        # scheduler = ExponentialLR(optimizer, ft_config.scheduler.gamma)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=0.000001)

    # Virtual display for the synthesizer
    disp = Display()
    disp.start()

    # Exploration
    deterministic = True

    for epoch in tqdm(range(ft_config.train.expl_epoch), desc='exploration epoch', position=0):
        dataloader_iter = iter(dataloader['train'])

        for i in tqdm(range(len(dataloader['train'])), desc='exploration batch', position=1, leave=False):
            sample = next(dataloader_iter)
            x_wav, x_in, v_in, preset_UID = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[4].to(device)
            
            with torch.no_grad():
                v_out = model_parallel(x_in)

            full_preset_out, actions = preset_processor(v_out, deterministic)
            inferred_wavs = audio_renderer.multi_process_render(full_preset_out)
            inferred_wavs = inferred_wavs / torch.abs(inferred_wavs + 1e-5).max(dim=1)[0].unsqueeze(1)
            sc, log_mae, mfcc_mae, _ = spec_processor.calculate_metrics(x_wav, inferred_wavs)
            rewards = calculate_rewards(sc, log_mae, mfcc_mae, ft_config.loss.pg_logp_threshold)
            buffer.store(preset_UID, actions, rewards)

        deterministic = False

    saved_buffer_path = model_path.joinpath('buffer', ft_config.train.dataset)
    buffer_path = logger.log_dir.joinpath('buffer', ft_config.train.dataset)
    os.makedirs(buffer_path, exist_ok=True)
    
    if ft_config.train.expl_epoch > 0:
        buffer.save(buffer_path)
    else:
        buffer.load(saved_buffer_path)

    rewards_top = {'value': np.zeros(10), 'epoch': np.zeros(10, dtype=np.int16)}

    # Train
    for epoch in tqdm(range(ft_config.train.start_epoch, ft_config.train.n_epochs), desc='epoch', position=0):
        model_parallel.train()
        dataloader_iter = iter(dataloader['train'])

        for _, s in scalars_train.items():
            s.on_new_epoch()

        # LR warmup (bypasses the scheduler during first epochs)
        lr_epoch = epoch - ft_config.train.start_epoch
        if lr_epoch <= ft_config.scheduler.warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = scalars_train['Sched/LRwarmup'].get(lr_epoch) * ft_config.optim.initial_lr

        for i in tqdm(range(len(dataloader['train'])), desc='training batch', position=1, leave=False):
            sample = next(dataloader_iter)
            x_wav, x_in, v_in, preset_UID = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[4].to(device)
            optimizer.zero_grad()
            v_out = model_parallel(x_in)

            if ft_config.loss.pg:
                full_preset_out, actions = preset_processor(v_out, deterministic=False)
                inferred_wavs = audio_renderer.multi_process_render(full_preset_out)
                inferred_wavs = inferred_wavs / torch.abs(inferred_wavs + 1e-5).max(dim=1)[0].unsqueeze(1)
                sc, log_mae, mfcc_mae, _ = spec_processor.calculate_metrics(x_wav, inferred_wavs)
                rewards = calculate_rewards(sc, log_mae, mfcc_mae, ft_config.loss.pg_logp_threshold)

                if ft_config.loss.per:
                    buffer.store(preset_UID, actions, rewards)
                    actions, rewards = buffer.sample(preset_UID)
                    
                mean_log_probs = preset_processor.get_mean_log_probs(v_out, actions, importance_sampling=False)
                pg_loss = -(rewards * mean_log_probs).mean()
                
                c = ft_config.loss.pg_coef
                alpha = linear_scheduler(epoch, c['s_value'], c['e_value'], c['s_epoch'], c['e_epoch'])
                scalars_train['Specs/LogProb/Train'].append(mean_log_probs.mean().item())
                scalars_train['Specs/PGLoss/Train'].append(pg_loss.item())
                scalars_train['Specs/SC/Train'].append(sc.mean().item())
                scalars_train['Specs/logMAE/Train'].append(log_mae.mean().item())
                scalars_train['Specs/mfccMAE/Train'].append(mfcc_mae.mean().item())
                scalars_train['Specs/Rewards/Train'].append(rewards.mean().item())
            else:
                pg_loss = torch.zeros(1).to(device)
                alpha = 0

            if ft_config.loss.param:
                cont_loss = controls_criterion(v_out, v_in)
                scalars_train['Controls/ParamLoss/Train'].append(cont_loss.item())
            else:
                cont_loss = torch.zeros(1).to(device)
                alpha = 1

            if ft_config.train.dataset == 'dexed':
                with torch.no_grad():
                    scalars_train['Controls/QLoss/Train'].append(controls_num_eval_criterion(v_out, v_in))
                    scalars_train['Controls/Accuracy/Train'].append(controls_accuracy_criterion(v_out, v_in))
            
            loss = alpha * pg_loss + (1 - alpha) * cont_loss
            loss.backward()
            optimizer.step()

        scalars_train['Sched/LR'] = SimpleMetric(optimizer.param_groups[0]['lr'])
        scheduler.step()

        for k, s in scalars_train.items():
            logger.tensorboard.add_scalar(k, s.get(), epoch)

        # Evaluation on validation dataset
        if epoch % ft_config.train.eval_period == 0:
            model_parallel.eval()

            for _, s in scalars_valid.items():
                s.on_new_epoch()
            
            for i, sample in tqdm(enumerate(dataloader['validation']), desc='validation batch', position=1, total=len(dataloader['validation']), leave=False):
                x_wav, x_in, v_in = sample[0].to(device), sample[1].to(device), sample[2].to(device)
                
                with torch.no_grad():
                    v_out = model_parallel(x_in)

                    if ft_config.loss.pg:
                        full_preset_out, actions = preset_processor(v_out, deterministic=True)
                        inferred_wavs = audio_renderer.multi_process_render(full_preset_out)
                        inferred_wavs = inferred_wavs / torch.abs(inferred_wavs + 1e-5).max(dim=1)[0].unsqueeze(1)
                        sc, log_mae, mfcc_mae, _ = spec_processor.calculate_metrics(x_wav, inferred_wavs)
                        rewards = calculate_rewards(sc, log_mae, mfcc_mae, ft_config.loss.pg_logp_threshold)
                        scalars_valid['Specs/SC/Valid'].append(sc.mean().item())
                        scalars_valid['Specs/logMAE/Valid'].append(log_mae.mean().item())
                        scalars_valid['Specs/mfccMAE/Valid'].append(mfcc_mae.mean().item())
                        scalars_valid['Specs/Rewards/Valid'].append(rewards.mean().item())

                    if ft_config.loss.param:
                        cont_loss = controls_criterion(v_out, v_in)
                        scalars_valid['Controls/ParamLoss/Valid'].append(cont_loss.item())
                
                if ft_config.train.dataset == 'dexed':
                    # Monitoring loss
                    scalars_valid['Controls/QLoss/Valid'].append(controls_num_eval_criterion(v_out, v_in))
                    scalars_valid['Controls/Accuracy/Valid'].append(controls_accuracy_criterion(v_out, v_in))

            for k, s in scalars_valid.items():
                logger.tensorboard.add_scalar(k, s.get(), epoch)

            if scalars_valid['Specs/Rewards/Valid'].get() > rewards_top['value'].min():
                update_idx = rewards_top['value'].argmin()
                file_path = logger.checkpoints_dir.joinpath('{:05d}.tar'.format(rewards_top['epoch'][update_idx]))
                if os.path.exists(file_path):
                    os.remove(file_path)
                rewards_top['value'][update_idx] = scalars_valid['Specs/Rewards/Valid'].get()
                rewards_top['epoch'][update_idx] = epoch
                logger.save_checkpoint(epoch, model, optimizer, scheduler)
                buffer.save(buffer_path)
                
        if epoch == ft_config.train.n_epochs - 1:
            logger.save_checkpoint(epoch, model, optimizer, scheduler)
        

        logger.on_epoch_finished(epoch)

    disp.stop()
    logger.on_training_finished()
    print('Finetuning process finished')
