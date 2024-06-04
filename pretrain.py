import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

import logs
from utils.distrib import get_parallel_devices
from data import build
from model.audiomae.models_mae import MaskedAutoencoderViT
from logs.logger import get_scalars


if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')
    dataset = build.get_dataset(config)
    dataloader, sub_datasets_lengths = build.get_split_dataloaders(config, dataset)
    root_path = Path(config.logs_root_dir)
    logger = logs.logger.RunLogger(root_path, config)

    model = MaskedAutoencoderViT(img_size=config.model.spectrogram_size)
    model.eval()
    config.model.input_tensor_size = [config.train.minibatch_size, 1] + config.model.spectrogram_size
    logger.init_with_model(model, config.model.input_tensor_size)
    
    device, device_ids = get_parallel_devices(config.main_cuda_device_idx)
    model = model.to(device)
    model_parallel = nn.DataParallel(
        model,
        device_ids=device_ids,
        output_device=device,
    )

    scalars = get_scalars(config)

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.initial_learning_rate,
        weight_decay=config.train.weight_decay,
        betas=config.train.adam_betas
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.train.scheduler_lr_factor,
        patience=config.train.scheduler_patience,
        cooldown=config.train.scheduler_cooldown,
        threshold=config.train.scheduler_threshold,
        verbose=(config.verbosity >= 2)
    )
    

    for epoch in tqdm(range(config.train.start_epoch, config.train.n_epochs), desc='epoch', position=0):
        for _, s in scalars.items():
            s.on_new_epoch()

        if epoch <= config.train.lr_warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = scalars['Sched/LRwarmup'].get(epoch) * config.train.initial_learning_rate

        model_parallel.train()
        dataloader_iter = iter(dataloader['train'])

        # Train one epoch
        for i in tqdm(range(len(dataloader['train'])), desc='training batch', position=1, leave=False):
            sample = next(dataloader_iter)
            x_in = sample[1].to(device)
            loss_recon = model.get_mask_predict_loss(x_in)
            scalars['ReconsLoss/Backprop/Train'].append(loss_recon)

            optimizer.zero_grad()
            loss_recon.backward()
            optimizer.step()

        # Validation
        model_parallel.eval()

        for i, sample in tqdm(enumerate(dataloader['validation']), desc='validation batch', position=1, total=len(dataloader['validation']), leave=False):
            x_in = sample[1].to(device)

            with torch.no_grad():
                loss_recon = model.get_mask_predict_loss(x_in)
            
            scalars['ReconsLoss/Backprop/Valid'].append(loss_recon)

        scheduler.step(sum([scalars['{}/Valid'.format(loss_name)].get() for loss_name in config.train.scheduler_loss]))
        scalars['Sched/LR'] = logs.metrics.SimpleMetric(optimizer.param_groups[0]['lr'])

        # Write epoch logs to tensorboard
        for k, s in scalars.items():
            logger.tensorboard.add_scalar(k, s.get(), epoch)

        if epoch % config.train.save_period == 0 or epoch == config.train.n_epochs - 1:
            logger.save_checkpoint(epoch, model, optimizer, scheduler)

        logger.on_epoch_finished(epoch)

    logger.on_training_finished()
