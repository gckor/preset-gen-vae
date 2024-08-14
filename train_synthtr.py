"""
This script performs a single training run for the configuration described
in config.py, when running as __main__.

Its train_config(...) function can also be called from another script,
with small modifications to the config (enqueued train runs).

See train_queue.py for enqueued training runs
"""

from pathlib import Path
from tqdm import tqdm

import mkl
import torch
import torch.nn as nn
import torch.optim
from pyvirtualdisplay import Display

import logs.logger
import logs.metrics
from logs.metrics import SimpleMetric, EpochMetric
import data.dataset
import data.build
import utils.figures
import utils.exception
from utils.audio import AudioEvaluator
from utils.scheduler import get_scheduler
from utils.distrib import get_parallel_devices
from config import load_config
from model.encoder import SynthTR
from model.loss import SynthParamsLoss, QuantizedNumericalParamsLoss, CategoricalParamsAccuracy, PresetProcessor, calculate_rewards
from utils.hparams import LinearDynamicParam


def train_config():
    """
    Performs a full training run, as described by parameters in config.py.
    Some attributes from config.py might be dynamically changed by train_queue.py (or this script,
    after loading the datasets) - so they can be different from what's currently written in config.py.
    """
    config = load_config()
    dataset = data.build.get_dataset(config)
    dataloader = data.build.get_split_dataloaders(config, dataset)
    root_path = Path(config.logs_root_dir)
    logger = logs.logger.RunLogger(root_path, config)

    # Synth parameter index information for alignment
    preset_idx_helper = dataset.preset_indexes_helper

    # Initialize the model
    model = SynthTR(preset_idx_helper, **config.model.encoder_kwargs)

    config.model.input_tensor_size = (
        config.train.minibatch_size,
        config.model.encoder_kwargs.spectrogram_channels,
        config.model.spectrogram_size[0],
        config.model.spectrogram_size[1]
    )
    model.eval()
    logger.init_with_model(model, config.model.input_tensor_size)

    # Training devices (GPU(s) only)
    if config.verbosity >= 1:
        print("Intel MKL num threads = {}. PyTorch num threads = {}. CUDA devices count: {} GPU(s)."
              .format(mkl.get_max_threads(), torch.get_num_threads(), torch.cuda.device_count()))
    
    device, device_ids = get_parallel_devices(config.main_cuda_device_idx)
    model = model.to(device)
    model_parallel = nn.DataParallel(model, device_ids=device_ids, output_device=device)

    # Parameter loss
    controls_criterion = SynthParamsLoss(
        preset_idx_helper,
        config.train.normalize_losses,
        cat_softmax_t=config.train.params_cat_softmax_temperature,
        label_smoothing=config.train.params_label_smoothing,
    )

    # Policy gradient loss
    if config.train.pg_loss:
        preset_processor = PresetProcessor(dataset, preset_idx_helper)
        audio_evaluator = AudioEvaluator(dataset, config.train.audio_eval_n_workers, device)

    # Monitoring losses always remain the same
    controls_num_eval_criterion = QuantizedNumericalParamsLoss(preset_idx_helper, numerical_loss=nn.MSELoss(reduction='mean'))
    controls_accuracy_criterion = CategoricalParamsAccuracy(preset_idx_helper, reduce=True, percentage_output=True)

    # Scalars, metrics, images and audio to be tracked in Tensorboard
    scalars_train = dict()
    scalars_valid = dict()
    scalars_train['Controls/BackpropLoss/Train'] = EpochMetric()
    scalars_valid['Controls/BackpropLoss/Valid'] = EpochMetric()
    scalars_train['Controls/QLoss/Train'] = EpochMetric()
    scalars_valid['Controls/QLoss/Valid'] = EpochMetric()
    scalars_train['Controls/Accuracy/Train'] = EpochMetric()
    scalars_valid['Controls/Accuracy/Valid'] = EpochMetric()

    if config.train.pg_loss:
        scalars_train['Specs/LogProb/Train'] = EpochMetric()
        scalars_valid['Specs/LogProb/Valid'] = EpochMetric()
        scalars_train['Specs/SpecMAE/Train'] = EpochMetric()
        scalars_valid['Specs/SpecMAE/Valid'] = EpochMetric()
        scalars_train['Specs/PGLoss/Train'] = EpochMetric()
        scalars_valid['Specs/PGLoss/Valid'] = EpochMetric()
        scalars_train['Specs/RewardSampleNum'] = EpochMetric()

    scalars_train['Sched/LR'] = SimpleMetric(config.train.initial_learning_rate)
    scalars_train['Sched/LRwarmup'] = LinearDynamicParam(
        start_value=config.train.lr_warmup_start_factor,
        end_value=1.0,
        end_epoch=config.train.lr_warmup_epochs,
        current_epoch=config.train.start_epoch
    )

    # Optimizer and Scheduler
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.initial_learning_rate,
        weight_decay=config.train.weight_decay,
        betas=config.train.adam_betas
    )
    scheduler = get_scheduler(config.train.scheduler_name, optimizer, **config.train.scheduler_kwargs)

    disp = Display()
    disp.start()

    # Model training epochs
    for epoch in tqdm(range(config.train.start_epoch, config.train.n_epochs), desc='epoch', position=0):

        # Re-init of epoch metrics and useful scalars (warmup ramps, ...)
        for _, s in scalars_train.items():
            s.on_new_epoch()

        # LR warmup (bypasses the scheduler during first epochs)
        if epoch <= config.train.lr_warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = scalars_train['Sched/LRwarmup'].get(epoch) * config.train.initial_learning_rate

        # Train all mini-batches
        model_parallel.train()
        dataloader_iter = iter(dataloader['train'])
        
        for i in tqdm(range(len(dataloader['train'])), desc='training batch', position=1, leave=False):
            sample = next(dataloader_iter)
            x_wav, x_in, v_in, sample_info = sample[0].numpy(), sample[1].to(device), sample[2].to(device), sample[3].numpy()
            optimizer.zero_grad()
            v_out = model_parallel(x_in)

            if config.train.pg_loss:
                full_preset_out, mean_log_probs = preset_processor(v_out)
                spec_maes = audio_evaluator.multi_process_measure(x_wav, full_preset_out, sample_info)
                rewards = calculate_rewards(spec_maes, config.train.pg_logp_threshold)
                pg_loss = -(rewards * mean_log_probs).mean()
                scalars_train['Specs/LogProb/Train'].append(mean_log_probs.mean().item())
                scalars_train['Specs/SpecMAE/Train'].append(spec_maes.mean().item())
                scalars_train['Specs/PGLoss/Train'].append(pg_loss.item() * config.train.pg_loss_coef)
                scalars_train['Specs/RewardSampleNum'].append((spec_maes < config.train.pg_logp_threshold).sum().item())
                if pg_loss > 1:
                    print(f'spec_maes: {spec_maes[rewards > 0]}')
                    print(f'rewards: {rewards[rewards > 0]}')
                    print(f'mean_log_probs: {mean_log_probs[rewards > 0]}')
                    print(f'pg_loss: {pg_loss}')
            else:
                pg_loss = torch.zeros(1).to(device)

            # Monitoring losses
            with torch.no_grad():
                scalars_train['Controls/QLoss/Train'].append(controls_num_eval_criterion(v_out, v_in))
                scalars_train['Controls/Accuracy/Train'].append(controls_accuracy_criterion(v_out, v_in))
            
            cont_loss = controls_criterion(v_out, v_in)

            # Log backpropagation losses
            scalars_train['Controls/BackpropLoss/Train'].append(cont_loss)

            # Update parameters
            loss = config.train.pg_loss_coef * pg_loss + cont_loss
            utils.exception.check_nan_values(epoch, cont_loss, pg_loss)
            loss.backward()
            optimizer.step()

        # Evaluation on validation dataset
        if epoch % config.train.eval_period == 0:
            for _, s in scalars_valid.items():
                s.on_new_epoch()

            with torch.no_grad():
                model_parallel.eval()  # BN stops running estimates
                for i, sample in tqdm(enumerate(dataloader['validation']), desc='validation batch', position=1, total=len(dataloader['validation']), leave=False):
                    x_wav, x_in, v_in, sample_info = sample[0].numpy(), sample[1].to(device), sample[2].to(device), sample[3].numpy()
                    v_out = model_parallel(x_in)

                    if config.train.pg_loss:
                        full_preset_out, mean_log_probs = preset_processor(v_out)
                        spec_maes = audio_evaluator.multi_process_measure(x_wav, full_preset_out, sample_info)
                        rewards = calculate_rewards(spec_maes, config.train.pg_logp_threshold)
                        pg_loss = -(rewards * mean_log_probs).mean()
                        scalars_valid['Specs/LogProb/Valid'].append(mean_log_probs.mean().item())
                        scalars_valid['Specs/SpecMAE/Valid'].append(spec_maes.mean().item())
                        scalars_valid['Specs/PGLoss/Valid'].append(pg_loss.item() * config.train.pg_loss_coef)

                    # Loss
                    cont_loss = controls_criterion(v_out, v_in)
                    
                    # Monitoring losses
                    scalars_valid['Controls/QLoss/Valid'].append(controls_num_eval_criterion(v_out, v_in))
                    scalars_valid['Controls/Accuracy/Valid'].append(controls_accuracy_criterion(v_out, v_in))

                    # Log backpropagation valid losses
                    scalars_valid['Controls/BackpropLoss/Valid'].append(cont_loss)

            for k, s in scalars_valid.items():
                logger.tensorboard.add_scalar(k, s.get(), epoch)

        # Dynamic LR scheduling depends on validation performance
        # Summed losses for plateau-detection are chosen in config.py
        scalars_train['Sched/LR'] = logs.metrics.SimpleMetric(optimizer.param_groups[0]['lr'])

        if config.train.scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(sum([scalars_valid['{}/Valid'.format(loss_name)].get() for loss_name in config.train.scheduler_loss]))
        else:
            scheduler.step()

        early_stop = (optimizer.param_groups[0]['lr'] < config.train.early_stop_lr_threshold)

        # Epoch logs (scalars/sounds/images + updated metrics)
        for k, s in scalars_train.items():  # All available scalars are written to tensorboard
            logger.tensorboard.add_scalar(k, s.get(), epoch)

        # Model+optimizer(+scheduler) save - ready for next epoch
        if (epoch > 0 and epoch % config.train.save_period == 0)\
                or (epoch == config.train.n_epochs-1) or early_stop:
            logger.save_checkpoint(epoch, model, optimizer, scheduler)

        logger.on_epoch_finished(epoch)

        if early_stop:
            print("[train.py] Training stopped early (final loss plateau)")
            break

    # Logger final stats
    logger.on_training_finished()


    # "Manual GC" (to try to prevent random CUDA out-of-memory between enqueued runs
    disp.stop()
    del scheduler, optimizer
    del model_parallel
    del model
    del logger
    del dataloader, dataset


if __name__ == "__main__":
    # Normal run, config.py only will be used to parametrize learning and models
    train_config()

