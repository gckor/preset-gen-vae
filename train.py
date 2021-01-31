"""
Performs training for the configuration described in config.py
"""

import sys
import os
from pathlib import Path
import contextlib

import mkl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
from torch.autograd import profiler

import config
from model import VAE, encoder, decoder
import model.loss
import logs.logger
import logs.metrics
import data.dataset
import utils.data
import utils.profile


# ========== Datasets and DataLoaders ==========
full_dataset = data.dataset.DexedDataset(note_duration=config.model.note_duration,
                                         n_fft=config.model.stft_args[0], fft_hop=config.model.stft_args[1])
# dataset and dataloader are dicts with 'train', 'validation' and 'test' keys
dataset = utils.data.random_split(full_dataset, config.train.datasets_proportions, random_gen_seed=0)
dataloader = dict()
_debugger = False
if config.train.profiler_args['enabled'] and config.train.profiler_args['use_cuda']:
    num_workers = 0  # CUDA PyTorch profiler does not work with a multiprocess-dataloader
elif sys.gettrace() is not None:
    _debugger = True
    print("Debugger detected - num_workers=0 for all DataLoaders")
    num_workers = 0  # PyCharm debug behaves badly with multiprocessing...
else:  # We should use an higher CPU count for real-time audio rendering
    num_workers = min(config.train.minibatch_size, torch.cuda.device_count() * 4)  # Optimal w/ light dataloader
for dataset_type in dataset:
    dataloader[dataset_type] = DataLoader(dataset[dataset_type], config.train.minibatch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=True)
    if config.train.verbosity >= 1:
        print("Dataset '{}' contains {}/{} samples ({:.1f}%). num_workers={}".format(dataset_type, len(dataset[dataset_type]), len(full_dataset), 100.0 * len(dataset[dataset_type])/len(full_dataset), num_workers))


# ========== Model definition ==========
# Encoder and decoder with the same architecture
encoder_model = encoder.SpectrogramEncoder(config.model.encoder_architecture, config.model.dim_z,
                                           config.model.spectrogram_size)
decoder_model = decoder.SpectrogramDecoder(config.model.encoder_architecture, config.model.dim_z,
                                           config.model.spectrogram_size)
ae_model = VAE.BasicVAE(encoder_model, config.model.dim_z, decoder_model)  # Not parallelized yet


# ========== Logger init ==========
ae_model.eval()
logger = logs.logger.RunLogger(Path(__file__).resolve().parent, config.model, config.train)
logger.init_with_model(ae_model, config.model.input_tensor_size)  # model must not be parallel


# ========== Training devices (GPU(s) only) ==========
if config.train.verbosity >= 1:
    print("Intel MKL num threads = {}. PyTorch num threads = {}. CUDA devices count: {} GPU(s)."
          .format(mkl.get_max_threads(), torch.get_num_threads(), torch.cuda.device_count()))
if torch.cuda.device_count() == 0:
    raise NotImplementedError()  # CPU training not available
elif torch.cuda.device_count() == 1 or config.train.profiler_1_GPU:
    if config.train.profiler_1_GPU:
        print("Using 1/{} GPUs for code profiling".format(torch.cuda.device_count()))
    device = 'cuda:0'
    ae_model = ae_model.to(device)
    ae_model_parallel = nn.DataParallel(ae_model, device_ids=[0])  # "Parallel" 1-GPU model
else:
    device = torch.device('cuda')
    ae_model.to(device)
    ae_model_parallel = nn.DataParallel(ae_model)  # We use all available GPUs


# ========== Losses (criterion functions) ==========
if config.train.ae_reconstruction_loss == 'MSE':
    reconstruction_criterion = nn.MSELoss(reduction='mean')
else:
    raise NotImplementedError()
if config.train.latent_loss == 'Dkl':
    latent_criterion = model.loss.GaussianDkl(normalize=config.train.normalize_latent_loss)
else:
    raise NotImplementedError()


# ========== Scalars, metrics, images and audio to be tracked in Tensorboard ==========
epoch_losses = {'ReconsLoss/Train': logs.metrics.EpochMetric(), 'ReconsLoss/Valid': logs.metrics.EpochMetric(),
                'LatLoss/Train': logs.metrics.EpochMetric(), 'LatLoss/Valid': logs.metrics.EpochMetric()}
# TODO learning rate scalar, ...
# Losses here are Validation losses. Metrics need an '_' to be different from scalars (tensorboard mixes them)
metrics = {'ReconsLoss/Valid_': logs.metrics.BufferedMetric(),
           'LatLoss/Valid_': logs.metrics.BufferedMetric(),
           'epochs': 0}
# TODO check metrics as required in config.py
logger.tensorboard.init_hparams_and_metrics(metrics)


# ========== Optimizer and Scheduler ==========
ae_model.train()
optimizer = torch.optim.Adam(ae_model.parameters())


# ========== PyTorch Profiling (optional) ==========
is_profiled = config.train.profiler_args['enabled']
ae_model.is_profiled = is_profiled


# ========== Model training epochs ==========
# TODO consider re-start epoch
for epoch in range(0, config.train.n_epochs):
    # = = = = = Re-init of epoch metrics = = = = =
    for _, l in epoch_losses.items():
        l.on_new_epoch()

    # = = = = = Train all mini-batches (optional profiling) = = = = =
    # when profiling is disabled: true no-op context manager, and prof is None
    with utils.profile.get_optional_profiler(config.train.profiler_args) as prof:
        ae_model.train()
        dataloader_iter = iter(dataloader['train'])
        for i in range(len(dataloader['train'])):
            with profiler.record_function("DATA_LOAD") if is_profiled else contextlib.nullcontext():
                sample = next(dataloader_iter)
                x_in, params_in, midi_in = sample[0].to(device), sample[1].to(device), sample[2].to(device)
            optimizer.zero_grad()
            z_mu_logvar, z_sampled, x_out = ae_model_parallel(x_in)
            with profiler.record_function("BACKPROP") if is_profiled else contextlib.nullcontext():
                recons_loss = reconstruction_criterion(x_out, x_in)
                epoch_losses['ReconsLoss/Train'].append(recons_loss)
                lat_loss = latent_criterion(z_mu_logvar[:, 0, :], z_mu_logvar[:, 1, :])
                epoch_losses['LatLoss/Train'].append(lat_loss)
                (recons_loss + lat_loss).backward()
            with profiler.record_function("OPTIM_STEP") if is_profiled else contextlib.nullcontext():
                optimizer.step()  # TODO refaire propre
            logger.on_minibatch_finished(i)
            # For full-trace profiling: we need to stop after a few mini-batches
            if config.train.profiler_full_trace and i == 2:
                break
    if prof is not None:
        logger.save_profiler_results(prof, config.train.profiler_full_trace)
    if config.train.profiler_full_trace:
        break  # Forced training stop

    # TODO = = = = = Evaluation on validation dataset = = = = =
    with torch.no_grad():
        ae_model_parallel.eval()  # BN stops running estimates
        for i, sample in enumerate(dataloader['validation']):
            # TODO
            epoch_losses['ReconsLoss/Valid'].append(-10.0 * epoch)
            epoch_losses['LatLoss/Valid'].append(-10.0 * epoch)

    # TODO  = = = = = Epoch logs (scalars/sounds/images + updated metrics) = = = = =
    # TODO all scalars
    for k, l in epoch_losses.items():
        logger.tensorboard.add_scalar(k, l.mean(), epoch)
    # TODO metrics...
    metrics['epochs'] = epoch + 1
    metrics['ReconsLoss/Valid_'].append(epoch_losses['ReconsLoss/Valid'].mean())
    metrics['LatLoss/Valid_'].append(epoch_losses['LatLoss/Valid'].mean())
    logger.tensorboard.update_metrics(metrics)
    # TODO Spectrograms

    # TODO  = = = = = Model save - ready for next epoch = = = = =
    logger.on_epoch_finished(epoch, ae_model)  # TODO


# ========== Logger final stats ==========
logger.on_training_finished()
