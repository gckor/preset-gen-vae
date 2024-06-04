import os
import shutil
import json
import datetime
import pathlib
import humanize
import torch
import torchinfo
import numpy as np
from omegaconf import OmegaConf

from logs.tbwriter import TensorboardSummaryWriter  # Custom modified summary writer
from logs.metrics import SimpleMetric, EpochMetric, LatentMetric
from utils.hparams import LinearDynamicParam


def get_model_run_directory(root_path, config):
    """ Returns the directory where saved models and config.json are stored, for a particular run.
    Does not check whether the directory exists or not (it must have been created by the RunLogger) """
    return root_path.joinpath(config.model.name).joinpath(config.model.run_name)


def get_model_checkpoint(root_path: pathlib.Path, config, epoch, device=None):
    """ Returns the path to a .tar saved checkpoint, or prints all available checkpoints and raises an exception
    if the required epoch has no saved .tar checkpoint. """
    checkpoints_dir = root_path.joinpath(config.model.name)\
        .joinpath(config.model.run_name).joinpath('checkpoints')
    checkpoint_path = checkpoints_dir.joinpath('{:05d}.tar'.format(epoch))
    try:
        if device is None:
            checkpoint = torch.load(checkpoint_path)  # Load on original device
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)  # e.g. train on GPU, load on CPU
    except (OSError, IOError) as e:
        available_checkpoints = "Available checkpoints: {}".format([f.name for f in checkpoints_dir.glob('*.tar')])
        print(available_checkpoints)
        raise ValueError("Cannot load checkpoint for epoch {}: {}".format(epoch, e))
    return checkpoint


def get_model_last_checkpoint(root_path: pathlib.Path, config, verbose=True, device=None):
    checkpoints_dir = root_path.joinpath(config.model.name)\
        .joinpath(config.model.run_name).joinpath('checkpoints')
    available_epochs = [int(f.stem) for f in checkpoints_dir.glob('*.tar')]
    assert len(available_epochs) > 0  # At least 1 checkpoint should be available
    if verbose:
        print("Loading epoch {} from {}".format(max(available_epochs), checkpoints_dir))
    return get_model_checkpoint(root_path, config, max(available_epochs), device)


def erase_run_data(root_path, config):
    """ Erases all previous data (Tensorboard, config, saved models)
    for a particular run of the model. """
    print("[RunLogger] '{}' run for model '{}' will be erased.".format(config.model.run_name, config.model.name))
    shutil.rmtree(get_model_run_directory(root_path, config))


def get_scalars(config):
    scalars = dict()
    scalars['ReconsLoss/Backprop/Train'] = EpochMetric()
    scalars['ReconsLoss/Backprop/Valid'] = EpochMetric()
    scalars['ReconsLoss/MSE/Train'] = EpochMetric()
    scalars['ReconsLoss/MSE/Valid'] = EpochMetric()
    scalars['MSSpecLoss/Backprop/Train'] = EpochMetric()
    scalars['MSSpecLoss/Backprop/Valid'] = EpochMetric()
    scalars['ContrastiveLoss/Backprop/Train'] = EpochMetric()
    scalars['ContrastiveLoss/Backprop/Valid'] = EpochMetric()
    scalars['Controls/BackpropLoss/Train'] = EpochMetric()
    scalars['Controls/BackpropLoss/Valid'] = EpochMetric()
    scalars['Controls/QLoss/Train'] = EpochMetric()
    scalars['Controls/QLoss/Valid'] = EpochMetric()
    scalars['Controls/Accuracy/Train'] = EpochMetric()
    scalars['Controls/Accuracy/Valid'] = EpochMetric()
    scalars['Sched/LR'] = SimpleMetric(config.train.initial_learning_rate)
    scalars['Sched/LRwarmup'] = LinearDynamicParam(
        start_value=config.train.lr_warmup_start_factor,
        end_value=1.0,
        end_epoch=config.train.lr_warmup_epochs,
        current_epoch=config.train.start_epoch
    )
    scalars['Sched/beta'] = LinearDynamicParam(
        start_value=config.train.beta_start_value,
        end_value=config.train.beta,
        end_epoch=config.train.beta_warmup_epochs,
        current_epoch=config.train.start_epoch
    )
    return scalars


class RunLogger:
    """ Class for saving interesting data during a training run:
     - graphs, losses, metrics, and some results to Tensorboard
     - config.py as a json file
     - trained models

     See ../README.md to get more info on storage location.
     """
    def __init__(self, root_path, config, minibatches_count=0):
        """

        :param root_path: pathlib.Path of the project's root folder
        :param model_config: from config.py
        :param train_config: from config.py
        :param minibatches_count: Length of the 'train' dataloader
        """
        # Configs are stored but not modified by this class
        self.config = config
        self.verbosity = config.verbosity
        self.restart_from_checkpoint = (config.train.start_epoch > 0)

        # Directories creation (if not exists) for model
        self.log_dir = root_path.joinpath(config.model.name, config.model.run_name)
        self.checkpoints_dir = self.log_dir.joinpath('checkpoints')
        self._make_dirs_if_dont_exist(self.checkpoints_dir)

        if self.verbosity >= 1:
            print("[RunLogger] Starting logging into '{}'".format(self.log_dir))

        # If run folder already exists
        if self.restart_from_checkpoint:
            print("[RunLogger] Will load saved checkpoint (previous epoch: {})"
                    .format(self.config.train.start_epoch - 1))
        else: # Start a new fresh training
            if not config.allow_erase_run:
                raise RuntimeError("Config does not allow to erase the '{}' run for model '{}'"
                                    .format(config.model.run_name, config.model.name))
            else:
                erase_run_data(root_path, config)
                self._make_model_run_dirs()

        # Epochs, Batches, ...
        self.minibatches_count = minibatches_count
        self.minibatch_duration_running_avg = 0.0
        self.minibatch_duration_avg_coeff = 0.05  # auto-regressive running average coefficient
        self.last_minibatch_start_datetime = datetime.datetime.now()
        self.epoch_start_datetimes = [datetime.datetime.now()]  # This value can be erased in init_with_model
        
        # Tensorboard
        self.tensorboard = TensorboardSummaryWriter(
            log_dir=self.log_dir,
            flush_secs=5,
            config=config,
        )

    @staticmethod
    def _make_dirs_if_dont_exist(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _make_model_run_dirs(self):
        """
        Creates (no check) the directories for storing config and saved models.
        """
        os.makedirs(self.checkpoints_dir)

    def init_with_model(self, main_model, input_tensor_size):
        """
        Finishes to initialize this logger given the fully-build model.
        This function must be called after all checks (configuration consistency, etc...)
        have been performed, because it overwrites files.
        """
        # Write config file on startup only - any previous config file will be erased
        # New/Saved configs compatibility must have been checked before calling this function
        OmegaConf.save(self.config, self.log_dir.joinpath('config.yaml'))

        if not self.restart_from_checkpoint:  # Graphs written at epoch 0 only
            self.write_model_summary(main_model, input_tensor_size, 'VAE')

        self.epoch_start_datetimes = [datetime.datetime.now()]

    def write_model_summary(self, model, input_tensor_size, model_name):
        if not self.restart_from_checkpoint:  # Graphs written at epoch 0 only
            description = torchinfo.summary(model, tuple(input_tensor_size), depth=5, device='cpu', verbose=0)
            
            with open(self.log_dir.joinpath('torchinfo_summary_{}.txt'.format(model_name)), 'w') as f:
                f.write(description.__str__())

    def get_previous_config(self):
        full_config = OmegaConf.load(self.log_dir.joinpath('config.yaml'))
        return full_config

    def save_checkpoint(self, epoch, ae_model, optimizer, scheduler):
        torch.save({'epoch': epoch, 'ae_model_state_dict': ae_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()},
                   self.checkpoints_dir.joinpath('{:05d}.tar'.format(epoch)))

    def on_epoch_finished(self, epoch):
        self.epoch_start_datetimes.append(datetime.datetime.now())
        epoch_duration = self.epoch_start_datetimes[-1] - self.epoch_start_datetimes[-2]
        avg_duration_s = np.asarray([(self.epoch_start_datetimes[i+1] - self.epoch_start_datetimes[i]).total_seconds()
                                     for i in range(len(self.epoch_start_datetimes) - 1)])
        avg_duration_s = avg_duration_s.mean()
        run_total_epochs = self.config.train.n_epochs - self.config.train.start_epoch
        remaining_datetime = avg_duration_s * (run_total_epochs - (epoch - self.config.train.start_epoch) - 1)
        remaining_datetime = datetime.timedelta(seconds=int(remaining_datetime))
        
        if self.verbosity >= 1:
            print("End of epoch {} ({}/{}). Duration={:.1f}s, avg={:.1f}s. Estimated remaining time: {} ({})"
                  .format(epoch, epoch-self.config.train.start_epoch + 1, run_total_epochs,
                          epoch_duration.total_seconds(), avg_duration_s,
                          remaining_datetime, humanize.naturaldelta(remaining_datetime)))

    def on_training_finished(self):
        self.tensorboard.flush()
        self.tensorboard.close()
        if self.verbosity >= 1:
            print("[RunLogger] Training has finished")
