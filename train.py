"""
This script performs a single training run for the configuration described
in config.py, when running as __main__.

Its train_config(...) function can also be called from another script,
with small modifications to the config (enqueued train runs).

See train_queue.py for enqueued training runs
"""

from pathlib import Path
import contextlib

import mkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import config
import model.loss
import model.build
import model.flows
import logs.logger
import logs.metrics
from logs.metrics import SimpleMetric, EpochMetric, LatentMetric
import data.dataset
import data.build
import utils.profile
from utils.hparams import LinearDynamicParam
import utils.figures
import utils.exception
from tqdm import tqdm


def train_config():
    """ Performs a full training run, as described by parameters in config.py.

    Some attributes from config.py might be dynamically changed by train_queue.py (or this script,
    after loading the datasets) - so they can be different from what's currently written in config.py. """


    # ========== Datasets and DataLoaders ==========
    # Must be constructed first because dataset output sizes will be required to automatically
    # infer models output sizes.
    # ********************* config.model.dim_z will be changed if a flow network is used **********************
    dataset = data.build.get_dataset(config.model, config.train)
    # dataloader is a dict of 3 subsets dataloaders ('train', 'validation' and 'test')
    dataloader, sub_datasets_lengths = data.build.get_split_dataloaders(config.train, dataset)


    # ========== Logger init (required to load from checkpoint) and Config check ==========
    root_path = Path(config.model.logs_root_dir)
    logger = logs.logger.RunLogger(root_path, config.model, config.train)
    if logger.restart_from_checkpoint:
        model.build.check_configs_on_resume_from_checkpoint(config.model, config.train,
                                                            logger.get_previous_config_from_json())
    if config.train.start_epoch > 0:  # Resume from checkpoint?
        start_checkpoint = logs.logger.get_model_checkpoint(root_path, config.model, config.train.start_epoch - 1)
    else:
        start_checkpoint = None


    # ========== Model definition (requires the full_dataset to be built) ==========
    _, _, _, extended_ae_model = model.build.build_extended_ae_model(config.model, config.train,
                                                                     dataset.preset_indexes_helper)
    if start_checkpoint is not None:
        extended_ae_model.load_state_dict(start_checkpoint['ae_model_state_dict'])  # GPU tensor params
    extended_ae_model.eval()
    # will write tensorboard graph and torchinfo txt summary. model must not be parallel
    logger.init_with_model(extended_ae_model, config.model.input_tensor_size)  # main model
    logger.write_model_summary(extended_ae_model.reg_model, (config.train.minibatch_size, config.model.dim_z),
                               "reg")  # Another summary write


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
        extended_ae_model = extended_ae_model.to(device)
        parallel_device_ids = [0]  # "Parallel" 1-GPU model
    else:
        device = torch.device('cuda:{}'.format(config.train.main_cuda_device_idx))
        extended_ae_model = extended_ae_model.to(device)
        # We use all available GPUs - the main one must be first in list
        parallel_device_ids = [i for i in range(torch.cuda.device_count()) if i != config.train.main_cuda_device_idx]
        parallel_device_ids.insert(0, config.train.main_cuda_device_idx)
    ae_model_parallel = nn.DataParallel(extended_ae_model, device_ids=parallel_device_ids, output_device=device)
    reg_model_parallel = nn.DataParallel(extended_ae_model.reg_model, device_ids=parallel_device_ids,
                                         output_device=device)


    # ========== Losses (criterion functions) ==========
    # Training losses (for backprop) and Metrics (monitoring) losses and accuracies
    # Some losses are defined in the models themselves
    if config.model.input_type == 'waveform':
        reconstruction_criterion = nn.L1Loss()
        msspec_transformer = model.loss.MultiScaleMelSpectrogramLoss(config.model.sampling_rate,
                                                                     f_min=64, normalized=True, alphas=False).to(device)
        msspec_alphas = msspec_transformer.alphas
        msspec_transformer = nn.DataParallel(msspec_transformer, device_ids=parallel_device_ids, output_device=device)
    elif config.train.normalize_losses:  # Reconstruction backprop loss
        reconstruction_criterion = nn.MSELoss(reduction='mean')
    else:
        reconstruction_criterion = model.loss.L2Loss()
    # Controls backprop loss
    if config.model.forward_controls_loss:  # usual straightforward loss - compares inference and target
        if config.train.params_cat_bceloss:
            assert (not config.model.params_reg_softmax)  # BCE loss requires no-softmax at reg model output
        controls_criterion = model.loss.SynthParamsLoss(dataset.preset_indexes_helper,
                                                        config.train.normalize_losses,
                                                        cat_bce=config.train.params_cat_bceloss,
                                                        cat_softmax=(not config.model.params_reg_softmax
                                                                     and not config.train.params_cat_bceloss),
                                                        cat_softmax_t=config.train.params_cat_softmax_temperature)
    else:  # Inverse-flow-based loss
        controls_criterion = model.loss.FlowParamsLoss(dataset.preset_indexes_helper,
                                                       extended_ae_model.ae_model.flow_inverse_function,
                                                       extended_ae_model.reg_model.flow_inverse_function)

    # Monitoring losses always remain the same
    controls_num_eval_criterion = model.loss.QuantizedNumericalParamsLoss(dataset.preset_indexes_helper,
                                                                          numerical_loss=nn.MSELoss(reduction='mean'))
    controls_accuracy_criterion = model.loss.CategoricalParamsAccuracy(dataset.preset_indexes_helper,
                                                                       reduce=True, percentage_output=True)
    # Stabilizing loss for flow-based latent space
    flow_input_dkl = model.loss.GaussianDkl(normalize=config.train.normalize_losses)


    # ========== Scalars, metrics, images and audio to be tracked in Tensorboard ==========
    scalars = {  # Reconstruction loss (variable scale) + monitoring metrics comparable across all models
               'ReconsLoss/Backprop/Train': EpochMetric(), 'ReconsLoss/Backprop/Valid': EpochMetric(),
               'ReconsLoss/MSE/Train': EpochMetric(), 'ReconsLoss/MSE/Valid': EpochMetric(),
               # MSSpec loss
               'MSSpecLoss/Backprop/Train': EpochMetric(), 'MSSpecLoss/Backprop/Valid': EpochMetric(),
               # Contrastive loss
               'ContrastiveLoss/Backprop/Train': EpochMetric(), 'ContrastiveLoss/Backprop/Valid': EpochMetric(),
               # 'ReconsLoss/SC/Train': EpochMetric(), 'ReconsLoss/SC/Valid': EpochMetric(),  # TODO
               # Controls losses used for backprop + monitoring metrics (quantized numerical loss, categorical accuracy)
               'Controls/BackpropLoss/Train': EpochMetric(), 'Controls/BackpropLoss/Valid': EpochMetric(),
               'Controls/QLoss/Train': EpochMetric(), 'Controls/QLoss/Valid': EpochMetric(),
               'Controls/Accuracy/Train': EpochMetric(), 'Controls/Accuracy/Valid': EpochMetric(),       
               # Other misc. metrics
               'Sched/LR': SimpleMetric(config.train.initial_learning_rate),
               'Sched/LRwarmup': LinearDynamicParam(config.train.lr_warmup_start_factor, 1.0,
                                                    end_epoch=config.train.lr_warmup_epochs,
                                                    current_epoch=config.train.start_epoch),
               'Sched/beta': LinearDynamicParam(config.train.beta_start_value, config.train.beta,
                                                end_epoch=config.train.beta_warmup_epochs,
                                                current_epoch=config.train.start_epoch)}
    # Validation metrics have a '_' suffix to be different from scalars (tensorboard mixes them)
    metrics = {'ReconsLoss/MSE/Valid_': logs.metrics.BufferedMetric(),
               'Controls/QLoss/Valid_': logs.metrics.BufferedMetric(),
               'Controls/Accuracy/Valid_': logs.metrics.BufferedMetric(),
               'epochs': config.train.start_epoch}

    if config.model.stochastic_latent:
        # Latent-space and VAE losses
        scalars['LatLoss/Train'], scalars['LatLoss/Valid'] = EpochMetric(), EpochMetric()
        scalars['VAELoss/Train'], scalars['VAELoss/Valid'] = SimpleMetric(), SimpleMetric()
        scalars['LatCorr/Train'] = LatentMetric(config.model.dim_z, sub_datasets_lengths['train'])
        scalars['LatCorr/Valid'] = LatentMetric(config.model.dim_z, sub_datasets_lengths['validation'])
        metrics['LatLoss/Valid_'] = logs.metrics.BufferedMetric()
        metrics['LatCorr/Valid_'] = logs.metrics.BufferedMetric()
    
    logger.tensorboard.init_hparams_and_metrics(metrics)  # hparams added knowing config.*


    # ========== Optimizer and Scheduler ==========
    extended_ae_model.train()
    if config.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(extended_ae_model.parameters(), lr=config.train.initial_learning_rate,
                                     weight_decay=config.train.weight_decay, betas=config.train.adam_betas)
    else:
        raise NotImplementedError()
    if config.train.scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.\
            ReduceLROnPlateau(optimizer, factor=config.train.scheduler_lr_factor,
                              patience=config.train.scheduler_patience, cooldown=config.train.scheduler_cooldown,
                              threshold=config.train.scheduler_threshold, verbose=(config.train.verbosity >= 2))
    else:
        raise NotImplementedError()
    if start_checkpoint is not None:
        optimizer.load_state_dict(start_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(start_checkpoint['scheduler_state_dict'])


    # ========== Model training epochs ==========
    for epoch in tqdm(range(config.train.start_epoch, config.train.n_epochs), desc='epoch', position=0):
        # = = = = = Re-init of epoch metrics and useful scalars (warmup ramps, ...) = = = = =
        for _, s in scalars.items():
            s.on_new_epoch()
        should_plot = (epoch % config.train.plot_period == 0)

        # = = = = = LR warmup (bypasses the scheduler during first epochs) = = = = =
        if epoch <= config.train.lr_warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = scalars['Sched/LRwarmup'].get(epoch) * config.train.initial_learning_rate

        # = = = = = Train all mini-batches (optional profiling) = = = = =
        ae_model_parallel.train()
        dataloader_iter = iter(dataloader['train'])
        for i in tqdm(range(len(dataloader['train'])), desc='training batch', position=1, leave=False):
            sample = next(dataloader_iter)

            if config.model.input_type == 'waveform':
                x_in, v_in, sample_info = sample[0].to(device), sample[2].to(device), sample[3].to(device)
            else: # config.model.input_type == 'spectrogram'
                x_in, v_in, sample_info = sample[1].to(device), sample[2].to(device), sample[3].to(device)

            if config.model.contrastive:
                aug_specs = dataset.get_aug_specs(sample_info[:, 0]).to(device)
                x_in = torch.cat((x_in, aug_specs), dim=0)

            optimizer.zero_grad()
            ae_out = ae_model_parallel(x_in, sample_info)  # Spectral VAE - tuple output

            if config.model.stochastic_latent:
                z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac, x_out = ae_out
                scalars['LatCorr/Train'].append(z_0_mu_logvar, z_0_sampled)
            else:
                z_K_sampled, x_out = ae_out

            if config.model.contrastive:
                cross_entropy = nn.CrossEntropyLoss().to(device)
                logits, labels = model.loss.info_nce_loss(z_K_sampled, config.train.minibatch_size)
                contrastive_loss = cross_entropy(logits, labels)
                contrastive_loss = config.train.contrastive_coef * contrastive_loss
                z_K_sampled = z_K_sampled[:config.train.minibatch_size]
            else:
                contrastive_loss = torch.tensor([0], device=device)

            # Synth parameters regression. Flow-based: we do not care about v_out for backprop, but
            #     need it for monitoring (so we don't ask for the log abs det jacobian return)
            if isinstance(controls_criterion, model.loss.FlowParamsLoss):
                with torch.no_grad():
                    reg_model_parallel.eval()  # FIXME sub-optimal, for monitoring only...
                    v_out = reg_model_parallel(z_K_sampled)
                    reg_model_parallel.train()
            else:
                v_out = reg_model_parallel(z_K_sampled)
            
            if config.model.decoder_architecture is None:
                recons_loss = torch.tensor([0], device=device)
            else:
                recons_loss = reconstruction_criterion(x_out, x_in)

            if config.model.input_type == 'waveform' and config.model.encoder_architecture != 'encodec_pretrained':
                msspecs = msspec_transformer(x_in, x_out)
                msspec_loss = 0
                for j, specs in enumerate(msspecs):
                    s_x_1, s_y_1, s_x_2, s_y_2 = specs[:, 0], specs[:, 1], specs[:, 2], specs[:, 3]
                    msspec_loss += F.l1_loss(s_x_1, s_y_1) + msspec_alphas[j] * F.mse_loss(s_x_2, s_y_2)
                msspec_loss = msspec_loss / (2 * len(msspecs))
            else:
                msspec_loss = torch.tensor([0], device=device)

            # Latent loss computed on 1 GPU using the ae_model itself (not its parallelized version)
            if config.model.stochastic_latent:
                lat_loss = extended_ae_model.latent_loss(z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac)
                scalars['LatLoss/Train'].append(lat_loss)
                lat_loss *= scalars['Sched/beta'].get(epoch)
            else:
                lat_loss = torch.tensor([0], device=device)        

            # Monitoring losses
            with torch.no_grad():
                scalars['ReconsLoss/MSE/Train'].append(recons_loss if config.train.normalize_losses
                                                        or config.model.decoder_architecture is None
                                                        else F.mse_loss(x_out, x_in, reduction='mean'))
                scalars['Controls/QLoss/Train'].append(controls_num_eval_criterion(v_out, v_in))
                scalars['Controls/Accuracy/Train'].append(controls_accuracy_criterion(v_out, v_in))

            # Flow training stabilization loss?
            flow_input_loss = torch.tensor([0], device=device)
            if extended_ae_model.is_flow_based_latent_space and\
                    (config.train.latent_flow_input_regularization.lower() == 'dkl'):
                flow_input_loss = 0.1 * config.train.beta * flow_input_dkl(z_0_mu_logvar[:, 0, :],
                                                                            z_0_mu_logvar[:, 1, :])
            if config.model.forward_controls_loss:  # unused params might be modified by this criterion
                cont_loss = controls_criterion(v_out, v_in)
            else:
                cont_loss = controls_criterion(z_0_mu_logvar, v_in)

            # Log backpropagation losses
            scalars['ReconsLoss/Backprop/Train'].append(recons_loss)
            scalars['MSSpecLoss/Backprop/Train'].append(msspec_loss)
            scalars['ContrastiveLoss/Backprop/Train'].append(contrastive_loss)    
            scalars['Controls/BackpropLoss/Train'].append(cont_loss)

            # Update parameters
            utils.exception.check_nan_values(epoch, recons_loss, lat_loss, flow_input_loss, cont_loss, msspec_loss, cont_loss)
            (recons_loss + lat_loss + flow_input_loss + msspec_loss + contrastive_loss + cont_loss).backward()
            optimizer.step()

        if config.model.stochastic_latent:
            scalars['VAELoss/Train'] = SimpleMetric(scalars['ReconsLoss/Backprop/Train'].get()
                                                    + scalars['LatLoss/Train'].get())

        # = = = = = Evaluation on validation dataset (no profiling) = = = = =
        with torch.no_grad():
            ae_model_parallel.eval()  # BN stops running estimates
            v_error = torch.Tensor().to(device=device)  # Params inference error (Tensorboard plot)
            for i, sample in tqdm(enumerate(dataloader['validation']), desc='validation batch', position=1, total=len(dataloader['validation']), leave=False):
                if config.model.input_type == 'waveform':
                    x_in, v_in, sample_info = sample[0].to(device), sample[2].to(device), sample[3].to(device)
                else:
                    x_in, v_in, sample_info = sample[1].to(device), sample[2].to(device), sample[3].to(device)
                ae_out = ae_model_parallel(x_in, sample_info)  # Spectral VAE - tuple output

                if config.model.contrastive:
                    aug_specs = dataset.get_aug_specs(sample_info[:, 0]).to(device)
                    x_in = torch.cat((x_in, aug_specs), dim=0)

                if config.model.stochastic_latent:
                    z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac, x_out = ae_out
                    scalars['LatCorr/Valid'].append(z_0_mu_logvar, z_0_sampled)
                else:
                    z_K_sampled, x_out = ae_out

                if config.model.contrastive:
                    logits, labels = model.loss.info_nce_loss(z_K_sampled, config.train.minibatch_size)
                    contrastive_loss = cross_entropy(logits, labels)
                    contrastive_loss = config.train.contrastive_coef * contrastive_loss
                    z_K_sampled = z_K_sampled[:config.train.minibatch_size]
                else:
                    contrastive_loss = torch.tensor([0], device=device)

                v_out = reg_model_parallel(z_K_sampled)
                  
                if config.model.decoder_architecture is None:
                    recons_loss = torch.tensor([0], device=device)
                else:
                    recons_loss = reconstruction_criterion(x_out, x_in)               

                if config.model.input_type == 'waveform' and config.model.encoder_architecture != 'encodec_pretrained':
                    msspecs = msspec_transformer(x_in, x_out)
                    msspec_loss = 0
                    for j, specs in enumerate(msspecs):
                        s_x_1, s_y_1, s_x_2, s_y_2 = specs[:, 0], specs[:, 1], specs[:, 2], specs[:, 3]
                        msspec_loss += F.l1_loss(s_x_1, s_y_1) + msspec_alphas[j] * F.mse_loss(s_x_2, s_y_2)
                    msspec_loss = msspec_loss / (2 * len(msspecs))    
                else:
                    msspec_loss = torch.tensor([0], device=device)

                if config.model.stochastic_latent:
                    lat_loss = extended_ae_model.latent_loss(z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac)
                    scalars['LatLoss/Valid'].append(lat_loss)
                else:
                    lat_loss = torch.tensor([0], device=device)       

                if config.model.forward_controls_loss:  # unused params might be modified by this criterion
                    cont_loss = controls_criterion(v_out, v_in)
                else:
                    cont_loss = controls_criterion(z_0_mu_logvar, v_in)       
                
                # Monitoring losses
                scalars['ReconsLoss/MSE/Valid'].append(recons_loss if config.train.normalize_losses
                                                       or config.model.decoder_architecture is None
                                                       else F.mse_loss(x_out, x_in, reduction='mean'))
                scalars['Controls/QLoss/Valid'].append(controls_num_eval_criterion(v_out, v_in))
                scalars['Controls/Accuracy/Valid'].append(controls_accuracy_criterion(v_out, v_in))

                # Log backpropagation valid losses
                scalars['ReconsLoss/Backprop/Valid'].append(recons_loss)
                scalars['MSSpecLoss/Backprop/Valid'].append(msspec_loss)
                scalars['ContrastiveLoss/Backprop/Valid'].append(contrastive_loss)
                scalars['Controls/BackpropLoss/Valid'].append(cont_loss)

                # Validation plots
                if should_plot and config.model.decoder_architecture is not None and config.model.input_type == 'spectrogram':
                    v_error = torch.cat([v_error, v_out - v_in])  # Full-batch error storage
                    if i == 0:  # tensorboard samples for minibatch 'eval' [0] only
                        fig, _ = utils.figures.plot_train_spectrograms(x_in, x_out, sample_info, dataset,
                                                                       config.model, config.train)
                    logger.tensorboard.add_figure('Spectrogram', fig, epoch, close=True)

        if config.model.stochastic_latent:
            scalars['VAELoss/Valid'] = SimpleMetric(scalars['ReconsLoss/Backprop/Valid'].get()
                                                    + scalars['LatLoss/Valid'].get())
        # Dynamic LR scheduling depends on validation performance
        # Summed losses for plateau-detection are chosen in config.py
        scheduler.step(sum([scalars['{}/Valid'.format(loss_name)].get() for loss_name in config.train.scheduler_loss]))
        scalars['Sched/LR'] = logs.metrics.SimpleMetric(optimizer.param_groups[0]['lr'])
        # TODO replace early_stop by train_regression_only
        early_stop = (optimizer.param_groups[0]['lr'] < config.train.early_stop_lr_threshold)  # Early stop?
        # TODO regression_train_plateau should be the new early-stop

        # = = = = = Epoch logs (scalars/sounds/images + updated metrics) = = = = =
        for k, s in scalars.items():  # All available scalars are written to tensorboard
            logger.tensorboard.add_scalar(k, s.get(), epoch)
        if should_plot or early_stop:
            if v_error.size(0) > 0:  # u_error might be empty on early_stop
                fig, _ = utils.figures.plot_synth_preset_error(v_error.detach().cpu(),
                                                               dataset.preset_indexes_helper)
                logger.tensorboard.add_figure('SynthControlsError', fig, epoch)
            if config.model.stochastic_latent:
                fig, _ = utils.figures.plot_latent_distributions_stats(latent_metric=scalars['LatCorr/Valid'])
                logger.tensorboard.add_figure('LatentMu', fig, epoch)
                fig, _ = utils.figures.plot_spearman_correlation(latent_metric=scalars['LatCorr/Valid'])
                logger.tensorboard.add_figure('LatentEntanglement', fig, epoch)
        metrics['epochs'] = epoch + 1
        metrics['ReconsLoss/MSE/Valid_'].append(scalars['ReconsLoss/MSE/Valid'].get())
        metrics['Controls/QLoss/Valid_'].append(scalars['Controls/QLoss/Valid'].get())
        metrics['Controls/Accuracy/Valid_'].append(scalars['Controls/Accuracy/Valid'].get())

        if config.model.stochastic_latent:
            metrics['LatLoss/Valid_'].append(scalars['LatLoss/Valid'].get())
            metrics['LatCorr/Valid_'].append(scalars['LatCorr/Valid'].get())
        logger.tensorboard.update_metrics(metrics)

        # = = = = = Model+optimizer(+scheduler) save - ready for next epoch = = = = =
        if (epoch > 0 and epoch % config.train.save_period == 0)\
                or (epoch == config.train.n_epochs-1) or early_stop:
            logger.save_checkpoint(epoch, extended_ae_model, optimizer, scheduler)
        logger.on_epoch_finished(epoch)
        if early_stop:
            print("[train.py] Training stopped early (final loss plateau)")
            break

    # ========== Logger final stats ==========
    logger.on_training_finished()


    # ========== "Manual GC" (to try to prevent random CUDA out-of-memory between enqueued runs ==========
    del scheduler, optimizer
    del reg_model_parallel, ae_model_parallel
    del extended_ae_model
    del controls_criterion, controls_num_eval_criterion, controls_accuracy_criterion, reconstruction_criterion
    del logger
    del dataloader, dataset


if __name__ == "__main__":
    # Normal run, config.py only will be used to parametrize learning and models
    train_config()

