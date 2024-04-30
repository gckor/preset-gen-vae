"""
Allows easy modification of all configuration parameters required to define,
train or evaluate a model.
This script is not intended to be run, it only describes parameters.
However, some dynamic hyper-parameters are properly set when this module is imported.

This configuration is used when running train.py as main.
When running train_queue.py, configuration changes are relative to this config.py file.

When a run starts, this file is stored as a config.json file. To ensure easy restoration of
parameters, please only use simple types such as string, ints, floats, tuples (no lists) and dicts.
"""


import datetime
from utils.config import _Config  # Empty class - to ease JSON serialization of this file


model = _Config()
model.name = 'CNNMLP_re'
model.run_name = 'kfold1'  # run: different hyperparams, optimizer, etc... for a given model
model.allow_erase_run = True  # If True, a previous run with identical name will be erased before training
# See model/encoder.py to view available architectures. Decoder architecture will be as symmetric as possible.
model.encoder_architecture = 'speccnn8l1_bn' # 'speccnn8l1_bn', 'seanet'
# Default: Same with the encoder architecture. Set to None to disable the decoder and reconstruction loss.
model.decoder_architecture = 'speccnn8l1_bn' # 'speccnn8l1_bn', 'seanet', None
model.latent_quantization = None # 'rvq', None
model.input_type = 'spectrogram' # 'waveform', 'spectrogram'
model.stochastic_latent = True # True (VAE), False (deterministic AE)
# Possible values: 'flow_realnvp_6l300', 'flow_realnvp_4l180', 'mlp_4l1024', ... (configurable numbers of layers and neurons)
model.params_regression_architecture = 'mlp_4l1024'
model.params_reg_softmax = False  # Apply softmax in the flow itself? If False: cat loss can be BCE or CCE
# Spectrogram size cannot easily be modified - all CNN decoders should be re-written
model.note_duration = (3.0, 1.0)
model.sampling_rate = 22050 # 22050, 24000
model.stft_args = (1024, 256)  # fft size and hop size
model.mel_bins = 257  # -1 disables Mel-scale spectrogram. Try: 257, 513, ...
model.mel_f_limits = (0, 11050)  # min/max Mel-spectrogram frequencies TODO implement
# Tuple of (pitch, velocity) tuples. Using only 1 midi note is fine.
model.midi_notes = ((60, 85), )  # Reference note
# model.midi_notes = ((40, 85), (50, 85), (60, 42), (60, 85), (60, 127), (70, 85))
model.stack_spectrograms = False  # If True, dataset will feed multi-channel spectrograms to the encoder
model.stack_specs_deepest_features_mix = False  # if True, feats mixed in the deepest 1x1 conv, else in the deepest 4x4
# If True, each preset is presented several times per epoch (nb of train epochs must be reduced) such that the
# dataset size is artificially increased (6x bigger with 6 MIDI notes) -> warmup and patience epochs must be scaled
model.increased_dataset_size = None  # See update_dynamic_config_params()
model.spectrogram_min_dB = -120.0
# Possible spectrogram sizes:
#   (513, 433): audio 5.0s, fft size 1024, fft hop 256
#   (257, 347): audio 4.0s, fft size 512 (or fft 1024 w/ mel_bins 257), fft hop 256
model.spectrogram_size = (257, 347)  # see data/dataset.py to retrieve this from audio/stft params
model.waveform_size = 88576 # 88576 (22050 Hz), 96256 (24000 Hz)
model.input_tensor_size = None  # see update_dynamic_config_params()
# If True, encoder output is reduced by 2 for 1 MIDI pitch and 1 velocity to be concatenated to the latent vector
model.concat_midi_to_z = None  # See update_dynamic_config_params()
# Latent space dimension  *************** When using a Flow regressor, this dim is automatically set ******************
model.dim_z = 610 # 610  # Including possibly concatenated midi pitch and velocity
# Latent flow architecture, e.g. 'realnvp_6l300', 'realnvp_4l200' (4 flows, 200 hidden features per flow)
#    - base architectures can be realnvp, maf, ...
#    - set to None to disable latent space flow transforms
model.latent_flow_arch = None # 'realnvp_6l300', None
# If True, loss compares v_out and v_in. If False, we will flow-invert v_in to get loss in the q_Z0 domain.
# This option has implications on the regression model itself (the flow will be used in direct or inverse order)
model.forward_controls_loss = True  # Must be true for non-invertible MLP regression

model.synth = 'dexed'
# Dexed-specific auto rename: '*' in 'al*_op*_lab*' will be replaced by the actual algorithms, operators and labels
model.synth_args_str = 'al*_op*_lab*'  # Auto-generated string (see end of script)
model.synth_params_count = -1  # Will be set automatically - see data.build.get_full_and_split_datasets
model.learnable_params_tensor_length = -1  # Will be set automatically - see data.build.get_full_and_split_datasets
# Modeling of synth controls probability distributions
# Possible values: None, 'vst_cat' or 'all<=xx' where xx is numerical params threshold cardinal
model.synth_vst_params_learned_as_categorical = 'all<=32'
# flags/values to describe the dataset to be used
model.dataset_labels = None  # tuple of labels (e.g. ('harmonic', 'percussive')), or None to use all available labels
# Dexed: Preset Algorithms, and activated Operators (Lists of ints, None to use all)
# Limited algorithms (non-symmetrical only): [1, 2, 7, 8, 9, 14, 28, 3, 4, 11, 16, 18]
# Other synth: ...?
model.dataset_synth_args = (None, [1, 2, 3, 4, 5, 6])
# Directory for saving metrics, samples, models, etc... see README.md
model.logs_root_dir = "/exp_logs/preset-gen-vae"  # Path from this directory
model.dataset_dir = "/dataset/preset-gen-vae/numcatpp"


train = _Config()
train.start_datetime = datetime.datetime.now().isoformat()
train.minibatch_size = 160 # 160
train.main_cuda_device_idx = 1  # CUDA device for nonparallel operations (losses, ...)
train.test_holdout_proportion = 0.2
train.k_folds = 5
train.current_k_fold = 1 # 0, 1, 2, 3, 4
train.start_epoch = 0  # 0 means a restart (previous data erased). If > 0: will load start_epoch-1 checkpoint
# Total number of epochs (including previous training epochs)
train.n_epochs = 400  # See update_dynamic_config_params().  16k sample dataset: set to 700
train.save_period = 20  # Period for checkpoint saves (large disk size). Tensorboard scalars/metric logs at all epochs.
train.plot_period = 20  # Period (in epochs) for plotting graphs into Tensorboard (quite CPU and SSD expensive)
train.latent_loss = 'Dkl'  # Latent regularization loss: Dkl or MMD for Basic VAE (Flow VAE has its own specific loss)
# When using a latent flow z0-->zK, z0 is not regularized. To keep values around 0.0, batch-norm or a 0.1Dkl can be used
train.latent_flow_input_regularization = 'bn'  # 'bn' (on encoder output) or 'dkl' (on q_Z0 gaussian flow input)
train.params_cat_bceloss = False  # If True, disables the Categorical Cross-Entropy loss to compute BCE loss instead
train.params_cat_softmax_temperature = 0.2  # Temperature if softmax if applied in the loss only
# Losses normalization allow to get losses in the same order of magnitude, but does not optimize the true ELBO.
# When un-normalized, the reconstruction loss (log-probability of a multivariate gaussian) is orders of magnitude
# bigger than other losses. Train does not work with normalize=False at the moment - use train.beta to compensate
train.normalize_losses = True  # Normalize all losses over the vector-dimension (e.g. spectrogram pixels count, D, ...)


# TODO train regression network alone when full-train has finished?
train.optimizer = 'Adam'
# Maximal learning rate (reached after warmup, then reduced on plateaus)
# LR decreased if non-normalized losses (which are expected to be 90,000 times bigger with a 257x347 spectrogram)
train.initial_learning_rate = 2e-4  # e-9 LR with e+4 loss does not allow any train (vanishing grad?)
# Learning rate warmup (see https://arxiv.org/abs/1706.02677)
train.lr_warmup_epochs = 6  # See update_dynamic_config_params(). 16k samples dataset: set to 10
train.lr_warmup_start_factor = 0.1
train.adam_betas = (0.9, 0.999)  # default (0.9, 0.999)
train.weight_decay = 1e-4  # Dynamic weight decay?
train.fc_dropout = 0.3
train.reg_fc_dropout = 0.4
# (beta<1, normalize=True) corresponds to (beta>>1, normalize=False) in the beta-VAE formulation (ICLR 2017)
train.beta = 0.2  # latent loss factor - use much lower value (e-2) to get closer the ELBO
train.beta_start_value = 0.1  # Should not be zero (risk of a very unstable training)
# Epochs of warmup increase from start_value to beta
train.beta_warmup_epochs = 25  # See update_dynamic_config_params(). 16k samples dataset: set to 40
train.beta_cycle_epochs = -1  # beta cyclic annealing (https://arxiv.org/abs/1903.10145). -1 deactivates TODO do

train.scheduler_name = 'ReduceLROnPlateau'  # TODO try CosineAnnealing
# Possible values: 'VAELoss' (total), 'ReconsLoss', 'Controls/BackpropLoss', ... All required losses will be summed
train.scheduler_loss = ('ReconsLoss/Backprop', 'Controls/BackpropLoss')
train.scheduler_lr_factor = 0.2
# Set a longer patience with smaller datasets and quite unstable trains
train.scheduler_patience = 6  # See update_dynamic_config_params(). 16k samples dataset:  set to 10
train.scheduler_cooldown = 6  # See update_dynamic_config_params(). 16k samples dataset: set to 10
train.scheduler_threshold = 1e-4
# Training considered "dead" when dynamic LR reaches this value
train.early_stop_lr_threshold = None  # See update_dynamic_config_params()

train.verbosity = 3  # 0: no console output --> 3: fully-detailed per-batch console output
train.init_security_pause = 0.0  # Short pause before erasing an existing run
# Number of logged audio and spectrograms for a given epoch
train.logged_samples_count = 4  # See update_dynamic_config_params()
train.profiler_args = {'enabled': False, 'use_cuda': True, 'record_shapes': False,
                       'profile_memory': False, 'with_stack': False}
train.profiler_full_trace = False  # If True, runs only a few batches then exits - but saves a fully detailed trace.json
train.profiler_1_GPU = False  # Profiling on only 1 GPU allow a much better understanding of trace.json


evaluate = _Config()
evaluate.epoch = -1  # Trained model to be loaded for post-training evaluation.


# ---------------------------------------------------------------------------------------


def update_dynamic_config_params():
    """
    Updates dynamic some global attributes of this config.py module.
    This function should be called after any modification of this module's attributes.
    """

    # stack_spectrograms must be False for 1-note datasets - security check
    model.stack_spectrograms = model.stack_spectrograms and (len(model.midi_notes) > 1)
    # Artificially increased data size?
    model.increased_dataset_size = (len(model.midi_notes) > 1) and not model.stack_spectrograms
    model.concat_midi_to_z = (len(model.midi_notes) > 1) and not model.stack_spectrograms
    # Mini-batch size can be smaller for the last mini-batches and/or during evaluation
    if model.input_type == 'spectrogram':
        model.input_tensor_size = (train.minibatch_size, 1 if not model.stack_spectrograms else len(model.midi_notes),
                               model.spectrogram_size[0], model.spectrogram_size[1])
    else:
        model.input_tensor_size = (train.minibatch_size, 1, model.waveform_size)

    # Dynamic train hyper-params
    train.early_stop_lr_threshold = 0 # train.initial_learning_rate * 1e-3
    train.logged_samples_count = max(train.logged_samples_count, len(model.midi_notes))
    # Train hyper-params (epochs counts) that should be increased when using a subset of the dataset
    if model.dataset_synth_args[0] is not None:  # Limited Dexed algorithms?  TODO handle non-dexed synth
        train.n_epochs = 700
        train.lr_warmup_epochs = 10
        train.scheduler_patience = 10
        train.scheduler_cooldown = 10
        train.beta_warmup_epochs = 40
    # Train hyper-params (epochs counts) that should be reduced with artificially increased datasets
    # Augmented  datasets introduce 6x more backprops <=> 6x more epochs. Patience and cooldown must however remain >= 2
    if model.increased_dataset_size:  # Stacked spectrogram do not increase the dataset size (number of items)
        N = len(model.midi_notes) - 1  # reduce a bit less that dataset's size increase
        train.n_epochs = 1 + train.n_epochs // N
        train.lr_warmup_epochs = 1 + train.lr_warmup_epochs // N
        train.scheduler_patience = 1 + train.scheduler_patience // N
        train.scheduler_cooldown = 1 + train.scheduler_cooldown // N
        train.beta_warmup_epochs = 1 + train.beta_warmup_epochs // N

    # Automatic model.synth string update - to summarize this info into 1 Tensorboard string hparam
    if model.synth == "dexed":
        if model.dataset_synth_args[0] is not None:  # Algorithms
            model.synth_args_str = model.synth_args_str.replace("al*", "al" +
                                                                '.'.join(
                                                                    [str(alg) for alg in model.dataset_synth_args[0]]))
        if model.dataset_synth_args[1] is not None:  # Operators
            model.synth_args_str = model.synth_args_str.replace("_op*", "_op" +
                                                                ''.join(
                                                                    [str(op) for op in model.dataset_synth_args[1]]))
        if model.dataset_labels is not None:  # Labels
            model.synth_args_str = model.synth_args_str.replace("_lab*", '_' +
                                                                '_'.join(
                                                                    [label[0:4] for label in model.dataset_labels]))
    else:
        raise NotImplementedError("Unknown synth prefix for model.synth '{}'".format(model.synth))


# Call this function again after modifications from the outside of this module
update_dynamic_config_params()

