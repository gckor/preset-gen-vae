"""
Utility functions for building a new model (using only config from config.py),
or for building a previously trained model before loading state dicts.

Decomposed into numerous small function for easier module-by-module debugging.
"""

from model import VAE, encoder, decoder, extendedAE, regression, seanet, encodec, lstm
from model.quantization import vq
from model.audiomae import models_mae, misc
from model.encodec.model import EncodecModel
import numpy as np
import torch
import torch.nn as nn

import timm.optim.optim_factory as optim_factory


def build_encoder_and_decoder_models(model_config, train_config):
    # Backward compatibility - recently added config args
    if not hasattr(model_config, 'stack_specs_deepest_features_mix'):
        model_config.stack_specs_deepest_features_mix = True  # Default: multi-spec features mixed by 1x1 conv layer
    # Multi-MIDI notes but single-ch spectrogram model must be bigger (to perform a fairer comparison w/ multi-ch)
    force_bigger_network = ((len(model_config.midi_notes) > 1) and not model_config.stack_spectrograms)
    # Encoder and decoder with the same architecture
    enc_z_length = (model_config.dim_z - 2 if model_config.concat_midi_to_z else model_config.dim_z)

    print("FORCE BIGGER ENC/DEC NETWORKS = {}".format(force_bigger_network))  # TODO remove
    print("MIDI NOTES = {}".format(model_config.midi_notes))  # TODO remove
    if model_config.encoder_architecture == 'seanet':
        encoder_model = seanet.SEANetEncoder(dimension=model_config.dim_z, n_filters=8, ratios=[8, 4, 4, 2])
    elif model_config.encoder_architecture == 'encodec_pretrained':
        encodec_pretrained = EncodecModel.encodec_model_24khz()
        encoder_model = encodec_pretrained.encoder
    elif model_config.encoder_architecture == 'audiomae_pretrained':
        encoder_model = models_mae.__dict__['mae_vit_base_patch16'](
            img_size=(336, 256),
            in_chans=1,
            audio_exp=True,
            decoder_mode=1,
        )
        checkpoint = model_config.pretrained_dir
        misc.load_model(checkpoint, encoder_model)
    else:
        encoder_model = \
            encoder.SpectrogramEncoder(model_config.encoder_architecture, enc_z_length,
                                    model_config.input_tensor_size, train_config.fc_dropout,
                                    output_bn=(train_config.latent_flow_input_regularization.lower() == 'bn'),
                                    deepest_features_mix=model_config.stack_specs_deepest_features_mix,
                                    force_bigger_network=force_bigger_network,
                                    stochastic_latent=model_config.stochastic_latent)
        
    if model_config.decoder_architecture == 'seanet':
        decoder_model = seanet.SEANetDecoder(dimension=model_config.dim_z, n_filters=16, ratios=[8, 4, 4, 2])
    elif model_config.decoder_architecture == 'encodec_pretrained':
        decoder_model = encodec_pretrained.decoder
    elif model_config.decoder_architecture is None:
        print("Decoder is not included in the architecture")
        decoder_model = None
    else:
        decoder_model = decoder.SpectrogramDecoder(model_config.decoder_architecture, model_config.dim_z,
                                               model_config.input_tensor_size, train_config.fc_dropout,
                                               force_bigger_network=force_bigger_network)
    return encoder_model, decoder_model


def build_ae_model(model_config, train_config):
    """
    Builds an auto-encoder model given a configuration. Built model can be initialized later
    with a previous state_dict.

    :param model_config: model global attribute from the config.py module
    :param train_config: train attributes (a few are required, e.g. dropout probability)
    :return: Tuple: encoder, decoder, full AE model
    """
    if model_config.encoder_architecture == 'encodec_pretrained':
        encodec_pretrained = EncodecModel.encodec_model_24khz()
        encodec_pretrained.set_target_bandwidth(6)
        encoder = encodec_pretrained.encoder
        decoder = None
        quantizer = encodec_pretrained.quantizer
        emb_dim = encoder.dimension * (model_config.waveform_size // np.prod(encoder.ratios) + 1)
        frame_rate = model_config.sampling_rate // encoder.hop_length
        context_model = nn.Sequential(nn.Linear(emb_dim, emb_dim // 8), nn.ReLU(),
                                      nn.Linear(emb_dim // 8, 1024), nn.ReLU(),
                                      nn.Linear(1024, model_config.dim_z))
        ae_model = VAE.DeterministicAE(encoder, model_config.dim_z, decoder, quantizer, context_model,
                                       frame_rate=frame_rate, freeze=True)
        return encoder, decoder, ae_model

    encoder_model, decoder_model = build_encoder_and_decoder_models(model_config, train_config)
    # AE model
    if model_config.stochastic_latent:
        if model_config.latent_flow_arch is None:
            ae_model = VAE.BasicVAE(encoder_model, model_config.dim_z, decoder_model, train_config.normalize_losses,
                                    train_config.latent_loss, concat_midi_to_z=model_config.concat_midi_to_z)
        else:
            # TODO test latent flow dropout (in all but the last flow layers)
            ae_model = VAE.FlowVAE(encoder_model, model_config.dim_z, decoder_model, train_config.normalize_losses,
                                model_config.latent_flow_arch, concat_midi_to_z0=model_config.concat_midi_to_z)
    else:
        if model_config.latent_quantization == 'rvq':
            quantizer = vq.ResidualVectorQuantizer(dimension=model_config.dim_z)
            context_model = lstm.ContextModel(dimension=model_config.dim_z)
            frame_rate = model_config.sampling_rate // encoder_model.hop_length
            ae_model = encodec.EncodecModel(encoder_model, decoder_model, quantizer, context_model,
                                            frame_rate, sample_rate=model_config.sampling_rate, channels=1)
        elif model_config.input_type == 'spectrogram':
            context_model = None
            ae_model = VAE.DeterministicAE(encoder_model, model_config.dim_z, decoder_model,
                                           context_model=context_model, concat_midi_to_z=model_config.concat_midi_to_z)
        elif model_config.input_type == 'waveform':
            context_model = lstm.ContextModel(dimension=model_config.dim_z)
            ae_model = VAE.DeterministicAE(encoder_model, model_config.dim_z, decoder_model,
                                           context_model=context_model, concat_midi_to_z=model_config.concat_midi_to_z)
    return encoder_model, decoder_model, ae_model


def build_extended_ae_model(model_config, train_config, idx_helper):
    """ Builds a spectral auto-encoder model, and a synth parameters regression model which takes
    latent vectors as input. Both models are integrated into an ExtendedAE model. """
    # Spectral VAE
    encoder_model, decoder_model, ae_model = build_ae_model(model_config, train_config)
    # Config checks - for backward compatibility
    if not hasattr(model_config, 'params_reg_softmax'):
        model_config.params_reg_softmax = True  # Default value is True (legacy behavior)
    # Regression model - extension of the VAE model
    if model_config.params_regression_architecture.startswith("mlp_"):
        assert model_config.forward_controls_loss is True  # Non-invertible MLP cannot inverse target values
        reg_arch = model_config.params_regression_architecture.replace("mlp_", "")
        reg_model = regression.MLPRegression(reg_arch, model_config.dim_z, idx_helper, train_config.reg_fc_dropout,
                                             cat_softmax_activation=model_config.params_reg_softmax)
    elif model_config.params_regression_architecture.startswith("flow_"):
        assert model_config.learnable_params_tensor_length > 0  # Flow models require dim_z to be equal to this length
        reg_arch = model_config.params_regression_architecture.replace("flow_", "")
        reg_model = regression.FlowRegression(reg_arch, model_config.dim_z, idx_helper,
                                              fast_forward_flow=model_config.forward_controls_loss,
                                              dropout_p=train_config.reg_fc_dropout,
                                              cat_softmax_activation=model_config.params_reg_softmax)
    else:
        raise NotImplementedError("Synth param regression arch '{}' not implemented"
                                  .format(model_config.params_regression_architecture))
    extended_ae_model = extendedAE.ExtendedAE(ae_model, reg_model, idx_helper, train_config.fc_dropout)
    return encoder_model, decoder_model, ae_model, extended_ae_model


def _is_attr_equal(attr1, attr2):
    """ Compares two config attributes - lists auto converted to tuples. """
    _attr1 = tuple(attr1) if isinstance(attr1, list) else attr1
    _attr2 = tuple(attr2) if isinstance(attr2, list) else attr2
    return _attr1 == _attr2


def check_configs_on_resume_from_checkpoint(new_model_config, new_train_config, config_json_checkpoint):
    """
    Performs a full consistency check between the last checkpoint saved config (stored into a .json file)
    and the new required config as described in config.py

    :raises: ValueError if any incompatibility is found

    :param new_model_config: model Class of the config.py file
    :param new_train_config: train Class of the config.py file
    :param config_json_checkpoint: config.py attributes from previous run, loaded from the .json file
    :return:
    """
    # Model config check TODO add/update attributes to check
    prev_config = config_json_checkpoint['model']
    attributes_to_check = ['name', 'run_name', 'encoder_architecture',
                           'dim_z', 'concat_midi_to_z', 'latent_flow_arch',
                           'logs_root_dir',
                           'note_duration',
                           # 'midi_notes',  # FIXME json 2D list to tuple conversion required for comparison
                           'stack_spectrograms', 'increased_dataset_size',
                           'stft_args', 'spectrogram_size', 'mel_bins']
    for attr in attributes_to_check:
        if not _is_attr_equal(prev_config[attr], new_model_config.__dict__[attr]):
            raise ValueError("Model attribute '{}' is different in the new config.py ({}) and the old config.json ({})"
                             .format(attr, new_model_config.__dict__[attr], prev_config[attr]))
    # Train config check TODO add.update attributes to check
    prev_config = config_json_checkpoint['train']
    attributes_to_check = ['minibatch_size', 'test_holdout_proportion', 'normalize_losses',
                           'optimizer', 'scheduler_name']
    for attr in attributes_to_check:
        if not _is_attr_equal(prev_config[attr], new_train_config.__dict__[attr]):
            raise ValueError("Train attribute '{}' is different in the new config.py ({}) and the old config.json ({})"
                             .format(attr, new_train_config.__dict__[attr], prev_config[attr]))


# Model Build tests - see also params_regression.ipynb
if __name__ == "__main__":

    import sys
    import pathlib  # Dirty path trick to import config.py from project root dir
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config

    # Manual config changes (for test purposes only)
    config.model.synth_params_count = 144

    _encoder_model, _decoder_model, _ae_model, _extended_ae_model \
        = build_extended_ae_model(config.model, config.train)
