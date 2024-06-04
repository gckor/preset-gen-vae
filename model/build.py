"""
Utility functions for building a new model (using only config from config.py),
or for building a previously trained model before loading state dicts.

Decomposed into numerous small function for easier module-by-module debugging.
"""

from model import VAE, encoder, decoder, extendedAE, regression, seanet, encodec, lstm
from model.quantization import vq
from model.audiomae import misc
from model.audiomae.models_mae import MaskedAutoencoderViT
from model.encodec.model import EncodecModel
import numpy as np
import torch.nn as nn


def build_encoder_and_decoder_models(config):
    # Encoder and decoder with the same architecture
    if config.model.stack_spectrograms:
        spectrogram_channels = len(config.dataset.midi_notes)
    else:
        spectrogram_channels = 1

    config.model.input_tensor_size = (
        config.train.minibatch_size,
        spectrogram_channels,
        config.model.spectrogram_size[0],
        config.model.spectrogram_size[1]
    )

    if config.model.concat_midi_to_z:
        enc_z_length = config.model.dim_z - 2
    else:
        enc_z_length = config.model.dim_z

    if config.model.encoder_architecture == 'seanet':
        encoder_model = seanet.SEANetEncoder(dimension=config.model.dim_z, n_filters=8, ratios=[8, 4, 4, 2])
    elif config.model.encoder_architecture == 'encodec_pretrained':
        encodec_pretrained = EncodecModel.encodec_model_24khz()
        encoder_model = encodec_pretrained.encoder
    elif config.model.encoder_architecture.startswith('audiomae'):
        encoder_model = MaskedAutoencoderViT(
            img_size=config.model.spectrogram_size,
            stochastic_latent=config.model.stochastic_latent,
        )
        if 'pretrained' in config.model.encoder_architecture:
            checkpoint = config.model.pretrained_dir
            misc.load_model(checkpoint, encoder_model)
    else:
        encoder_model = encoder.SpectrogramEncoder(
            config.model.encoder_architecture,
            enc_z_length,
            config.model.input_tensor_size,
            config.train.fc_dropout,
            output_bn=(config.train.latent_flow_input_regularization.lower() == 'bn'),
            deepest_features_mix=config.model.stack_specs_deepest_features_mix,
            stochastic_latent=config.model.stochastic_latent
        )
        
    if config.model.decoder_architecture == 'seanet':
        decoder_model = seanet.SEANetDecoder(dimension=config.model.dim_z, n_filters=16, ratios=[8, 4, 4, 2])
    elif config.model.decoder_architecture == 'encodec_pretrained':
        decoder_model = encodec_pretrained.decoder
    elif config.model.decoder_architecture is None:
        print("Decoder is not included in the architecture")
        decoder_model = None
    else:
        decoder_model = decoder.SpectrogramDecoder(
            config.model.decoder_architecture,
            config.model.dim_z,
            config.model.input_tensor_size, 
            config.train.fc_dropout,
        )

    return encoder_model, decoder_model


def build_ae_model(config):
    """
    Builds an auto-encoder model given a configuration. Built model can be initialized later
    with a previous state_dict.

    :param model_config: model global attribute from the config.py module
    :param train_config: train attributes (a few are required, e.g. dropout probability)
    :return: Tuple: encoder, decoder, full AE model
    """
    if config.model.encoder_architecture == 'encodec_pretrained':
        encodec_pretrained = EncodecModel.encodec_model_24khz()
        encodec_pretrained.set_target_bandwidth(6)
        encoder = encodec_pretrained.encoder
        decoder = None
        quantizer = encodec_pretrained.quantizer
        emb_dim = encoder.dimension * (config.model.waveform_size // np.prod(encoder.ratios) + 1)
        frame_rate = config.model.sampling_rate // encoder.hop_length
        context_model = nn.Sequential(nn.Linear(emb_dim, emb_dim // 8), nn.ReLU(),
                                      nn.Linear(emb_dim // 8, 1024), nn.ReLU(),
                                      nn.Linear(1024, config.model.dim_z))
        ae_model = VAE.DeterministicAE(encoder, config.model.dim_z, decoder, quantizer, context_model,
                                       frame_rate=frame_rate, freeze=True)
        return encoder, decoder, ae_model

    encoder_model, decoder_model = build_encoder_and_decoder_models(config)
    # AE model
    if config.model.stochastic_latent:
        if config.model.latent_flow_arch is None:
            ae_model = VAE.BasicVAE(
                encoder_model,
                config.model.dim_z,
                decoder_model,
                config.train.normalize_losses,
                config.train.latent_loss,
                concat_midi_to_z=config.model.concat_midi_to_z
            )
        else:
            ae_model = VAE.FlowVAE(
                encoder_model,
                config.model.dim_z,
                decoder_model,
                config.train.normalize_losses,
                config.model.latent_flow_arch,
                concat_midi_to_z0=config.model.concat_midi_to_z
            )
    else:
        if config.model.latent_quantization == 'rvq':
            quantizer = vq.ResidualVectorQuantizer(dimension=config.model.dim_z)
            context_model = lstm.ContextModel(dimension=config.model.dim_z)
            frame_rate = config.model.sampling_rate // encoder_model.hop_length
            ae_model = encodec.EncodecModel(
                encoder_model,
                decoder_model,
                quantizer,
                context_model,
                frame_rate,
                sample_rate=config.model.sampling_rate, 
                channels=1
            )
        elif config.model.input_type == 'spectrogram':
            context_model = None
            ae_model = VAE.DeterministicAE(
                encoder_model,
                config.model.dim_z,
                decoder_model,
                context_model=context_model,
                concat_midi_to_z=config.model.concat_midi_to_z
            )
        elif config.model.input_type == 'waveform':
            context_model = lstm.ContextModel(dimension=config.model.dim_z)
            ae_model = VAE.DeterministicAE(
                encoder_model,
                config.model.dim_z,
                decoder_model,
                context_model=context_model,
                concat_midi_to_z=config.model.concat_midi_to_z
            )
    return encoder_model, decoder_model, ae_model


def build_extended_ae_model(config, idx_helper):
    """
    Builds a spectral auto-encoder model, and a synth parameters regression model which takes
    latent vectors as input. Both models are integrated into an ExtendedAE model.
    """
    # Spectral VAE
    encoder_model, decoder_model, ae_model = build_ae_model(config)

    # Regression model - extension of the VAE model
    if config.model.params_regression_architecture.startswith("mlp_"):
        assert config.model.forward_controls_loss is True  # Non-invertible MLP cannot inverse target values
        reg_arch = config.model.params_regression_architecture.replace("mlp_", "")
        reg_model = regression.MLPRegression(
            reg_arch,
            config.model.dim_z,
            idx_helper,
            config.train.reg_fc_dropout,
            cat_softmax_activation=config.model.params_reg_softmax
        )
    elif config.model.params_regression_architecture.startswith("flow_"):
        assert config.learnable_params_tensor_length > 0  # Flow models require dim_z to be equal to this length
        reg_arch = config.model.params_regression_architecture.replace("flow_", "")
        reg_model = regression.FlowRegression(
            reg_arch,
            config.model.dim_z,
            idx_helper,
            fast_forward_flow=config.model.forward_controls_loss,
            dropout_p=config.train.reg_fc_dropout,
            cat_softmax_activation=config.model.params_reg_softmax
        )
    else:
        raise NotImplementedError("Synth param regression arch '{}' not implemented"
                                  .format(config.model.params_regression_architecture))
    
    extended_ae_model = extendedAE.ExtendedAE(ae_model, reg_model, idx_helper, config.train.fc_dropout)
    return extended_ae_model


def _is_attr_equal(attr1, attr2):
    """ Compares two config attributes - lists auto converted to tuples. """
    _attr1 = tuple(attr1) if isinstance(attr1, list) else attr1
    _attr2 = tuple(attr2) if isinstance(attr2, list) else attr2
    return _attr1 == _attr2


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
