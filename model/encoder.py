import sys
import torch
import torch.nn as nn
from typing import Dict, Any, List

from data.preset import PresetIndexesHelper
from model import layer
from model.layer import TransformerEncoder, TransformerDecoder, CSWinTransformer


class SpectrogramCNN(nn.Module):
    """
    Contains a spectrogram-input CNN and some MLP layers,
    and outputs the mu and logs(var) values

    :param architecture:
    :param dim_z:
    :param spectrogram_channels: Input channel
    :param spectrogram_size: 2D input shape
    :param fc_dropout:
    :param output_bn:
    :param deepest_features_mix: (applies to multi-channel spectrograms only) If True, features mixing will be
        done on the 1x1 deepest conv layer. If False, mixing will be done before the deepest conv layer (see
        details in implementation)
    :param force_bigger_network: Optional, to impose a higher number of channels for the last 4x4 (should be
        used for fair comparisons between single/multi-specs encoder)
    """

    def __init__(
        self,
        dim_z,
        spectrogram_channels,
        spectrogram_size,
        fc_dropout,
        output_bn=False,
        deepest_features_mix=True,
        stochastic_latent=True
    ):
        super().__init__()
        self.dim_z = dim_z  # Latent-vector size (2*dim_z encoded values - mu and logs sigma 2)
        self.deepest_features_mix = deepest_features_mix
        self.stochastic_latent = stochastic_latent
        # 2048 if single-ch, 1024 if multi-channel 4x4 mixer (to compensate for the large number of added params)
        self.spectrogram_channels = spectrogram_channels
        self.mixer_1x1conv_ch = 2048 # if (self.spectrogram_channels == 1) else 1024
        self.fc_dropout = fc_dropout

        # 1) Main CNN encoder (applied once per input spectrogram channel) - - - - -
        # stacked spectrograms: don't add the final 1x1 conv layer, or the 2 last conv layers (1x1 and 4x4)
        self.single_ch_cnn = SingleChannelCNN(last_layers_to_remove=(1 if self.deepest_features_mix else 2))
        
        # 2) Features mixer
        self.features_mixer_cnn = nn.Sequential()
        if self.deepest_features_mix:
            self.features_mixer_cnn = layer.Conv2D(
                512 * self.spectrogram_channels,
                self.mixer_1x1conv_ch,
                [1, 1], [1, 1], 0, [1, 1],
                activation=nn.LeakyReLU(0.1),
                name_prefix='enc8',
                batch_norm=None
            )
        else:  # mixing conv layer: deepest-1 (4x4 kernel)
            n_4x4_ch = 512 # if self.spectrogram_channels == 1 else 768
            self.features_mixer_cnn = nn.Sequential(
                layer.Conv2D(
                    256 * self.spectrogram_channels,
                    n_4x4_ch,
                    [4, 4], [2, 2], 2, [1, 1],
                    activation=nn.LeakyReLU(0.1),
                    name_prefix='enc7'
                ),
                layer.Conv2D(
                    n_4x4_ch,
                    self.mixer_1x1conv_ch,
                    [1, 1], [1, 1], 0, [1, 1],
                    activation=nn.LeakyReLU(0.1),
                    name_prefix='enc8',
                    batch_norm=None
                )
            )

        # 3) MLP for extracting properly-sized latent vector
        # Automatic CNN output tensor size inference
        with torch.no_grad():
            dummy_shape = [1] + [spectrogram_channels] + spectrogram_size
            dummy_spectrogram = torch.zeros(tuple(dummy_shape))
            self.cnn_out_size = self._forward_cnns(dummy_spectrogram).size()
        cnn_out_items = self.cnn_out_size[1] * self.cnn_out_size[2] * self.cnn_out_size[3]
        mlp_out_dim = self.dim_z * 2 if self.stochastic_latent else self.dim_z
        
        # No activation - outputs are latent mu/logvar
        self.mlp = nn.Sequential(
            nn.Dropout(self.fc_dropout),
            nn.Linear(cnn_out_items, mlp_out_dim)
        )

        if output_bn:
            self.mlp.add_module('lat_in_regularization', nn.BatchNorm1d(mlp_out_dim))

    def _forward_cnns(self, x_spectrograms):
        # apply main cnn multiple times
        single_channel_cnn_out = [self.single_ch_cnn(torch.unsqueeze(x_spectrograms[:, ch, :, :], dim=1))
                                  for ch in range(self.spectrogram_channels)]
        # Then mix features from different input channels - and flatten the result
        return self.features_mixer_cnn(torch.cat(single_channel_cnn_out, dim=1))

    def forward(self, x_spectrograms):
        n_minibatch = x_spectrograms.size()[0]
        cnn_out = self._forward_cnns(x_spectrograms).view(n_minibatch, -1)  # 2nd dim automatically inferred
        # print("Forward CNN out size = {}".format(cnn_out.size()))
        out = self.mlp(cnn_out)
        # Last dim contains a latent proba distribution value, last-1 dim is 2 (to retrieve mu or logs sigma2)
        if self.stochastic_latent:
            out = torch.reshape(out, (n_minibatch, 2, self.dim_z))
        return out


class SingleChannelCNN(nn.Module):
    """ A encoder CNN network for spectrogram input """

    def __init__(self, last_layers_to_remove=0):
        """
        Automatically defines an autoencoder given the specified architecture

        :param last_layers_to_remove: Number of deepest conv layers to omit in this module (they will be added in
            the owner of this pure-CNN module).
        """
        super().__init__()
        act = nn.LeakyReLU
        act_p = 0.1  # Activation param
        self.enc_nn = nn.Sequential(
            layer.Conv2D(
                1, 8, [5, 5], [2, 2], 2, [1, 1],
                batch_norm=None, activation=act(act_p), name_prefix='enc1'
            ),
            layer.Conv2D(
                8, 16, [4, 4], [2, 2], 2, [1, 1],
                activation=act(act_p), name_prefix='enc2'
            ),
            layer.Conv2D(
                16, 32, [4, 4], [2, 2], 2, [1, 1],
                activation=act(act_p), name_prefix='enc3'
            ),
            layer.Conv2D(
                32, 64, [4, 4], [2, 2], 2, [1, 1],
                activation=act(act_p), name_prefix='enc4'
            ),
            layer.Conv2D(
                64, 128, [4, 4], [2, 2], 2, [1, 1],
                activation=act(act_p), name_prefix='enc5'
            ),
            layer.Conv2D(
                128, 256, [4, 4], [2, 2], 2, [1, 1],
                activation=act(act_p), name_prefix='enc6'
            )
        )

        if last_layers_to_remove <= 1:
            self.enc_nn.add_module(
                '4x4conv',
                layer.Conv2D(
                    256, 512, [4, 4], [2, 2], 2, [1, 1],
                    activation=act(act_p), name_prefix='enc7'
                )
            )

        if last_layers_to_remove == 0:
            self.enc_nn.add_module(
                '1x1conv',
                layer.Conv2D(
                    512, 1024, [1, 1], [1, 1], 0, [1, 1],
                    batch_norm=None, activation=act(act_p), name_prefix='enc8'
                )
            )

    def forward(self, x_spectrogram):
        return self.enc_nn(x_spectrogram)


class Head(nn.Module):
    def __init__(self, in_dim, out_dim, architecture: List[int] = [1024, 1024, 1024, 1024], dropout_p: int = 0.4):
        super().__init__()
        self.head = nn.Sequential()

        for i, hidden_dim in enumerate(architecture):
            if i == 0:
                self.head.add_module(f'fc{i + 1}', nn.Linear(in_dim, hidden_dim))
            else:
                self.head.add_module(f'fc{i + 1}', nn.Linear(hidden_dim, hidden_dim))
            
            if i < (len(architecture) - 1):
                self.head.add_module(f'bn{i + 1}', nn.BatchNorm1d(hidden_dim))
                self.head.add_module(f'drp{i + 1}', nn.Dropout(dropout_p))
            
            self.head.add_module(f'act{i + 1}', nn.ReLU())
        
        self.head.add_module(f'fc{len(architecture) + 1}', nn.Linear(hidden_dim, out_dim))
        self.head.add_module('act', nn.Hardtanh(min_val=0.0, max_val=1.0))

    def forward(self, x):
        return self.head(x)


class CNNBackbone(nn.Module):
    """
    CNN feature extractor for the backbone of Transformer encoder
    """

    def __init__(self, out_dim: int = 256, **backbone_kwargs):
        super().__init__()
        act = nn.LeakyReLU
        act_p = 0.1  # Activation param
        self.enc_nn = nn.Sequential(
            layer.Conv2D(
                backbone_kwargs['spectrogram_channels'], 16, [5, 5], [2, 2], 2, [1, 1],
                norm=None, activation=act(act_p), name_prefix='enc1'
            ),
            layer.Conv2D(
                16, 32, [4, 4], [2, 2], 2, [1, 1],
                activation=act(act_p), name_prefix='enc2'
            ),
            layer.Conv2D(
                32, 64, [4, 4], [2, 2], 2, [1, 1],
                activation=act(act_p), name_prefix='enc3'
            ),
            layer.Conv2D(
                64, 128, [4, 4], [2, 2], 2, [1, 1],
                activation=act(act_p), name_prefix='enc4'
            ),
            layer.Conv2D(
                128, out_dim, [4, 4], [2, 2], 2, [1, 1],
                activation=act(act_p), name_prefix='enc5'
            ),
            # layer.Conv2D(
            #     256, self.out_dim, [4, 4], [2, 2], 2, [1, 1],
            #     activation=act(act_p), name_prefix='enc6'
            # ),
        )

    def forward(self, x_spectrogram):
        return self.enc_nn(x_spectrogram)
    

class PatchEmbed(nn.Module):
    """
    Patch embedding
    """
    def __init__(
            self,
            spectrogram_size: List[int],
            out_dim: int,
            spectrogram_channels: int = 1,
            kernel_size: List[int] = [7, 7],
            stride: List[int] = [4, 6],
            padding: List[int] = [2, 2],
            norm: str = 'layernorm',
        ):
        super().__init__()
        act = nn.LeakyReLU
        act_p = 0.1
        out_shape = self._out_shape(
            spectrogram_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.proj = layer.Conv2D(
            spectrogram_channels,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=act(act_p),
            name_prefix='patch_embed',
            norm=norm,
            out_shape=out_shape,
        )

    def forward(self, x_spectrogram):
        return self.proj(x_spectrogram)

    def _out_shape(
            self,
            spectrogram_size: List[int],
            kernel_size: List[int],
            stride: List[int],
            padding: int = 2,
            dilation: List[int] = [1, 1],
        ):
        H_in, W_in = spectrogram_size
        H_out = (H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        W_out = (W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
        return [H_out, W_out]
    

class SynthTR(nn.Module):
    """
    SynthTR consists of CNN backbone, Transformer encoder and Transformer decoder.
    Output of the Transformer encoder serves as keys and values of the decoder.
    The transformer decoder receives learnable queries for synthesizer parameters.
    """

    def __init__(
            self,
            preset_idx_helper: PresetIndexesHelper,
            spectrogram_size: List[int],
            backbone: str = 'CNNBackbone',
            tr_encoder: str = 'TransformerEncoder',
            d_model: int = 256,
            n_queries: int = 144,
            backbone_kwargs: Dict[str, Any] = {},
            transformer_kwargs: Dict[str, Any] = {},
        ):
        super().__init__()
        self.out_dim = preset_idx_helper._learnable_preset_size
        self.cat_idx, self.num_idx = self._get_learnable_idx(preset_idx_helper)

        self.backbone = getattr(sys.modules[__name__], backbone)(
            spectrogram_size=spectrogram_size,
            **backbone_kwargs
        )
        self.tr_encoder = getattr(layer, tr_encoder)(
            d_model,
            spectrogram_size=spectrogram_size,
            **transformer_kwargs
        )
        self.tr_decoder = TransformerDecoder(n_queries, d_model, **transformer_kwargs)

        # Projection layers
        self.proj_dropout = nn.Dropout(0.3)
        self.proj = nn.Linear(d_model, self.out_dim)
        self.last_act = nn.Tanh()

        # For previous experiments (add dummy dropout layers)
        # self.proj = nn.Sequential(
        #     nn.Dropout(0),
        #     nn.Linear(d_model, self.out_dim),
        #     nn.Dropout(0),
        # )

        # Full sep
        proj = []

        for i in range(len(self.cat_idx)):
            proj.append(nn.Linear(d_model, len(self.cat_idx[i])))

        for i in range(len(self.num_idx)):
            proj.append(nn.Linear(d_model, 1))
        
        self.proj = nn.ModuleList(proj)

    def forward(self, spectrogram):
        features = self.backbone(spectrogram)
        enc_out, enc_pos_embed = self.tr_encoder(features)
        dec_out = self.tr_decoder(enc_out, memory_pos_embed=enc_pos_embed)

        # Synth param head
        batch_size, n_query, d_model = dec_out.shape
        out = torch.zeros((batch_size, self.out_dim), device=dec_out.device)
        
        dec_out = dec_out.reshape(-1, d_model)
        dec_out = self.proj_dropout(dec_out)
        dec_out = dec_out.reshape(batch_size, n_query, -1)

        for i in range(len(self.cat_idx)):
            out[:, self.cat_idx[i]] = self.proj[i](dec_out[:, i, :])

        for i in range(len(self.cat_idx), n_query):
            out[:, self.num_idx[i - len(self.cat_idx)]] = self.proj[i](dec_out[:, i, :]).squeeze()

        # Output Activation
        out = self.last_act(out)
        out = 0.5 * (out + 1.)
        return out
    
    def _get_learnable_idx(self, preset_idx_helper):
        full_idx = preset_idx_helper.full_to_learnable
        cat_idx, num_idx = [], []

        for idx in full_idx:
            if isinstance(idx, list):
                cat_idx.append(idx)
            elif isinstance(idx, int):
                num_idx.append(idx)

        return cat_idx, num_idx
