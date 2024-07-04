
import torch
import torch.nn as nn
from typing import Dict, Any, List
from functools import partial
from timm.models.vision_transformer import Block
from timm.models.swin_transformer import SwinTransformerBlock

from data.preset import PresetIndexesHelper
from model import layer
from model.layer import Transformer
from model.position_encoding import PositionEmbeddingSine, PositionalEncoding1D
from model.audiomae.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible
from model.audiomae.patch_embed import PatchEmbed_org


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


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=1,
                 embed_dim=612, depth=12, num_heads=12, avg_pool=False,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False, 
                 audio_exp=True, alpha=0.0, temperature=.2, mode=0, contextual_depth=8,
                 pos_trainable=False, use_nce=False, beta=4.0, decoder_mode=1,
                 mask_t_prob=0.6, mask_f_prob=0.5, mask_2d=False,
                 epoch=0, no_shift=False, stochastic_latent=False,
                 ):
        super().__init__()

        self.audio_exp=audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.avg_pool = avg_pool
        self.stochastic_latent = stochastic_latent

        if stochastic_latent:
            self.mlp = nn.Linear(embed_dim, embed_dim * 2)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding


        self.no_shift=no_shift


        self.decoder_mode = decoder_mode
        window_size= (8, 7)
        feat_size = (16, 21)

        if self.decoder_mode == 1:
            decoder_modules = []

            for index in range(16):
                if self.no_shift:
                    shift_size = (0,0)
                else:
                    if (index % 2) == 0:
                        shift_size = (0,0)
                    else:
                        shift_size = (2,0)

                decoder_modules.append(
                    SwinTransformerBlock(
                        dim=decoder_embed_dim,
                        num_heads=16,
                        feat_size=feat_size,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        drop=0.0,
                        drop_attn=0.0,
                        drop_path=0.0,
                        extra_norm=False,
                        sequential_attn=False,
                        norm_layer=norm_layer, #nn.LayerNorm,
                    )
                )
            self.decoder_blocks = nn.ModuleList(decoder_modules)        
        else:
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.patch_size=patch_size
        self.stride=stride

        # audio exps
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        self.log_softmax=nn.LogSoftmax(dim=-1)

        self.mask_t_prob=mask_t_prob
        self.mask_f_prob=mask_f_prob
        self.mask_2d=mask_2d

        self.epoch = epoch

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)    
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.audio_exp:   
            decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
        else:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        L = (H/p)*(W/p)
        """
        p = self.patch_embed.patch_size[0]
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        if self.audio_exp:
            h = imgs.shape[2] // p
            w = imgs.shape[3] // p
            #h,w = self.patch_embed.patch_hw
            x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        else:
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        specs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]    
        h = 1024//p
        w = 128//p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        specs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return specs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim     
        T=64
        F=8
        #x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))

        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = torch.argsort(noise_t, dim=1) # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1) 
        ids_keep_t = ids_shuffle_t[:,:len_keep_t]
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(noise_f, dim=1) # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1) 
        ids_keep_f = ids_shuffle_f[:,:len_keep_f] #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:,:len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1,T,1) # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:,:len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1,F,1).permute(0,2,1) # N,T,F
        mask = 1-(1-mask_t)*(1-mask_f) # N, T, F

        # get masked x
        id2res=torch.Tensor(list(range(N*T*F))).reshape(N,T,F).to(x.device)
        id2res = id2res + 999*mask # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:,:len_keep_f*len_keep_t]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio, mask_2d=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_2d:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x, mask, ids_restore

    def forward_encoder_no_mask(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.avg_pool:
            emb = x[:, 1:, :].mean(dim=1)
        else:
            emb = x[:, 0]

        if self.stochastic_latent:
            emb = self.mlp(emb)
            emb = emb.reshape(x.shape[0], 2, self.embed_dim)

        return emb

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        
        if self.decoder_mode != 0:
            B, L, D = x.shape
            x = x[:, 1:, :]
        
        if self.decoder_mode > 3: # mvit
            x = self.decoder_blocks(x)
        else:
            for blk in self.decoder_blocks:
                x = blk(x)
        
        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)

        # remove cls token
        if self.decoder_mode != 0:
            pred = pred
        else:
            pred = pred[:, 1:, :]
        
        return pred

    def forward_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss      

    def forward(self, x):
        emb = self.forward_encoder_no_mask(x)
        return emb
    
    def get_mask_predict_loss(self, x, mask_ratio=0.8):
        emb_enc, mask, ids_restore = self.forward_encoder(x, mask_ratio, mask_2d=self.mask_2d)
        pred = self.forward_decoder(emb_enc, ids_restore)
        loss_recon = self.forward_loss(x, pred, mask, norm_pix_loss=self.norm_pix_loss)
        return loss_recon


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

    def __init__(self, spectrogram_channels: int = 1, out_dim: int = 256):
        super().__init__()
        act = nn.LeakyReLU
        act_p = 0.1  # Activation param
        self.enc_nn = nn.Sequential(
            layer.Conv2D(
                spectrogram_channels, 16, [5, 5], [2, 2], 2, [1, 1],
                batch_norm=None, activation=act(act_p), name_prefix='enc1'
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
    

class SynthTR(nn.Module):
    """
    SynthTR consists of CNN backbone, Transformer encoder and Transformer decoder.
    Output of the Transformer encoder serves as keys and values of the decoder.
    The transformer decoder receives learnable queries for synthesizer parameters.
    """

    def __init__(
            self,
            preset_idx_helper: PresetIndexesHelper,
            d_model: int = 256,
            spectrogram_channels: int = 1,
            num_queries: int = 144,
            transformer_kwargs: Dict[str, Any] = {},
        ):
        super().__init__()
        self.out_dim = preset_idx_helper._learnable_preset_size
        self.cat_idx, self.num_idx = self._get_learnable_idx(preset_idx_helper)

        self.backbone = CNNBackbone(spectrogram_channels, d_model)
        self.enc_pos_embed = PositionEmbeddingSine(d_model // 2)
        self.query_pos_embed = PositionalEncoding1D(d_model, num_queries)
        self.transformer = Transformer(num_queries, d_model, **transformer_kwargs)

        # Projection layers
        self.proj = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, self.out_dim),
            nn.Tanh(),
        )

        # Fullsep
        # projections = []

        # Shallow
        # for i in range(len(cat_idx)):
        #     projections.append(
        #         nn.Sequential(
        #             nn.Dropout(0.3),
        #             nn.Linear(dim_z, len(cat_idx[i])),
        #             nn.Tanh(),
        #         )
        #     )

        # for i in range(len(num_idx)):
        #     projections.append(
        #         nn.Sequential(
        #             nn.Dropout(0.3),
        #             nn.Linear(dim_z, 1),
        #             nn.Tanh(),
        #         )
        #     )


        # Deep
        # for i in range(len(cat_idx)):
        #     projections.append(
        #         nn.Sequential(
        #             nn.Dropout(0.3),
        #             nn.Linear(dim_z, 512),
        #             nn.BatchNorm1d(512),
        #             nn.LeakyReLU(0.1),
        #             nn.Dropout(0.3),
        #             nn.Linear(512, len(cat_idx[i])),
        #             nn.Tanh(),
        #         )
        #     )

        # for i in range(len(num_idx)):
        #     projections.append(
        #         nn.Sequential(
        #             nn.Dropout(0.3),
        #             nn.Linear(dim_z, 512),
        #             nn.BatchNorm1d(512),
        #             nn.LeakyReLU(0.1),
        #             nn.Dropout(0.3),
        #             nn.Linear(512, 1),
        #             nn.Tanh(),
        #         )
        #     )
        
        # self.projections = nn.ModuleList(projections)
                

        # Deep projection
        # self.mlp = nn.Sequential(
        #     nn.Dropout(0.3),
        #     nn.Linear(256, 512), #36864(dec flatten) #27648 (enc flatten) # 256(gap)
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 610),
        #     nn.Tanh(),
        # )

    def forward(self, spectrogram):
        features = self.backbone(spectrogram)
        enc_pos_embed = self.enc_pos_embed(features)
        query_pos_embed = self.query_pos_embed(features)

        # only encoder
        # enc_out = self.transformer(features, query_pos_embed, pos_embed)

        # # Flatten
        # enc_out = enc_out.flatten(1)

        # Global Average Pool
        # enc_out = enc_out.flatten(2).mean(dim=2) 
        
        # out = self.mlp(enc_out)
        # out = 0.5 * (out + 1.) # Tanh

        # out = self.head(out)

        # decoder
        dec_out = self.transformer(features, query_pos_embed, enc_pos_embed)

        # Flatten
        # dec_out = dec_out.flatten(1)

        # GAP
        # dec_out = dec_out.mean(dim=1)

        # Fullsep
        # batch_size, n_query, n_channel = dec_out.shape
        # out = torch.zeros((batch_size, 610), device=dec_out.device)

        # for i in range(len(self.cat_idx)):
        #     out[:, self.cat_idx[i]] = self.projections[i](dec_out[:, i, :])

        # for i in range(len(self.cat_idx), n_query):
        #     out[:, self.num_idx[i - len(self.cat_idx)]] = self.projections[i](dec_out[:, i, :]).squeeze()

        # out = 0.5 * (out + 1.) # Tanh
        

        # # Sep-head
        batch_size, n_query, d_model = dec_out.shape
        dec_out = dec_out.reshape(-1, d_model)
        
        # # Projection
        dec_out = self.proj(dec_out)
        dec_out = 0.5 * (dec_out + 1.) # Tanh

        # # Sep-head
        dec_out = dec_out.reshape(batch_size, n_query, -1)
        cat_out = dec_out[:, :len(self.cat_idx), :]
        num_out = dec_out[:, len(self.cat_idx):, :]

        out = torch.zeros((batch_size, self.out_dim), device=dec_out.device)

        for i in range(len(self.cat_idx)):
            out[:, self.cat_idx[i]] = cat_out[:, i, self.cat_idx[i]]

        for j in range(len(self.num_idx)):
            out[:, self.num_idx[j]] = num_out[:, j, self.num_idx[j]]

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
