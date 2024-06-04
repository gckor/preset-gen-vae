import typing as tp
from typing import Iterable, Sequence, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from nflows.transforms.base import CompositeTransform
from model.conv import pad_for_conv1d

from data.preset import PresetIndexesHelper
import utils.probability


AdvLossType = tp.Union[nn.Module, tp.Callable[[torch.Tensor], torch.Tensor]]
FeatLossType = tp.Union[nn.Module, tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]


class L2Loss:
    """
    L2 (squared difference) loss, with customizable normalization (averaging) options.

    When used to model the reconstruction probability p_theta( x | zK ), normalization has strong
    implications on the p_theta( x | zK ) model itself.
    E.g., for a 1-element batch, the non-normalized L2 loss implies a learned mean, fixed 1/√2 std
    gaussian model for each element of x.
    When normalizing the L2 error (i.e. MSE error), the fixed std is multiplied by √(nb of elements of x)
    (e.g. approx *300 for a 250x350 pixels spectrogram)

    Normalization over batch dimension should always be performed (monte-carlo log-proba estimation).
    """
    def __init__(self, contents_average=False, batch_average=True):
        """

        :param contents_average: If True, the loss value will be divided by the number of elements of a batch item.
        :param batch_average: If True, the loss value will be divided by batch size
        """
        self.contents_average = contents_average
        self.batch_average = batch_average

    def __call__(self, inferred, target):
        loss = torch.sum(torch.square(inferred - target))
        if self.batch_average:
            loss = loss / inferred.shape[0]
        if self.contents_average:
            loss = loss / inferred[0, :].numel()
        return loss


class GaussianDkl:
    """ Kullback-Leibler Divergence between independant Gaussian distributions (diagonal
    covariance matrices). mu 2 and logs(var) 2 are optional and will be resp. zeros and zeros if not given.

    A normalization over the batch dimension will automatically be performed.
    An optional normalization over the channels dimension can also be performed.

    All tensor sizes should be (N_minibatch, N_channels) """
    def __init__(self, normalize=True):
        self.normalize = normalize  # Normalization over channels

    def __call__(self, mu1, logvar1, mu2=None, logvar2=None):
        if mu2 is None and logvar2 is None:
            Dkl = 0.5 * torch.sum(torch.exp(logvar1) + torch.square(mu1) - logvar1 - 1.0)
        else:
            raise NotImplementedError("General Dkl not implemented yet...")
        Dkl = Dkl / mu1.size(0)
        if self.normalize:
            return Dkl / mu1.size(1)
        else:
            return Dkl


# TODO MMD



class SynthParamsLoss:
    """ A 'dynamic' loss which handles different representations of learnable synth parameters
    (numerical and categorical). The appropriate loss can be computed by passing a PresetIndexesHelper instance
    to this class constructor.

    The categorical loss is categorical cross-entropy. """
    def __init__(self, idx_helper: PresetIndexesHelper, normalize_losses: bool, categorical_loss_factor=0.2,
                 prevent_useless_params_loss=True,
                 cat_bce=True, cat_softmax=False, cat_softmax_t=0.1):
        """

        :param idx_helper: PresetIndexesHelper instance, created by a PresetDatabase, to convert vst<->learnable params
        :param normalize_losses: If True, losses will be divided by batch size and number of parameters
            in a batch element. If False, losses will only be divided by batch size.
        :param categorical_loss_factor: Factor to be applied to the categorical cross-entropy loss, which is
            much greater than the 'corresponding' MSE loss (if the parameter was learned as numerical)
        :param prevent_useless_params_loss: If True, the class will search for useless params (e.g. params which
            correspond to a disabled oscillator and have no influence on the output sound). This introduces a
            TODO describe overhead here
        :param cat_softmax: Should be set to True if the regression network does not apply softmax at its output.
            This implies that a Categorical Cross-Entropy Loss will be computed on categorical sub-vectors.
        :param cat_softmax_t: Temperature of the softmax activation applied to cat parameters
        :param cat_bce: Binary Cross-Entropy applied to independent outputs (see InverSynth 2019). Very bad
            perfs but option remains available.
        """
        self.idx_helper = idx_helper
        self.normalize_losses = normalize_losses
        if cat_bce and cat_softmax:
            raise ValueError("'cat_bce' (Binary Cross-Entropy) and 'cat_softmax' (implies Categorical Cross-Entropy)"
                             "cannot be both set to True")
        self.cat_bce = cat_bce
        self.cat_softmax = cat_softmax
        self.cat_softmax_t = cat_softmax_t
        self.cat_loss_factor = categorical_loss_factor
        self.prevent_useless_params_loss = prevent_useless_params_loss
        # Numerical loss criterion
        if self.normalize_losses:
            self.numerical_criterion = nn.MSELoss(reduction='mean')
        else:
            self.numerical_criterion = L2Loss()
        # Pre-compute indexes lists (to use less CPU). 'num' stands for 'numerical' (not number)
        self.num_indexes = self.idx_helper.get_numerical_learnable_indexes()
        self.cat_indexes = self.idx_helper.get_categorical_learnable_indexes()

    def __call__(self, u_out: torch.Tensor, u_in: torch.Tensor):
        """ Categorical parameters must be one-hot encoded. """
        # At first: we search for useless parameters (whose loss should not be back-propagated)
        useless_num_learn_param_indexes, useless_cat_learn_param_indexes = list(), list()
        batch_size = u_in.shape[0]
        if self.prevent_useless_params_loss:
            for row in range(batch_size):
                num_indexes, cat_indexes = self.idx_helper.get_useless_learned_params_indexes(u_in[row, :])
                useless_num_learn_param_indexes.append(num_indexes)
                useless_cat_learn_param_indexes.append(cat_indexes)
        num_loss = 0.0  # - - - numerical loss - - -
        if len(self.num_indexes) > 0:
            if self.prevent_useless_params_loss:
                # apply a 0.0 factor for disabled parameters (e.g. Dexed operator w/ output level 0.0)
                for row in range(u_in.shape[0]):
                    for num_idx in self.num_indexes:
                        if num_idx in useless_num_learn_param_indexes[row]:
                            u_in[row, num_idx] = 0.0
                            u_out[row, num_idx] = 0.0
            num_loss = self.numerical_criterion(u_out[:, self.num_indexes], u_in[:, self.num_indexes])
        cat_loss = 0.0  # - - - categorical loss - - -
        if len(self.cat_indexes) > 0:
            # For each categorical output (separate loss computations...)
            for cat_learn_indexes in self.cat_indexes:  # type: list
                # don't compute cat loss for disabled parameters (e.g. Dexed operator w/ output level 0.0)
                rows_to_remove = list()
                if self.prevent_useless_params_loss:
                    for row in range(batch_size):  # Need to check cat index 0 only
                        if cat_learn_indexes[0] in useless_cat_learn_param_indexes[row]:
                            rows_to_remove.append(row)
                useful_rows = None  # None means that the full batch is useful
                if len(rows_to_remove) > 0:  # If this batch contains useless inferred params
                    useful_rows = list(range(0, batch_size))
                    for row in rows_to_remove:
                        useful_rows.remove(row)
                # Direct cross-entropy computation. The one-hot target is used to select only q output probabilities
                # corresponding to target classes with p=1. We only need a limited number of output probabilities
                # (they actually all depend on each other thanks to the softmax output layer).
                if not self.cat_bce:  # Categorical CE
                    target_one_hot = u_in[:, cat_learn_indexes].bool()  # Will be used for tensor-element selection
                else:  # Binary CE: float values required
                    target_one_hot = u_in[:, cat_learn_indexes]
                if useful_rows is not None:  # Some rows can be discarded from loss computation
                    target_one_hot = target_one_hot[useful_rows, :]
                q_odds = u_out[:, cat_learn_indexes]  # contains all q odds required for BCE or CCE
                # The same rows must be discarded from loss computation (if the preset didn't use this cat param)
                if useful_rows is not None:
                    q_odds = q_odds[useful_rows, :]
                if not self.cat_bce:  # - - - - - NOT Binary CE => Categorical CE - - - - -
                    # softmax T° if required: q_odds might not sum to 1.0 already if no softmax was applied before
                    if self.cat_softmax:
                        q_odds = torch.softmax(q_odds / self.cat_softmax_t, dim=1)
                    # Then the cross-entropy can be computed (simplified formula thanks to p=1.0 one-hot odds)
                    q_odds = q_odds[target_one_hot]  # CE uses only 1 odd per output vector (thanks to softmax)
                    # batch-sum and normalization vs. batch size
                    param_cat_loss = - torch.sum(torch.log(q_odds)) / (batch_size - len(rows_to_remove))
                else:  # - - - - - Binary Cross-Entropy - - - - -
                    # empirical normalization factor - works quite well to get similar CCE and BCE values
                    param_cat_loss = F.binary_cross_entropy(q_odds, target_one_hot, reduction='mean') / 8.0
                # CCE and BCE: add the temp per-param loss
                cat_loss += param_cat_loss
                # TODO instead of final factor: maybe divide the each cat loss by the one-hot vector length?
                #    maybe not: cross-entropy always uses only 1 of the odds... (softmax does the job before)
            if self.normalize_losses:  # Normalization vs. number of categorical-learned params
                cat_loss = cat_loss / len(self.cat_indexes)
        # losses weighting - Cross-Entropy is usually be much bigger than MSE. num_loss
        return num_loss + cat_loss * self.cat_loss_factor



class QuantizedNumericalParamsLoss:
    """ 'Quantized' parameters loss: to get a meaningful (but non-differentiable) loss, inferred parameter
    values must be quantized as they would be in the synthesizer.

    Only numerical parameters are involved in this loss computation. The PresetIndexesHelper ctor argument
    allows this class to know which params are numerical.
    The loss to be applied after quantization can be passed as a ctor argument.

    This loss breaks the computation path (.backward cannot be applied to it).
    """
    def __init__(self, idx_helper: PresetIndexesHelper, numerical_loss=nn.MSELoss(),
                 reduce: bool = True, limited_vst_params_indexes: Optional[Sequence] = None):
        """

        :param idx_helper:
        :param numerical_loss:
        :param limited_vst_params_indexes: List of VST params to include into to the loss computation. Can be uses
            to measure performance of specific groups of params. Set to None to include all numerical parameters.
        """
        self.idx_helper = idx_helper
        self.numerical_loss = numerical_loss
        # Cardinality checks
        for vst_idx, _ in self.idx_helper.num_idx_learned_as_cat.items():
            assert self.idx_helper.vst_param_cardinals[vst_idx] > 0
        # Number of numerical parameters considered for this loss (after cat->num conversions). For tensor pre-alloc
        self.num_params_count = len(self.idx_helper.num_idx_learned_as_num)\
                                + len(self.idx_helper.num_idx_learned_as_cat)
        self.limited_vst_params_indexes = limited_vst_params_indexes
        self.reduce = reduce

    def __call__(self, u_out: torch.Tensor, u_in: torch.Tensor):
        """ Returns the loss for numerical VST params only (searched in u_in and u_out).
        Learnable representations can be numerical (in [0.0, 1.0]) or one-hot categorical.
        The type of representation has been stored in self.idx_helper """
        errors = dict()
        # Partial tensors (for final loss computation)
        minibatch_size = u_in.size(0)
        # pre-allocate tensors
        u_in_num = torch.empty((minibatch_size, self.num_params_count), device=u_in.device, requires_grad=False)
        u_out_num = torch.empty((minibatch_size, self.num_params_count), device=u_in.device, requires_grad=False)
        # if limited vst indexes: fill with zeros (some allocated cols won't be used). Slow but used for eval only.
        if self.limited_vst_params_indexes is not None:
            u_in_num[:, :], u_out_num[:, :] = 0.0, 0.0
        # Column-by-column tensors filling
        cur_num_tensors_col = 0
        # quantize numerical learnable representations
        for vst_idx, learn_idx in self.idx_helper.num_idx_learned_as_num.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            param_batch = u_in[:, learn_idx].detach()
            u_in_num[:, cur_num_tensors_col] = param_batch  # Data copy - does not modify u_in
            param_batch = u_out[:, learn_idx].detach().clone()
            if self.idx_helper.vst_param_cardinals[vst_idx] > 0:  # don't quantize <0 cardinal (continuous)
                cardinal = self.idx_helper.vst_param_cardinals[vst_idx]
                param_batch = torch.round(param_batch * (cardinal - 1.0)) / (cardinal - 1.0)
            u_out_num[:, cur_num_tensors_col] = param_batch
            errors[vst_idx] = self.numerical_loss(u_in_num[:, cur_num_tensors_col], u_out_num[:, cur_num_tensors_col]).item()
            cur_num_tensors_col += 1
        # convert one-hot encoded learnable representations of (quantized) numerical VST params
        for vst_idx, learn_indexes in self.idx_helper.num_idx_learned_as_cat.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            cardinal = len(learn_indexes)
            # Classes as column-vectors (for concatenation)
            in_classes = torch.argmax(u_in[:, learn_indexes], dim=-1).detach().type(torch.float)
            u_in_num[:, cur_num_tensors_col] = in_classes / (cardinal-1.0)
            out_classes = torch.argmax(u_out[:, learn_indexes], dim=-1).detach().type(torch.float)
            u_out_num[:, cur_num_tensors_col] = out_classes / (cardinal-1.0)
            errors[vst_idx] = self.numerical_loss(u_in_num[:, cur_num_tensors_col], u_out_num[:, cur_num_tensors_col]).item()
            cur_num_tensors_col += 1
        # Final size checks
        if self.limited_vst_params_indexes is None:
            assert cur_num_tensors_col == self.num_params_count
        else:
            pass  # No size check for limited params (a list with unlearned and/or cat params can be provided)
            #  assert cur_num_tensors_col == len(self.limited_vst_params_indexes)
        
        if self.reduce:
            return self.numerical_loss(u_out_num, u_in_num)  # Positive diff. if output > input
        else:
            return errors, self.numerical_loss(u_out_num, u_in_num)



class CategoricalParamsAccuracy:
    """ Only categorical parameters are involved in this loss computation. """
    def __init__(self, idx_helper: PresetIndexesHelper, reduce=True, percentage_output=True,
                 limited_vst_params_indexes: Optional[Sequence] = None):
        """
        :param idx_helper: allows this class to know which params are categorical
        :param reduce: If True, an averaged accuracy will be returned. If False, a dict of accuracies (keys =
          vst param indexes) is returned.
        :param percentage_output: If True, accuracies in [0.0, 100.0], else in [0.0, 1.0]
        :param limited_vst_params_indexes: List of VST params to include into to the loss computation. Can be uses
            to measure performance of specific groups of params. Set to None to include all numerical parameters.
        """
        self.idx_helper = idx_helper
        self.reduce = reduce
        self.percentage_output = percentage_output
        self.limited_vst_params_indexes = limited_vst_params_indexes

    def __call__(self, u_out: torch.Tensor, u_in: torch.Tensor):
        """ Returns accuracy (or accuracies) for all categorical VST params.
        Learnable representations can be numerical (in [0.0, 1.0]) or one-hot categorical.
        The type of representation is stored in self.idx_helper """
        accuracies = dict()
        # Accuracy of numerical learnable representations (involves quantization)
        for vst_idx, learn_idx in self.idx_helper.cat_idx_learned_as_num.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            cardinal = self.idx_helper.vst_param_cardinals[vst_idx]
            param_batch = torch.unsqueeze(u_in[:, learn_idx].detach(), 1)  # Column-vector
            # Class indexes, from 0 to cardinal-1
            target_classes = torch.round(param_batch * (cardinal - 1.0)).type(torch.int32)  # New tensor allocated
            param_batch = torch.unsqueeze(u_out[:, learn_idx].detach(), 1)
            out_classes = torch.round(param_batch * (cardinal - 1.0)).type(torch.int32)  # New tensor allocated
            accuracies[vst_idx] = (target_classes == out_classes).count_nonzero().item() / target_classes.numel()
        # accuracy of one-hot encoded categorical learnable representations
        for vst_idx, learn_indexes in self.idx_helper.cat_idx_learned_as_cat.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            target_classes = torch.argmax(u_in[:, learn_indexes], dim=-1)  # New tensor allocated
            out_classes = torch.argmax(u_out[:, learn_indexes], dim=-1)  # New tensor allocated
            accuracies[vst_idx] = (target_classes == out_classes).count_nonzero().item() / target_classes.numel()
        # Factor 100.0?
        if self.percentage_output:
            for k, v in accuracies.items():
                accuracies[k] = v * 100.0
        # Reduction if required
        if self.reduce:
            return np.asarray([v for _, v in accuracies.items()]).mean()
        else:
            return accuracies


class FlowParamsLoss:
    """
    Estimates the Dkl between the true distribution of synth params p*(v) and the current p_lambda(v) distribution.

    This requires to invert two flows (the regression and the latent flow) in order to estimate the probability of
    some v_in target parameters in the q_Z0(z0) distribution (z0 = invT(invU(v)).
    These invert flows (ideally parallelized) must be provided in the loss constructor
    """
    def __init__(self, idx_helper: PresetIndexesHelper, latent_flow_inverse_function, reg_flow_inverse_function):
        self.idx_helper = idx_helper
        self.latent_flow_inverse_function = latent_flow_inverse_function
        self.reg_flow_inverse_function = reg_flow_inverse_function

    def __call__(self, z_0_mu_logvar, v_target):
        """ Estimate the probability of v_target in the q_Z0(z0) distribution (see details in TODO REF) """

        # FIXME v_target should be "inverse-softmaxed" (because actual output will be softmaxed)

        # TODO apply a factor on categorical params (maybe divide by the size of the one-hot encoded vector?)
        #    how to do that with this inverse flow transform??????

        # Flows reversing - sum of log abs det of inverse Jacobian is used in the loss
        z_K, log_abs_det_jac_inverse_U = self.reg_flow_inverse_function(v_target)
        z_0, log_abs_det_jac_inverse_T = self.latent_flow_inverse_function(z_K)
        # Evaluate q_Z0(z0) (closed-form gaussian probability)
        z_0_log_prob = utils.probability.gaussian_log_probability(z_0, z_0_mu_logvar[:, 0, :], z_0_mu_logvar[:, 1, :])
        # Result is batch-size normalized
        # TODO loss factor as a ctor arg
        return - torch.mean(z_0_log_prob + log_abs_det_jac_inverse_T + log_abs_det_jac_inverse_U) / 1000.0


class MelSpectrogramWrapper(nn.Module):
    """Wrapper around MelSpectrogram torchaudio transform providing proper padding
    and additional post-processing including log scaling.

    Args:
        n_mels (int): Number of mel bins.
        n_fft (int): Number of fft.
        hop_length (int): Hop size.
        win_length (int): Window length.
        n_mels (int): Number of mel bins.
        sample_rate (int): Sample rate.
        f_min (float or None): Minimum frequency.
        f_max (float or None): Maximum frequency.
        log (bool): Whether to scale with log.
        normalized (bool): Whether to normalize the melspectrogram.
        floor_level (float): Floor level based on human perception (default=1e-5).
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: tp.Optional[int] = None,
                 n_mels: int = 80, sample_rate: float = 22050, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.n_fft = n_fft
        hop_length = int(hop_length)
        self.hop_length = hop_length
        self.mel_transform = MelSpectrogram(n_mels=n_mels, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                            win_length=win_length, f_min=f_min, f_max=f_max, normalized=normalized,
                                            window_fn=torch.hann_window, center=False)
        self.floor_level = floor_level
        self.log = log

    def forward(self, x):
        p = int((self.n_fft - self.hop_length) // 2)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (p, p), "reflect")
        # Make sure that all the frames are full.
        # The combination of `pad_for_conv1d` and the above padding
        # will make the output of size ceil(T / hop).
        x = pad_for_conv1d(x, self.n_fft, self.hop_length)
        self.mel_transform.to(x.device)
        mel_spec = self.mel_transform(x)
        B, C, freqs, frame = mel_spec.shape
        if self.log:
            mel_spec = torch.log10(self.floor_level + mel_spec)
        return mel_spec.reshape(B, C * freqs, frame)
    

class MultiScaleMelSpectrogramLoss(nn.Module):
    """Multi-Scale spectrogram loss (msspec).

    Args:
        sample_rate (int): Sample rate.
        range_start (int): Power of 2 to use for the first scale.
        range_stop (int): Power of 2 to use for the last scale.
        n_mels (int): Number of mel bins.
        f_min (float): Minimum frequency.
        f_max (float or None): Maximum frequency.
        normalized (bool): Whether to normalize the melspectrogram.
        alphas (bool): Whether to use alphas as coefficients or not.
        floor_level (float): Floor level value based on human perception (default=1e-5).
    """
    def __init__(self, sample_rate: int, range_start: int = 6, range_end: int = 11,
                 n_mels: int = 64, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 normalized: bool = False, alphas: bool = True, floor_level: float = 1e-5):
        super().__init__()
        l1s = list()
        l2s = list()
        self._alphas = list()
        for i in range(range_start, range_end):
            l1s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=False, normalized=normalized, floor_level=floor_level))
            l2s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=True, normalized=normalized, floor_level=floor_level))
            if alphas:
                self._alphas.append(np.sqrt(2 ** i - 1))
            else:
                self._alphas.append(1)

        self.l1s = nn.ModuleList(l1s)
        self.l2s = nn.ModuleList(l2s)

    def forward(self, x, y):
        msspecs = []
        self.l1s.to(x.device)
        self.l2s.to(x.device)
        for i in range(len(self.alphas)):
            s_x_1 = self.l1s[i](x)
            s_y_1 = self.l1s[i](y)
            s_x_2 = self.l2s[i](x)
            s_y_2 = self.l2s[i](y)
            msspecs.append(torch.stack([s_x_1, s_y_1, s_x_2, s_y_2]).permute(1, 0, 2, 3))
        return msspecs
    
    @property
    def alphas(self):
        return self._alphas



class AdversarialLoss(nn.Module):
    """Adversary training wrapper.

    Args:
        adversary (nn.Module): The adversary module will be used to estimate the logits given the fake and real samples.
            We assume here the adversary output is ``Tuple[List[torch.Tensor], List[List[torch.Tensor]]]``
            where the first item is a list of logits and the second item is a list of feature maps.
        optimizer (torch.optim.Optimizer): Optimizer used for training the given module.
        loss (AdvLossType): Loss function for generator training.
        loss_real (AdvLossType): Loss function for adversarial training on logits from real samples.
        loss_fake (AdvLossType): Loss function for adversarial training on logits from fake samples.
        loss_feat (FeatLossType): Feature matching loss function for generator training.
        normalize (bool): Whether to normalize by number of sub-discriminators.

    Example of usage:
        adv_loss = AdversarialLoss(adversaries, optimizer, loss, loss_real, loss_fake)
        for real in loader:
            noise = torch.randn(...)
            fake = model(noise)
            adv_loss.train_adv(fake, real)
            loss, _ = adv_loss(fake, real)
            loss.backward()
    """
    def __init__(self,
                 adversary: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss: AdvLossType,
                 loss_real: AdvLossType,
                 loss_fake: AdvLossType,
                 loss_feat: tp.Optional[FeatLossType] = None,
                 normalize: bool = True):
        super().__init__()
        self.adversary: nn.Module = adversary
        self.optimizer = optimizer
        self.loss = loss
        self.loss_real = loss_real
        self.loss_fake = loss_fake
        self.loss_feat = loss_feat
        self.normalize = normalize

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Add the optimizer state dict inside our own.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'optimizer'] = self.optimizer.state_dict()
        return destination

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Load optimizer state.
        self.optimizer.load_state_dict(state_dict.pop(prefix + 'optimizer'))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_adversary_pred(self, x):
        """Run adversary model, validating expected output format."""
        logits, fmaps = self.adversary(x)
        assert isinstance(logits, list) and all([isinstance(t, torch.Tensor) for t in logits]), \
            f'Expecting a list of tensors as logits but {type(logits)} found.'
        assert isinstance(fmaps, list), f'Expecting a list of features maps but {type(fmaps)} found.'
        for fmap in fmaps:
            assert isinstance(fmap, list) and all([isinstance(f, torch.Tensor) for f in fmap]), \
                f'Expecting a list of tensors as feature maps but {type(fmap)} found.'
        return logits, fmaps

    def train_adv(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Train the adversary with the given fake and real example.

        We assume the adversary output is the following format: Tuple[List[torch.Tensor], List[List[torch.Tensor]]].
        The first item being the logits and second item being a list of feature maps for each sub-discriminator.

        This will automatically synchronize gradients (with `flashy.distrib.eager_sync_model`)
        and call the optimizer.
        """
        loss = torch.tensor(0., device=fake.device)
        all_logits_fake_is_fake, _ = self.get_adversary_pred(fake.detach())
        all_logits_real_is_fake, _ = self.get_adversary_pred(real.detach())
        n_sub_adversaries = len(all_logits_fake_is_fake)
        for logit_fake_is_fake, logit_real_is_fake in zip(all_logits_fake_is_fake, all_logits_real_is_fake):
            loss += self.loss_fake(logit_fake_is_fake) + self.loss_real(logit_real_is_fake)

        if self.normalize:
            loss /= n_sub_adversaries

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Return the loss for the generator, i.e. trying to fool the adversary,
        and feature matching loss if provided.
        """
        adv = torch.tensor(0., device=fake.device)
        feat = torch.tensor(0., device=fake.device)
        with torch.no_grad():
            all_logits_fake_is_fake, all_fmap_fake = self.get_adversary_pred(fake)
            all_logits_real_is_fake, all_fmap_real = self.get_adversary_pred(real)
            n_sub_adversaries = len(all_logits_fake_is_fake)
            for logit_fake_is_fake in all_logits_fake_is_fake:
                adv += self.loss(logit_fake_is_fake)
            if self.loss_feat:
                for fmap_fake, fmap_real in zip(all_fmap_fake, all_fmap_real):
                    feat += self.loss_feat(fmap_fake, fmap_real)

        if self.normalize:
            adv /= n_sub_adversaries
            feat /= n_sub_adversaries

        return adv, feat


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for adversarial training.

    Args:
        loss (nn.Module): Loss to use for feature matching (default=torch.nn.L1).
        normalize (bool): Whether to normalize the loss.
            by number of feature maps.
    """
    def __init__(self, loss: nn.Module = torch.nn.L1Loss(), normalize: bool = True):
        super().__init__()
        self.loss = loss
        self.normalize = normalize

    def forward(self, fmap_fake: tp.List[torch.Tensor], fmap_real: tp.List[torch.Tensor]) -> torch.Tensor:
        assert len(fmap_fake) == len(fmap_real) and len(fmap_fake) > 0
        feat_loss = torch.tensor(0., device=fmap_fake[0].device)
        feat_scale = torch.tensor(0., device=fmap_fake[0].device)
        n_fmaps = 0
        for (feat_fake, feat_real) in zip(fmap_fake, fmap_real):
            assert feat_fake.shape == feat_real.shape
            n_fmaps += 1
            feat_loss += self.loss(feat_fake, feat_real)
            feat_scale += torch.mean(torch.abs(feat_real))

        if self.normalize:
            feat_loss /= n_fmaps

        return feat_loss



def hinge_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor([0.0], device=x.device)
    return -x.mean()


def hinge_real_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.min(x - 1, torch.tensor(0., device=x.device).expand_as(x)))


def hinge_fake_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.min(-x - 1, torch.tensor(0., device=x.device).expand_as(x)))


def info_nce_loss(features, batch_size, n_views=None, temperature=0.07):
    if n_views is None:
        n_views = int(features.shape[0] / batch_size)
    
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    return logits, labels