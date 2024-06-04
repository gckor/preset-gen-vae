
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from .metrics import BufferedMetric


class CorrectedSummaryWriter(SummaryWriter):
    """ SummaryWriter corrected to prevent extra runs to be created
    in Tensorboard when adding hparams.

    Original code in torch/utils/tensorboard.writer.py,
    modification by method overloading inspired by https://github.com/pytorch/pytorch/issues/32651 """

    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
        assert run_name is None  # Disabled feature. Run name init by summary writer ctor

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        # run_name argument is discarded and the writer itself is used (no extra writer instantiation)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


class TensorboardSummaryWriter(CorrectedSummaryWriter):
    """ Tensorboard SummaryWriter with corrected add_hparams method
     and extra functionalities. """

    def __init__(
        self,
        log_dir=None,
        comment='',
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix='',
        config=None,
    ):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        # Full-Config is required. Default constructor values allow to keep the same first constructor args
        self.config = config
        self.resume_from_checkpoint = (config.train.start_epoch > 0)
        self.hyper_params = dict()

        # General and dataset hparams
        self.hyper_params['batchsz'] = config.train.minibatch_size
        self.hyper_params['kfold'] = config.train.current_k_fold
        self.hyper_params['wdecay'] = config.train.weight_decay
        self.hyper_params['fcdrop'] = config.train.fc_dropout
        self.hyper_params['nmidi'] = '{}{}'.format(len(config.dataset.midi_notes),
                                                   ("stack" if config.model.stack_spectrograms else "inde"))
        self.hyper_params['catmodel'] = config.dataset.vst_params_learned_as_categorical
        self.hyper_params['normloss'] = config.train.normalize_losses
        
        # Latent space hparams
        self.hyper_params['z_dim'] = config.model.dim_z

        # Model hparams
        self.hyper_params['controls'] = config.synth_params_count
        self.hyper_params['regsoftm'] = config.model.params_reg_softmax
        self.hyper_params['regcatlo'] = 'BinCE' if config.train.params_cat_bceloss else 'CatCE'
        self.hyper_params['regarch'] = config.model.params_regression_architecture
        self.hyper_params['latfarch'] = config.model.latent_flow_arch
        self.hyper_params['encarch'] = config.model.encoder_architecture
        
        self.hyper_params['mels'] = config.dataset.n_mel_bins
        self.hyper_params['mindB'] = config.dataset.spectrogram_min_dB
        
    def init_hparams_and_metrics(self, metrics):
        """ Hparams and Metric initialization. Will pass if training resumes from saved checkpoint.
        Hparams will be definitely set but metrics can be updated during training.

        :param metrics: Dict of BufferedMetric
        """
        if not self.resume_from_checkpoint:  # tensorboard init at epoch 0 only
            # Some processing on hparams can be done here... none at the moment
            self.update_metrics(metrics)

    def update_metrics(self, metrics):
        """ Updates Tensorboard metrics

        :param metrics: Dict of values and/or BufferedMetric instances
        :return: None
        """
        metrics_dict = dict()
        for k, metric in metrics.items():
            if isinstance(metrics[k], BufferedMetric):
                try:
                    metrics_dict[k] = metric.mean
                except ValueError:
                    metrics_dict[k] = 0  # TODO appropriate default metric value?
            else:
                metrics_dict[k] = metric
        self.add_hparams(self.hyper_params, metrics_dict, hparam_domain_discrete=None)

