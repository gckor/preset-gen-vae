from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from pyvirtualdisplay import Display
import torch
import numpy as np
import multiprocessing

from data.build import get_dataset, get_split_dataloaders
from logs import logger
from logs.metrics import EpochMetric
from model.encoder import SynthTR
from model.loss import PresetProcessor, SynthParamsLoss
from utils.audio import SimilarityEvaluator


def measure_spec_mae_worker(worker_args):
    return measure_spec_mae(*worker_args)

def measure_spec_mae(x_wav, full_preset_out, midi_pitch, midi_velocity):
    spec_maes = []

    for i in range(full_preset_out.shape[0]):
        x_wav_inferred, _ = dataset._render_audio(full_preset_out[i], int(midi_pitch[i]), int(midi_velocity[i]))
        similarity_eval = SimilarityEvaluator((x_wav[i], x_wav_inferred))
        spec_maes.append(similarity_eval.get_mae_log_stft(return_spectrograms=False))

    return np.array(spec_maes)


if __name__ == '__main__':
    # Finetune config
    logs_root_dir = Path('/data2/personal/swc/exp_logs/preset-gen-vae')
    model_path = logs_root_dir.joinpath('rlft_num_as_25_cls/kfold0-s1')
    device = 'cuda'
    n_epochs = 20
    midi_pitch = 60
    midi_velocity = 85

    # Main
    config = OmegaConf.load(model_path.joinpath('config.yaml'))
    dataset = get_dataset(config)
    dataloader = get_split_dataloaders(config, dataset)
    preset_idx_helper = dataset.preset_indexes_helper
    device = torch.device(device)
    checkpoint = logger.get_model_last_checkpoint(logs_root_dir, config, device=device)
    model = SynthTR(preset_idx_helper, **config.model.encoder_kwargs)
    model.load_state_dict(checkpoint['ae_model_state_dict'])
    model = model.to(device).train()
    
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.initial_learning_rate,
            weight_decay=config.train.weight_decay,
            betas=config.train.adam_betas
        )
    preset_processor = PresetProcessor(dataset, preset_idx_helper)

    # Parameter loss
    controls_criterion = SynthParamsLoss(
        preset_idx_helper,
        config.train.normalize_losses,
        cat_bce=config.train.params_cat_bceloss,
        cat_softmax=(not config.model.params_reg_softmax and not
                        config.train.params_cat_bceloss),
        cat_softmax_t=config.train.params_cat_softmax_temperature
    )

    # Logger
    config.model.name, config.model.run_name = 'debug', 'debug'
    logger = logger.RunLogger(logs_root_dir, config)
    scalars = dict()
    scalars['SpecMAE/Train'] = EpochMetric()
    scalars['Logprob/Train'] = EpochMetric()
    scalars['PGLoss/Train'] = EpochMetric()
    scalars['ParamLoss/Train'] = EpochMetric()

    current_step = 0
    disp = Display()
    disp.start()

    for epoch in tqdm(range(n_epochs), desc='epoch', position=0):
        dataloader_iter = iter(dataloader['test'])

        for i in tqdm(range(len(dataloader['test'])), desc='training batch', position=1, leave=False):
            sample = next(dataloader_iter)
            x_wav, x_in, v_in, sample_info = sample[0].numpy(), sample[1].to(device), sample[2].to(device), sample[3].numpy()
            v_out = model(x_in)
            full_preset_out, mean_log_probs = preset_processor(v_out)
            batch_size = full_preset_out.shape[0]

            num_workers = 16
            x_wav_split = np.array_split(x_wav, num_workers, axis=0)
            midi_pitch_split = np.array_split(sample_info[:, 1], num_workers, axis=0)
            midi_velocity_split = np.array_split(sample_info[:, 2], num_workers, axis=0)
            full_preset_out_split = np.array_split(full_preset_out, num_workers, axis=0)
            workers_data = [(x_wav_split[i], full_preset_out_split[i], midi_pitch_split[i], midi_velocity_split[i]) for i in range(num_workers)]

            with multiprocessing.Pool(num_workers) as p:
                spec_maes_split = p.map(measure_spec_mae_worker, workers_data)

            spec_maes = np.hstack(spec_maes_split)
            spec_maes = torch.FloatTensor(spec_maes).unsqueeze(1).to(v_out.device)

            # RL Spectrogram loss
            pg_loss = (spec_maes * mean_log_probs).mean()

            # Parameter loss
            cont_loss = controls_criterion(v_out, v_in)

            loss = 0.01 * pg_loss + cont_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            for _, s in scalars.items():
                s.on_new_epoch()

            scalars['SpecMAE/Train'].append(spec_maes.mean().item())
            scalars['Logprob/Train'].append(mean_log_probs.mean().item())
            scalars['PGLoss/Train'].append(pg_loss.item())
            scalars['ParamLoss/Train'].append(cont_loss.item())

            for k, s in scalars.items():
                logger.tensorboard.add_scalar(k, s.get(), current_step)
            
            current_step += 1

        logger.save_checkpoint(epoch, model, optimizer, optimizer)

    disp.stop()
    print('Finetuning process finished')