"""
Evaluation of trained models

TODO write doc
"""

import os
import os.path
import psutil
from pathlib import Path
from datetime import datetime
from typing import Sequence
import multiprocessing
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
from scipy.signal import resample

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import data.build
import data.preset
import data.abstractbasedataset
import logs.logger
import logs.metrics
import model.build
from model.encoder import SynthTR
import model.loss
import utils.audio
import utils.config
import synth.dexed

import soundfile
from pyvirtualdisplay import Display
from tqdm import tqdm

def evaluate_all_models(eval_config: utils.config.EvalConfig):
    """
    Evaluates all models whose names can be found in the given text file.

    :param eval_config:
    :return: TODO
    """
    # Retrieve the list of models to be evaluated
    root_path = Path(eval_config.logs_root_dir)
    models_dirs_path = list()
    for model_name in eval_config.models_names:
        models_dirs_path.append(root_path.joinpath(model_name))
    print("{} models found for evaluation".format(len(models_dirs_path)))

    # Single-model evaluation
    for i, model_dir_path in enumerate(models_dirs_path):
        print("================================================================")
        print("===================== Evaluation of model {}/{} ==================".format(i+1, len(models_dirs_path)))
        evaluate_model(model_dir_path, eval_config)


def get_eval_pickle_file_path(eval_path: Path, dataset_type: str, force_multi_note=False):
    return eval_path.joinpath('eval_{}{}.dataframe.pickle'
                              .format(dataset_type, ('__MULTI_NOTE__' if force_multi_note else '')))


def evaluate_model(path_to_model_dir: Path, eval_config: utils.config.EvalConfig):
    root_path = Path(eval_config.logs_root_dir)
    t_start = datetime.now()
    config = OmegaConf.load(path_to_model_dir.joinpath('config.yaml'))
    eval_path = path_to_model_dir.joinpath('surge_add2', f'{eval_config.ckp_epoch:03d}epoch')
    audio_path = eval_path.joinpath('audio')
    spec_path = eval_path.joinpath('spectrogram')
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(spec_path, exist_ok=True)  
    eval_pickle_file_path = get_eval_pickle_file_path(eval_path, eval_config.dataset[1])
    
    if os.path.exists(eval_pickle_file_path):
        eval_df = pd.read_pickle(eval_pickle_file_path)
    
    dexed_dataset = data.build.get_dataset('dexed', config)
    eval_dataset = data.build.get_dataset(eval_config.dataset[0], config)

    # Synth parameter index information for alignment   
    preset_idx_helper = dexed_dataset.preset_indexes_helper

    # Rebuild model from last saved checkpoint (default: if trained on GPU, would be loaded on GPU)
    device = torch.device(eval_config.device)
    checkpoint = logs.logger.get_model_checkpoint(root_path, config, eval_config.ckp_epoch, device)
    eval_model = SynthTR(preset_idx_helper, **config.model.encoder_kwargs)
    eval_model.load_state_dict(checkpoint['ae_model_state_dict'])
    eval_model = eval_model.to(device).eval()
    torch.set_grad_enabled(False)

    eval_midi_notes = ((60, 85), )
    synth_params_inferred = []
    preset_UIDs = [584, 170,  73, 575, 552, 255, 619, 261, 438, 351, 303,   0, 344,
            609, 437, 302, 378, 308, 160, 101, 618, 193, 518, 628, 125,  65,
            460, 246, 591, 615, 583, 123, 613, 207, 368, 573, 234, 534,  45,
            191, 576, 601, 139, 133, 269, 586, 563, 554,  19, 105,  89, 590,
            561,  67, 228,  86, 284, 382, 183, 214,  57,  98, 436, 622,  94,
            566, 621, 556, 145, 154, 429, 274,  88, 158, 553, 581, 558,  70,
            120, 571, 182, 327, 100, 550, 328, 545, 199, 577, 236, 441,  24,
             21, 432,  93,  15, 270,  80, 263, 129, 398]
    
    for preset_UID in tqdm(preset_UIDs):
        x_in = eval_dataset.get_spec_file(preset_UID, 60, 85).to(device)
        v_out = eval_model(x_in.unsqueeze(1))
        out_presets_instance = data.preset.DexedPresetsParams(learnable_presets=v_out, dataset=dexed_dataset)
        synth_params_inferred.append(out_presets_instance.get_full()[0, :].cpu().numpy())

    synth_params_inferred = np.array(synth_params_inferred)
    preset_UIDs = np.array(preset_UIDs)
    num_workers = int(np.round(os.cpu_count() * eval_config.multiprocess_cores_ratio))
    preset_UIDs_split = np.array_split(preset_UIDs, num_workers, axis=0)
    synth_params_inferred_split = np.array_split(synth_params_inferred, num_workers, axis=0)
    workers_data = [(dexed_dataset, eval_dataset, eval_midi_notes, audio_path, spec_path, eval_config.sampling_rate,
                     preset_UIDs_split[i], synth_params_inferred_split[i], i)
                    for i in range(num_workers)]
    
    disp = Display()
    disp.start()

    # Multi-processing is absolutely necessary
    with multiprocessing.Pool(num_workers) as p:
        audio_errors_split = p.map(_measure_audio_errors_worker, workers_data)

    disp.stop()
    audio_errors = dict()

    for error_name in audio_errors_split[0]:
        audio_errors[error_name] = np.hstack([audio_errors_split[i][error_name]
                                              for i in range(len(audio_errors_split))])

    for i, preset_UID in enumerate(preset_UIDs):
        for key, value in audio_errors.items():
            eval_df.loc[preset_UID, key] = value[i]

    eval_df.to_pickle(eval_pickle_file_path)

    if eval_config.verbosity >= 1:
        print("Finished evaluation ({}) in {:.1f}s".format(eval_pickle_file_path,
                                                           (datetime.now() - t_start).total_seconds()))


def _measure_audio_errors_worker(worker_args):
    pid = os.getpid()
    cpus = list(range(psutil.cpu_count()))
    os.sched_setaffinity(pid, cpus)
    return _measure_audio_errors(*worker_args)


def _measure_audio_errors(dexed_dataset, eval_dataset, midi_notes, audio_path, spec_path,
                          sampling_rate: int, preset_UIDs: Sequence, synth_params_inferred: np.ndarray, i):
    # Dict of per-UID errors (if multiple notes: note-averaged values)
    errors = {'spec_mae': list(), 'spec_sc': list(), 'mfcc13_mae': list(), 'mfcc40_mae': list()}

    for idx, preset_UID in tqdm(enumerate(preset_UIDs), position=i, desc=f'Process {i}', leave=True, total=len(preset_UIDs)):
        mae, sc, mfcc13_mae, mfcc40_mae = list(), list(), list(), list()  # Per-note errors (might be 1-element lists)   

        for midi_pitch, midi_velocity in midi_notes:  # Possible multi-note evaluation
            x_wav_original = eval_dataset.get_wav_file(preset_UID, midi_pitch, midi_velocity)  # Pre-rendered file
            x_wav_original = x_wav_original / np.abs(x_wav_original + 1e-5).max()
            with utils.audio.suppress_output():
                x_wav_inferred, Fs = dexed_dataset._render_audio(synth_params_inferred[idx], midi_pitch, midi_velocity)
                num_samples = int(len(x_wav_inferred) * sampling_rate / Fs)
                x_wav_inferred = resample(x_wav_inferred, num_samples)
                x_wav_inferred = x_wav_inferred / np.abs(x_wav_inferred + 1e-5).max()

            # Save .wav files
            filename_gt = os.path.join(audio_path, f'{preset_UID}_p{midi_pitch}_v{midi_velocity}_gt.wav')
            filename_inferred = os.path.join(audio_path, f'{preset_UID}_p{midi_pitch}_v{midi_velocity}.wav')
            soundfile.write(filename_gt, x_wav_original, sampling_rate, subtype='FLOAT')
            soundfile.write(filename_inferred, x_wav_inferred, sampling_rate, subtype='FLOAT')
            
            # Save log spectrogram figures
            similarity_eval = utils.audio.SimilarityEvaluator((x_wav_original, x_wav_inferred))
            _mae, log_stft = similarity_eval.get_mae_log_stft(return_spectrograms=True)
            fig, _ = similarity_eval.display_stft(log_stft)
            filename_spec = spec_path.joinpath(f'{preset_UID}_p{midi_pitch}_v{midi_velocity}.png')
            fig.savefig(filename_spec)
            plt.close(fig)
            
            mae.append(_mae)
            sc.append(similarity_eval.get_spectral_convergence(return_spectrograms=False))
            mfcc13_mae.append(similarity_eval.get_mae_mfcc(return_mfccs=False, n_mfcc=13))
            mfcc40_mae.append(similarity_eval.get_mae_mfcc(return_mfccs=False, n_mfcc=40))

        # Average errors over all played MIDI notes
        errors['spec_mae'].append(np.mean(mae))
        errors['spec_sc'].append(np.mean(sc))
        errors['mfcc13_mae'].append(np.mean(mfcc13_mae))
        errors['mfcc40_mae'].append(np.mean(mfcc40_mae))

    for error_name in errors:
        errors[error_name] = np.asarray(errors[error_name])

    return errors


if __name__ == "__main__":
    import evalconfig
    eval_config = evalconfig.eval

    print("Starting models evaluation using configuration from evalconfig.py, using '{}' dataset"
          .format(eval_config.dataset[1]))
    evaluate_all_models(eval_config)
