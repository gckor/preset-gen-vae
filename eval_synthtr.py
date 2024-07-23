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


def get_eval_pickle_file_path(path_to_model_dir: Path, dataset_type: str, force_multi_note=False):
    return path_to_model_dir.joinpath('eval_{}{}.dataframe.pickle'
                                      .format(dataset_type, ('__MULTI_NOTE__' if force_multi_note else '')))


def evaluate_model(path_to_model_dir: Path, eval_config: utils.config.EvalConfig):
    """
    Loads a model from given directory (and its associated dataset) and performs a full evaluation
    TODO describe output
    """
    root_path = Path(eval_config.logs_root_dir)
    t_start = datetime.now()

    # Reload model and train config
    config = OmegaConf.load(path_to_model_dir.joinpath('config.yaml'))
    
    # Eval file to be created
    eval_pickle_file_path = get_eval_pickle_file_path(path_to_model_dir, eval_config.dataset)
    
    # Return now if eval already exists, and should not be overridden
    if os.path.exists(eval_pickle_file_path):
        if not eval_config.override_previous_eval:
            if eval_config.verbosity >= 1:
                print("Evaluation file '{}' already exists. Skipping (override_previous_eval={})"
                      .format(eval_pickle_file_path, eval_config.override_previous_eval))
            return

    # Reload the corresponding dataset, dataloaders and models
    config.verbosity = 1
    config.train.minibatch_size = eval_config.minibatch_size  # Will setup dataloaders as requested
    
    dataset = data.build.get_dataset(config)
    dataloader = data.build.get_split_dataloaders(config, dataset)

    # Synth parameter index information for alignment   
    preset_idx_helper = dataset.preset_indexes_helper

    # Rebuild model from last saved checkpoint (default: if trained on GPU, would be loaded on GPU)
    device = torch.device(eval_config.device)
    checkpoint = logs.logger.get_model_last_checkpoint(root_path, config, device=device)
    eval_model = SynthTR(preset_idx_helper, **config.model.encoder_kwargs)
    eval_model.load_state_dict(checkpoint['ae_model_state_dict'])
    eval_model = eval_model.to(device).eval()
    torch.set_grad_enabled(False)
        
    eval_midi_notes = ((60, 85), )

    if eval_config.verbosity >= 1:
        print("Evaluation will be performed on {} MIDI note(s): {}".format(len(eval_midi_notes), eval_midi_notes))


    # 0) Structures and Criteria for evaluation metrics
    # Empty dicts (one dict per preset), eventually converted to a pandas dataframe
    eval_metrics = list()  # list of dicts
    preset_UIDs = list()
    synth_params_GT = list()
    synth_params_inferred = list()
    eval_accuracies = list()
    eval_maes = list()
    # Parameters criteria
    controls_num_mse_criterion = model.loss.QuantizedNumericalParamsLoss(dataset.preset_indexes_helper,
                                                                         numerical_loss=nn.MSELoss(reduction='mean'))
    controls_num_mae_criterion = model.loss.QuantizedNumericalParamsLoss(dataset.preset_indexes_helper, reduce=False,
                                                                         numerical_loss=nn.L1Loss(reduction='mean'))
    controls_accuracy_criterion = model.loss.CategoricalParamsAccuracy(
        dataset.preset_indexes_helper,
        reduce=False,
        percentage_output=True
    )
    # Controls related to MIDI key and velocity (to compare single- and multi-channel spectrograms models)
    if dataset.synth_name.lower() == "dexed":
        dynamic_vst_controls_indexes = synth.dexed.Dexed.get_midi_key_related_param_indexes()
    else:
        raise NotImplementedError("")
    dynamic_controls_num_mae_crit = model.loss.QuantizedNumericalParamsLoss(
        dataset.preset_indexes_helper,
        numerical_loss=nn.L1Loss(reduction='mean'),
        limited_vst_params_indexes=dynamic_vst_controls_indexes
    )
    dynamic_controls_acc_crit = model.loss.CategoricalParamsAccuracy(
        dataset.preset_indexes_helper,
        reduce=True, 
        limited_vst_params_indexes=dynamic_vst_controls_indexes
    )

    # 1) Infer all preset parameters
    assert eval_config.minibatch_size == 1  # Required for per-preset metrics

    for i, sample in tqdm(enumerate(dataloader[eval_config.dataset]), total=len(dataloader[eval_config.dataset])):
        x_in, v_in, sample_info = sample[1].to(device), sample[2].to(device), sample[3].to(device)
        v_out = eval_model(x_in)

        # Metrics
        accuracies = controls_accuracy_criterion(v_out, v_in)
        acc_value = np.asarray([v for _, v in accuracies.items()]).mean()
        maes, mae_value = controls_num_mae_criterion(v_out, v_in)

        # Parameters inference metrics
        preset_UIDs.append(sample_info[0, 0].item())
        eval_metrics.append(dict())
        eval_metrics[-1]['preset_UID'] = sample_info[0, 0].item()
        eval_metrics[-1]['num_controls_MSEQ'] = controls_num_mse_criterion(v_out, v_in).item()
        eval_metrics[-1]['num_controls_MAEQ'] = mae_value.item()
        eval_metrics[-1]['cat_controls_acc'] = acc_value
        eval_metrics[-1]['num_dyn_cont_MAEQ'] = dynamic_controls_num_mae_crit(v_out, v_in).item()
        eval_metrics[-1]['cat_dyn_cont_acc'] = dynamic_controls_acc_crit(v_out, v_in)
        # Compute corresponding flexible presets instances
        in_presets_instance = data.preset.DexedPresetsParams(learnable_presets=v_in, dataset=dataset)
        out_presets_instance = data.preset.DexedPresetsParams(learnable_presets=v_out, dataset=dataset)
        # VST-compatible full presets (1-element batch of presets)
        synth_params_GT.append(in_presets_instance.get_full()[0, :].cpu().numpy())
        synth_params_inferred.append(out_presets_instance.get_full()[0, :].cpu().numpy())
        eval_accuracies.append(accuracies)
        eval_maes.append(maes)

    # Numpy matrix of preset values. Reconstructed spectrograms are not stored
    synth_params_GT, synth_params_inferred = np.asarray(synth_params_GT), np.asarray(synth_params_inferred)
    preset_UIDs = np.asarray(preset_UIDs)
    acc_df = pd.DataFrame(eval_accuracies, index=preset_UIDs)
    mae_df = pd.DataFrame(eval_maes, index=preset_UIDs)
    acc_df.to_pickle(path_to_model_dir.joinpath('cat_params_acc.pickle'))
    mae_df.to_pickle(path_to_model_dir.joinpath('num_params_mae.pickle'))


    # 2) Evaluate audio from inferred synth parameters
    audio_path = path_to_model_dir.joinpath('audio')
    spec_path = path_to_model_dir.joinpath('spectrogram')
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(spec_path, exist_ok=True)  

    num_workers = int(np.round(os.cpu_count() * eval_config.multiprocess_cores_ratio))
    preset_UIDs_split = np.array_split(preset_UIDs, num_workers, axis=0)
    synth_params_GT_split = np.array_split(synth_params_GT, num_workers, axis=0)
    synth_params_inferred_split = np.array_split(synth_params_inferred, num_workers, axis=0)
    workers_data = [(dataset, eval_midi_notes, audio_path, spec_path, eval_config.sampling_rate,
                     preset_UIDs_split[i], synth_params_inferred_split[i], i)
                    for i in range(num_workers)]
    # Multi-processing is absolutely necessary
    with multiprocessing.Pool(num_workers) as p:
        audio_errors_split = p.map(_measure_audio_errors_worker, workers_data)
    audio_errors = dict()
    for error_name in audio_errors_split[0]:
        audio_errors[error_name] = np.hstack([audio_errors_split[i][error_name]
                                              for i in range(len(audio_errors_split))])


    # 3) Concatenate results into a dataframe
    eval_df = pd.DataFrame(eval_metrics)
    # append audio errors
    for error_name, err in audio_errors.items():
        eval_df[error_name] = err
    # multi-note case: average results with the same preset UID (Python set prevents duplicates)
    # This also sorts the dataframe presets UIDs and will done for all evaluations (sub-optimal but small data structs)
    preset_UIDs_no_duplicates = list(set(eval_df['preset_UID'].values))
    preset_UIDs_no_duplicates.sort()
    eval_metrics_no_duplicates = list()  # Will eventually be a dataframe
    # We use the original list to build a new dataframe
    for preset_UID in preset_UIDs_no_duplicates:
        eval_metrics_no_duplicates.append(dict())
        eval_sub_df = eval_df.loc[eval_df['preset_UID'] == preset_UID]
        eval_metrics_no_duplicates[-1]['preset_UID'] = preset_UID
        for col in eval_sub_df:  # Average all metrics
            if col != 'preset_UID':
                eval_metrics_no_duplicates[-1][col] = eval_sub_df[col].mean()
    eval_df = pd.DataFrame(eval_metrics_no_duplicates)


    # 4) Write eval files
    eval_df.to_pickle(eval_pickle_file_path)

    if eval_config.verbosity >= 1:
        print("Finished evaluation ({}) in {:.1f}s".format(eval_pickle_file_path,
                                                           (datetime.now() - t_start).total_seconds()))


def _measure_audio_errors_worker(worker_args):
    pid = os.getpid()
    cpus = list(range(psutil.cpu_count()))
    os.sched_setaffinity(pid, cpus)
    return _measure_audio_errors(*worker_args)


def _measure_audio_errors(dataset: data.abstractbasedataset.PresetDataset, midi_notes, audio_path, spec_path,
                          sampling_rate: int, preset_UIDs: Sequence, synth_params_inferred: np.ndarray, i):
    # Dict of per-UID errors (if multiple notes: note-averaged values)
    errors = {'spec_mae': list(), 'spec_sc': list(), 'mfcc13_mae': list(), 'mfcc40_mae': list()}
    disp = Display()
    disp.start()

    for idx, preset_UID in tqdm(enumerate(preset_UIDs), position=i, desc=f'Process {i}', leave=True, total=len(preset_UIDs)):
        mae, sc, mfcc13_mae, mfcc40_mae = list(), list(), list(), list()  # Per-note errors (might be 1-element lists)   
        for midi_pitch, midi_velocity in midi_notes:  # Possible multi-note evaluation
            x_wav_original, _ = dataset.get_wav_file(preset_UID, midi_pitch, midi_velocity)  # Pre-rendered file
            with utils.audio.suppress_output():
                x_wav_inferred, _ = dataset._render_audio(synth_params_inferred[idx], midi_pitch, midi_velocity)

            # Save .wav files
            filename_gt = os.path.join(audio_path, f'{preset_UID}_p{midi_pitch}_v{midi_velocity}_gt.wav')
            filename_inferred = os.path.join(audio_path, f'{preset_UID}_p{midi_pitch}_v{midi_velocity}.wav')
            soundfile.write(filename_gt, x_wav_original, sampling_rate)
            soundfile.write(filename_inferred, x_wav_inferred, sampling_rate)
            
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

    disp.stop()

    for error_name in errors:
        errors[error_name] = np.asarray(errors[error_name])

    return errors


if __name__ == "__main__":
    import evalconfig
    eval_config = evalconfig.eval

    print("Starting models evaluation using configuration from evalconfig.py, using '{}' dataset"
          .format(eval_config.dataset))
    evaluate_all_models(eval_config)
