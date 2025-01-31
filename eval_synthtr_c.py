"""
Evaluation of trained models

TODO write doc
"""

import os
import os.path
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

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
from model.loss import PresetProcessor
from utils.audio import AudioRenderer, Spectrogram_Processor
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
    """
    Loads a model from given directory (and its associated dataset) and performs a full evaluation
    TODO describe output
    """
    root_path = Path(eval_config.logs_root_dir)
    t_start = datetime.now()

    # Reload model and train config
    config = OmegaConf.load(path_to_model_dir.joinpath('config.yaml'))
    
    # Eval file to be created
    # eval_path = path_to_model_dir.joinpath(eval_config.dataset[0], f'{eval_config.ckp_epoch:03d}epoch')
    eval_path = path_to_model_dir.joinpath(eval_config.dataset[0], f'{eval_config.ckp_epoch:03d}epoch')
    audio_path = eval_path.joinpath('audio')
    os.makedirs(audio_path, exist_ok=True)
    eval_pickle_file_path = get_eval_pickle_file_path(eval_path, eval_config.dataset[1])
    
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
    
    dexed_dataset = data.build.get_dataset('dexed', config)
    eval_dataset = data.build.get_dataset(eval_config.dataset[0], config)

    if eval_config.dataset[1] == 'all':
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
        )
    else:
        dataloader = data.build.get_split_dataloaders(config, eval_dataset)
        dataloader = dataloader[eval_config.dataset[1]]

    # Synth parameter index information for alignment   
    preset_idx_helper = dexed_dataset.preset_indexes_helper

    # Rebuild model from last saved checkpoint (default: if trained on GPU, would be loaded on GPU)
    device = torch.device(eval_config.device)
    checkpoint = logs.logger.get_model_checkpoint(root_path, config, eval_config.ckp_epoch, device)
    eval_model = SynthTR(preset_idx_helper, **config.model.encoder_kwargs)
    eval_model.load_state_dict(checkpoint['ae_model_state_dict'])
    eval_model = eval_model.to(device).eval()
    torch.set_grad_enabled(False)

    # 0) Structures and Criteria for evaluation metrics
    # Empty dicts (one dict per preset), eventually converted to a pandas dataframe
    eval_metrics = list()  # list of dicts
    preset_UIDs = list()
    synth_params_GT = list()
    eval_accuracies = list()
    eval_maes = list()
    # Parameters criteria
    controls_num_mse_criterion = model.loss.QuantizedNumericalParamsLoss(dexed_dataset.preset_indexes_helper,
                                                                         numerical_loss=nn.MSELoss(reduction='mean'))
    controls_num_mae_criterion = model.loss.QuantizedNumericalParamsLoss(dexed_dataset.preset_indexes_helper, reduce=False,
                                                                         numerical_loss=nn.L1Loss(reduction='mean'))
    controls_accuracy_criterion = model.loss.CategoricalParamsAccuracy(
        dexed_dataset.preset_indexes_helper,
        reduce=False,
        percentage_output=True
    )
    # Controls related to MIDI key and velocity (to compare single- and multi-channel spectrograms models)
    if dexed_dataset.synth_name.lower() == "dexed":
        dynamic_vst_controls_indexes = synth.dexed.Dexed.get_midi_key_related_param_indexes()
    else:
        raise NotImplementedError("")
    dynamic_controls_num_mae_crit = model.loss.QuantizedNumericalParamsLoss(
        dexed_dataset.preset_indexes_helper,
        numerical_loss=nn.L1Loss(reduction='mean'),
        limited_vst_params_indexes=dynamic_vst_controls_indexes
    )
    dynamic_controls_acc_crit = model.loss.CategoricalParamsAccuracy(
        dexed_dataset.preset_indexes_helper,
        reduce=True, 
        limited_vst_params_indexes=dynamic_vst_controls_indexes
    )

    preset_processor = PresetProcessor(dexed_dataset, preset_idx_helper, device)
    audio_renderer = AudioRenderer(
        dexed_dataset,
        config.model.write_sr,
        1,
        device
    )
    spec_processor = Spectrogram_Processor(
        config.train.pg_nfft,
        config.train.pg_hop,
        config.model.write_sr,
    ).to(device)

    # 1) Infer all preset parameters
    assert eval_config.minibatch_size == 1  # Required for per-preset metrics

    disp = Display()
    disp.start()

    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        x_wav, x_in, v_in, preset_UID = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3].item()

        with torch.no_grad():
            v_out = eval_model(x_in)
            full_preset_out, _ = preset_processor(v_out, deterministic=True)
            inferred_wav = audio_renderer.single_process_render(full_preset_out)
            inferred_wav = inferred_wav / torch.abs(inferred_wav + 1e-5).max(dim=1)[0]
            sc, log_mae, mfcc13_mae, mfcc40_mae = spec_processor.calculate_metrics(x_wav, inferred_wav)

        filename_gt = audio_path.joinpath(f'{preset_UID}_gt.wav')
        filename_inferred = audio_path.joinpath(f'{preset_UID}.wav')
        soundfile.write(filename_gt, x_wav.cpu().numpy()[0], eval_config.sampling_rate)
        soundfile.write(filename_inferred, inferred_wav.cpu().numpy()[0], eval_config.sampling_rate)

        eval_metrics.append(dict())
        eval_metrics[-1]['preset_UID'] = preset_UID
        eval_metrics[-1]['spec_sc'] = sc.item()
        eval_metrics[-1]['spec_mae'] = log_mae.item()
        eval_metrics[-1]['mfcc13_mae'] = mfcc13_mae.item()
        eval_metrics[-1]['mfcc40_mae'] = mfcc40_mae.item()
        preset_UIDs.append(preset_UID)

        if eval_config.dataset[0] == 'dexed':
            # Metrics
            accuracies = controls_accuracy_criterion(v_out, v_in)
            acc_value = np.asarray([v for _, v in accuracies.items()]).mean()
            maes, mae_value = controls_num_mae_criterion(v_out, v_in)

            # Parameters inference metrics
            eval_metrics[-1]['num_controls_MSEQ'] = controls_num_mse_criterion(v_out, v_in).item()
            eval_metrics[-1]['num_controls_MAEQ'] = mae_value.item()
            eval_metrics[-1]['cat_controls_acc'] = acc_value
            eval_metrics[-1]['num_dyn_cont_MAEQ'] = dynamic_controls_num_mae_crit(v_out, v_in).item()
            eval_metrics[-1]['cat_dyn_cont_acc'] = dynamic_controls_acc_crit(v_out, v_in)
            eval_accuracies.append(accuracies)
            eval_maes.append(maes)
            in_presets_instance = data.preset.DexedPresetsParams(learnable_presets=v_in, dataset=dexed_dataset)
            synth_params_GT.append(in_presets_instance.get_full()[0, :].cpu().numpy())

    disp.stop()        

    # Numpy matrix of preset values. Reconstructed spectrograms are not stored
    preset_UIDs = np.asarray(preset_UIDs)

    if eval_config.dataset[0] == 'dexed':
        acc_df = pd.DataFrame(eval_accuracies, index=preset_UIDs)
        mae_df = pd.DataFrame(eval_maes, index=preset_UIDs)
        acc_df.to_pickle(eval_path.joinpath('cat_params_acc.pickle'))
        mae_df.to_pickle(eval_path.joinpath('num_params_mae.pickle'))

    # 3) Concatenate results into a dataframe
    eval_df = pd.DataFrame(eval_metrics)

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


if __name__ == "__main__":
    import evalconfig
    eval_config = evalconfig.eval

    print("Starting models evaluation using configuration from evalconfig.py, using '{}' dataset"
          .format(eval_config.dataset[1]))
    evaluate_all_models(eval_config)
