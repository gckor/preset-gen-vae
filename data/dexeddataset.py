"""
Implementation of the DexedDataset, based on the PresetBased abstract class.

Wav files, spectrograms statistics, etc., can be re-generated by running this script as main.
See end of file.
"""
import os
import pathlib
import json
import h5py
from typing import Optional, Iterable
import multiprocessing
from datetime import datetime
from typing import List, Tuple, Any

import torch
import torch.utils
import soundfile
import numpy as np

from synth import dexed
from data import abstractbasedataset  # 'from .' raises ImportError when run from PyCharm as __main__
from data.preset import DexedPresetsParams, PresetIndexesHelper

# Global lock... Should be the same for all forked Unix processes
#dexed_vst_lock = Lock()  # Unused - pre-rendered audio (at the moment)



class DexedDataset(abstractbasedataset.PresetDataset):
    def __init__(
        self,
        note_duration: List[float],
        n_fft: int,
        fft_hop: int,
        midi_notes: Tuple[Tuple[int, int]] = ((60, 100),),
        multichannel_stacked_spectrograms: bool = False,
        n_mel_bins: int = -1,
        normalize_audio: bool = False,
        spectrogram_min_dB: float = -120.0,
        spectrogram_normalization: str = 'min_max',
        algos: Optional[List[int]] = None,
        operators: Optional[List[int]] = None,
        vst_params_learned_as_categorical: Optional[str] = None,
        restrict_to_labels: Optional[List[str]] = None,
        constant_filter_and_tune_params: bool = True,
        check_constrains_consistency: bool = True,
        dataset_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        num_param_bins: int = -1,
        sample_rate: int = 22050,
    ):
        """
        Allows access to Dexed preset values and names, and generates spectrograms and corresponding
        parameters values. Can manage a reduced number of synth parameters (using default values for non-
        learnable params). Only Dexed-specific ctor args are described - see base PresetDataset class.

        It uses both the SQLite DB (through dexed.PresetDatabase) and the pre-written files extracted from
        the DB (see dexed.PresetDatabase.write_all_presets_to_files(...)).

        :param algos: List. Can be used to limit the DX7 algorithms included in this dataset. Set to None
            to use all available algorithms
        :param operators: List of ints, or None. Enables the specified operators only, or all of them if None.
        :param vst_params_learned_as_categorical: 'all_num' to learn all vst params as numerical, 'vst_cat'
            to learn vst cat params as categorical, or 'all<=x' to learn all vst params (including numerical) with
            cardinality <= xxx (e.g. 8 or 32) as categorical
        :param restrict_to_labels: List of strings. If not None, presets of this dataset will be selected such
            that they are tagged with at least one of the given labels.
        :param constant_filter_and_tune_params: if True, the main filter and the main tune settings are default
        :param check_constrains_consistency: Set to False when this dataset instance is used to pre-render
            audio files
        """
        super().__init__(
            note_duration,
            n_fft,
            fft_hop,
            midi_notes,
            multichannel_stacked_spectrograms,
            n_mel_bins,
            normalize_audio,
            spectrogram_min_dB,
            spectrogram_normalization,
            dataset_dir,
            dataset_name,
            sample_rate
        )
        self.constant_filter_and_tune_params = constant_filter_and_tune_params
        if check_constrains_consistency:  # pre-rendered audio consistency
            self.check_audio_render_constraints_file()
        self.algos = algos if algos is not None else []
        self._operators = operators if operators is not None else [1, 2, 3, 4, 5, 6]
        self.restrict_to_labels = restrict_to_labels

        # Full SQLite DB read and temp storage in np arrays
        dexed_db = dexed.PresetDatabase()
        self._total_nb_presets = dexed_db.presets_mat.shape[0]
        self._total_nb_params = dexed_db.presets_mat.shape[1]
        self._param_names = dexed_db.get_param_names()

        # Constraints on parameters, learnable VST parameters
        self.learnable_params_idx = list(range(0, dexed_db.presets_mat.shape[1]))
        
        if self.constant_filter_and_tune_params:  # (see dexed_db_explore.ipynb)
            for vst_idx in [0, 1, 2, 3, 13]:
                self.learnable_params_idx.remove(vst_idx)
        
        for i_op in range(6):  # Search for disabled operators
            if (i_op + 1) not in self._operators:  # If disabled: we remove all corresponding learnable params
                for vst_idx in range(21):  # Don't remove the 22nd param (OP on/off selector) yet
                    self.learnable_params_idx.remove(23 + 22 * i_op + vst_idx)  # idx 23 is the first param of op 1
        
        # Oscillators can be enabled or disabled, but OP SWITCHES are never learnable parameters
        for col in [44, 66, 88, 110, 132, 154]:
            self.learnable_params_idx.remove(col)

        # Valid presets - UIDs of presets, and not their database row index
        # Select valid presets by algorithm
        if len(self.algos) == 0:  # All presets are valid
            self.valid_preset_UIDs = dexed_db.all_presets_df["index_preset"].values
        else:
            if len(self.algos) == 1:
                self.learnable_params_idx.remove(4)  # Algo parameter column idx
            valid_presets_row_indexes = dexed_db.get_preset_indexes_for_algorithms(self.algos)
            self.valid_preset_UIDs = dexed_db.all_presets_df.iloc[valid_presets_row_indexes]['index_preset'].values

        # Select valid presets by label. We build a list of list-indexes to remove
        if self.restrict_to_labels is not None:
            self.valid_preset_UIDs = [uid for uid in self.valid_preset_UIDs
                                      if any([self.is_label_included(l) for l in self.get_labels_name(uid)])]
        
        # DB class deleted (we need a low memory usage for multi-process dataloaders)
        del dexed_db
        
        # Parameters constraints, cardinality, indexes management, ...
        # Param cardinalities are stored - Dexed cardinality involves a short search which can be avoided
        # This cardinality is the LEARNING REPRESENTATION cardinality - will be used for categorical representations
        self._params_cardinality = np.asarray([dexed.Dexed.get_param_cardinality(idx, num_param_bins)
                                               for idx in range(self.total_nb_params)])
        self._params_default_values = dict()

        # Algo cardinality is manually set. We consider an algo-limited DX7 to be a new synth
        if len(self.algos) > 0:  # len 0 means all algorithms are used
            self._params_cardinality[4] = len(self.algos)
        if len(self.algos) == 1:  # 1 algo: constrained constant param
            self._params_default_values[4] = (self.algos[0] - 1) / 31.0

        # cardinality 1 for constrained parameters (operators are always constrained)
        self._params_cardinality[[44, 66, 88, 110, 132, 154]] = np.ones((6,), dtype=np.int)
       
        for op_i, op_switch_idx in enumerate([44, 66, 88, 110, 132, 154]):
            self._params_default_values[op_switch_idx] = 1.0 if ((op_i + 1) in self._operators) else 0.0
       
        if self.constant_filter_and_tune_params:
            self._params_cardinality[[0, 1, 2, 3, 13]] = np.ones((5,), dtype=np.int)
            self._params_default_values[0] = 1.0
            self._params_default_values[1] = 0.0
            self._params_default_values[2] = 1.0
            self._params_default_values[3] = 0.5
            self._params_default_values[13] = 0.5

        # None / Numerical / Categorical learnable status array
        self._vst_param_learnable_model = list()
        num_vst_learned_as_cat_cardinal_threshold = None
        
        if vst_params_learned_as_categorical is not None:
            if vst_params_learned_as_categorical.startswith('all<='):
                num_vst_learned_as_cat_cardinal_threshold = int(vst_params_learned_as_categorical.replace('all<=', ''))
            else:
                assert vst_params_learned_as_categorical == 'vst_cat'

        # We go through all VST params indexes
        for vst_idx in range(self.total_nb_params):
            if vst_idx not in self.learnable_params_idx:
                self._vst_param_learnable_model.append(None)
            else:
                if vst_params_learned_as_categorical is None:  # Default: forced numerical only
                    self._vst_param_learnable_model.append('num')
                else:  # Mixed representations: is the VST param numerical?
                    if vst_idx in dexed.Dexed.get_numerical_params_indexes():
                        if num_vst_learned_as_cat_cardinal_threshold is None:  # If no threshold: learned as numerical
                            self._vst_param_learnable_model.append('num')
                        # If a non-continuous param has a small enough cardinality: might be learned as categorical
                        elif 1 < self._params_cardinality[vst_idx] <= num_vst_learned_as_cat_cardinal_threshold:
                            self._vst_param_learnable_model.append('cat')
                        else:
                            self._vst_param_learnable_model.append('num')
                    # If categorical VST param: must be learned as cat (at this point)
                    elif vst_idx in dexed.Dexed.get_categorical_params_indexes():
                        self._vst_param_learnable_model.append('cat')
                    else:
                        raise ValueError("VST param idx={} is neither numerical nor categorical".format(vst_idx))
        # Final initializations
        self._preset_idx_helper = PresetIndexesHelper(self)
        self._load_spectrogram_stats()  # Must be called after super() ctor

    @property
    def synth_name(self):
        return "Dexed"

    def __str__(self):
        return "{}. Restricted to labels: {}. Enabled algorithms: {}. Enabled operators: {}"\
            .format(super().__str__(), self.restrict_to_labels,
                    ('all' if len(self.algos) == 0 else self.algos), self._operators)

    @property
    def total_nb_presets(self):
        return self._total_nb_presets

    @property
    def vst_param_learnable_model(self):
        return self._vst_param_learnable_model

    @property
    def numerical_vst_params(self):
        return dexed.Dexed.get_numerical_params_indexes()

    @property
    def categorical_vst_params(self):
        return dexed.Dexed.get_categorical_params_indexes()

    @property
    def params_default_values(self):
        return self._params_default_values

    @property
    def total_nb_params(self):
        return self._total_nb_params

    @property
    def preset_indexes_helper(self):
        return self._preset_idx_helper

    @property
    def preset_param_names(self):
        return self._param_names

    def get_preset_param_cardinality(self, idx, learnable_representation=True):
        if idx == 4 and learnable_representation is False:
            return 32  # Algorithm is always an annoying special case... could be improved
        return self._params_cardinality[idx]

    def get_full_preset_params(self, preset_UID):
        raw_full_preset = dexed.PresetDatabase.get_preset_params_values_from_file(preset_UID)
        return DexedPresetsParams(full_presets=torch.unsqueeze(torch.tensor(raw_full_preset, dtype=torch.float32), 0),
                                  dataset=self)

    def is_label_included(self, label):
        """ Returns True if the label belongs to the restricted labels list. """
        if self.restrict_to_labels is None:
            return True
        else:
            return any([label == l_ for l_ in self.restrict_to_labels])

    def get_labels_tensor(self, preset_UID):  # TODO
        """ Returns a tensor of torch.int8 zeros and ones - each value is 1 if the preset is tagged with the
        corresponding label """
        return torch.tensor([1], dtype=torch.int8)  # 'NoLabel' is the only default label

    def get_labels_name(self, preset_UID):
        return dexed.PresetDatabase.get_preset_labels_from_file(preset_UID)

    @property
    def available_labels_names(self):
        """ Returns a tuple of string descriptions of labels. """
        return dexed.PresetDatabase.get_available_labels()

    def _render_audio(self, preset_params: Iterable, midi_note, midi_velocity):
        # reload the VST to prevent hanging notes/sounds
        dexed_renderer = dexed.Dexed(midi_note_duration_s=self.note_duration[0],
                                     render_duration_s=self.note_duration[0] + self.note_duration[1],
                                     sample_rate=self.sample_rate)
        dexed_renderer.assign_preset(dexed.PresetDatabase.get_params_in_plugin_format(preset_params))
        x_wav = dexed_renderer.render_note(midi_note, midi_velocity, normalize=self.normalize_audio)
        return x_wav, dexed_renderer.Fs

    def _get_spectrogram_stats_file_stem(self):
        return super()._get_spectrogram_stats_file_stem() + self._operators_suffix

    @property
    def _operators_suffix(self):
        """ Returns a suffix (to be used in files names) that describes enabled DX7 operators, as configured
        in this dataset's constructor. Return an empty string if all operators are used. """
        ops_suffix = ''
        if self._operators != [1, 2, 3, 4, 5, 6]:
            ops_suffix = '_op' + ''.join(['{}'.format(op) for op in self._operators])
        return ops_suffix

    def generate_data(self, idx):
        midi_pitch, midi_velocity = self.midi_notes[0]

        preset_UID = self.valid_preset_UIDs[idx]
        preset_params = self.get_full_preset_params(preset_UID)

        return (
            preset_params.get_learnable().squeeze().numpy(),
            np.array([preset_UID, midi_pitch, midi_velocity], dtype=np.int32),
            self.get_labels_tensor(preset_UID).numpy(),
        )

    def get_aug_specs(self, preset_UIDs, augmented_pitch=[40, 50, 70]):
        midi_pitch = np.random.choice(augmented_pitch)
        midi_velocity = 85
        aug_specs = []
        
        for preset_UID in preset_UIDs:
            spectrogram = self.get_spec_file(preset_UID, midi_pitch, midi_velocity)
            aug_specs.append(spectrogram)

        return torch.stack(aug_specs).unsqueeze(1)
    
    def get_data_from_file(self, preset_UID, midi_pitch, midi_velocity):
        with h5py.File(self.dataset_dir.joinpath(f'{self.dataset_name}.h5py'), 'r') as f:
            data = f[f'{preset_UID:06d}_{midi_pitch}_{midi_velocity}']
            synth_param = data['synth_param'][:]
            sample_info = data['sample_info'][:]
            label = data['label'][:]
        waveform = self.get_wav_file(preset_UID, midi_pitch, midi_velocity)[0].astype(np.float32)
        spectrogram = self.get_spec_file(preset_UID, midi_pitch, midi_velocity).unsqueeze(0)
        return waveform, spectrogram, synth_param, sample_info, label
    
    def get_spec_file_path(self, preset_UID, midi_note, midi_velocity):
        """ Returns the path of a spectrogram (from dexed_presets folder). Operators"""
        filename = "preset{:06d}_midi{:03d}vel{:03d}{}.pt".format(preset_UID, midi_note, midi_velocity,
                                                                   self._operators_suffix)
        return self.spec_files_dir.joinpath(filename)

    def get_spec_file(self, preset_UID, midi_note, midi_velocity):
        file_path = self.get_spec_file_path(preset_UID, midi_note, midi_velocity)
        try:
            spectrogram = torch.load(file_path)

            if self.spectrogram_normalization == 'min_max':  # result in [-1, 1]
                spectrogram = -1.0 + (spectrogram - self.spec_stats['min'])\
                            / ((self.spec_stats['max'] - self.spec_stats['min']) / 2.0)
            elif self.spectrogram_normalization == 'mean_std':
                spectrogram = (spectrogram - self.spec_stats['mean']) / self.spec_stats['std']

            return spectrogram
        except RuntimeError:
            raise RuntimeError("[data/dataset.py] Can't open file {}. Please pre-render spectrogram files for this "
                               "dataset configuration.".format(file_path))

    def get_wav_file_path(self, preset_UID, midi_note, midi_velocity):
        """ Returns the path of a wav (from dexed_presets folder). Operators"""
        filename = "preset{:06d}_midi{:03d}vel{:03d}{}.wav".format(int(preset_UID), midi_note, midi_velocity,
                                                                   self._operators_suffix)
        return self.wav_files_dir.joinpath(filename)

    def get_wav_file(self, preset_UID, midi_note, midi_velocity):
        file_path = self.get_wav_file_path(preset_UID, midi_note, midi_velocity)
        try:
            return soundfile.read(file_path)
        except RuntimeError:
            raise RuntimeError("[data/dataset.py] Can't open file {}. Please pre-render audio files for this "
                               "dataset configuration.".format(file_path))

    def generate_wav_files(self):
        """ Reads all presets (names, param values, and labels) from .pickle and .txt files
         (see dexed.PresetDatabase.write_all_presets_to_files(...)) and renders them
         using attributes and constraints of this class (midi note, normalization, etc...)

         Floating-point .wav files will be stored in dexed presets' folder (see synth/dexed.py)

         Also writes a audio_render_constraints.json file that should be checked when loading data.
         """
        t_start = datetime.now()
        os.makedirs(self.wav_files_dir, exist_ok=True)
        # TODO multiple midi notes generation
        num_workers = os.cpu_count()
        workers_args = self._get_multi_note_workers_args(num_workers)
        # Multi-process rendering
        with multiprocessing.Pool(num_workers) as p:  # automatically closes and joins all workers
            p.map(self._generate_wav_files_batch, workers_args)
        self.write_audio_render_constraints_file()
        delta_t = (datetime.now() - t_start).total_seconds()
        num_wav_written = len(self.valid_preset_UIDs) * len(self.midi_notes)
        print("Finished writing {} .wav files ({:.1f}s total, {:.1f}ms/file)"
              .format(num_wav_written, delta_t, 1000.0*delta_t/num_wav_written))

    def _generate_wav_files_batch(self, worker_args):
        """ Generates wav files using the given list of (preset_UID, midi_pitch, midi_vel) tuples. """
        for preset_UID, midi_pitch, midi_vel in worker_args:
            self._generate_single_wav_file(preset_UID, midi_pitch, midi_vel)

    def _generate_single_wav_file(self, preset_UID, midi_pitch, midi_velocity):
        # Constrained params (1-element batch)
        preset_params = self.get_full_preset_params(preset_UID)
        x_wav, Fs = self._render_audio(torch.squeeze(preset_params.get_full(apply_constraints=True), 0),
                                       midi_pitch, midi_velocity)  # Re-Loads the VST
        soundfile.write(self.get_wav_file_path(preset_UID, midi_pitch, midi_velocity),
                        x_wav, Fs, subtype='FLOAT')

    def write_audio_render_constraints_file(self):
        file_path = dexed.PresetDatabase._get_presets_folder().joinpath("audio_render_constraints_file.json")
        with open(file_path, 'w') as f:
            json.dump({'constant_filter_and_tune_params': self.constant_filter_and_tune_params}, f)

    def check_audio_render_constraints_file(self):
        """ Raises a RuntimeError if the constraints used to pre-rendered audio are different from
        this instance constraints (S&H locked, filter/tune general params, ...) """
        file_path = dexed.PresetDatabase._get_presets_folder().joinpath("audio_render_constraints_file.json")
        with open(file_path, 'r') as f:
            constraints = json.load(f)
            if constraints['constant_filter_and_tune_params'] != self.constant_filter_and_tune_params:
                raise RuntimeError()\




if __name__ == "__main__":
    import sys
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir

    # xvfb display activation via pyvirtualdisplay wrapper
    from pyvirtualdisplay import Display
    disp = Display().start()

    # ============== DATA RE-GENERATION - FROM config.py ==================
    regenerate_wav = True  # multi-notes: a few minutes on a powerful CPU (20+ cores) - else: much longer
    # WARNING: when computing stats, please make sure that *all* midi notes are available
    regenerate_spectrograms_stats = True  # approx 3 min - 30e3 preset, single MIDI note (16mins for 16 MIDI notes)
    if regenerate_spectrograms_stats:
        assert len(config.model.midi_notes) > 1  # all MIDI notes (6?) must be used to compute stats


    #operators = config.model.dataset_synth_args[1]  # Custom operators limitation?
    operators = None

    # No label restriction, no normalization, etc...
    # But: OPERATORS LIMITATIONS and DEFAULT PARAM CONSTRAINTS (main params (filter, transpose,...) are constant)
    dexed_dataset = DexedDataset(note_duration=config.model.note_duration,
                                 midi_notes=config.model.midi_notes,
                                 multichannel_stacked_spectrograms=config.model.stack_spectrograms,
                                 n_fft=config.model.stft_args[0], fft_hop=config.model.stft_args[1],
                                 n_mel_bins=config.model.mel_bins,
                                 spectrogram_normalization=None,  # No normalization: we want to compute stats
                                 algos=None,  # allow all algorithms
                                 operators=operators,  # Operators limitation (config.py, or chosen above)
                                 # Params learned as categorical: maybe comment
                                 vst_params_learned_as_categorical=config.model.synth_vst_params_learned_as_categorical,
                                 restrict_to_labels=None,
                                 spectrogram_min_dB=config.model.spectrogram_min_dB,
                                 check_constrains_consistency=False,
                                 dataset_dir=config.model.dataset_dir,
                                 sample_rate=config.model.sampling_rate)
    print(dexed_dataset.preset_indexes_helper)
    if not regenerate_wav and not regenerate_spectrograms_stats:
        print(dexed_dataset)  # All files must be pre-rendered before printing
        for i in range(100):
            test = dexed_dataset[i]  # try get an item - for debug purposes

    if regenerate_wav:
        # WRITE ALL WAV FILES (approx. 10.5Go for 4.0s audio, 1 midi note)
        dexed_dataset.generate_wav_files()
    if regenerate_spectrograms_stats:
        # whole-dataset stats (for proper normalization)
        dexed_dataset.compute_and_store_spectrograms_stats()
    # ============== DATA RE-GENERATION - FROM config.py ==================

    # xvfb display deactivation
    disp.stop()


    # Dataloader debug tests
    if False:
        # Test dataload - to trigger potential errors
        # _, _, _ = dexed_dataset[0]

        dexed_dataloader = torch.utils.data.DataLoader(dexed_dataset, batch_size=128, shuffle=False,
                                                       num_workers=1)#os.cpu_count() * 9 // 10)
        t0 = time.time()
        for batch_idx, sample in enumerate(dexed_dataloader):
            print(batch_idx)
            print(sample)
            if batch_idx%10 == 0:
                print("batch {}".format(batch_idx))
        print("Full dataset read in {:.1f} minutes.".format((time.time() - t0) / 60.0))


