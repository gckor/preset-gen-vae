

import os
import json
import torch
import soundfile
import numpy as np
import pandas as pd
from typing import Tuple


class NoisemakerDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_dir: str,
            n_fft: int = 1024,
            fft_hop: int = 256,
            n_mel_bins: int = 128,
            midi_notes: Tuple[Tuple[int, int]] = ((60, 85), ),
            multichannel_stacked_spectrograms: bool = False,
            spectrogram_normalization: str = 'min_max',
            **dataset_kwargs,
        ):
        self.dataset_config = f'nfft{n_fft:04d}hop{fft_hop:04d}mels{n_mel_bins:04d}'
        self.dataset_dir = os.path.join(dataset_dir, 'noisemaker')
        self.midi_notes = midi_notes
        self.multichannel_stacked_spectrograms = multichannel_stacked_spectrograms
        self.spectrogram_normalization = spectrogram_normalization
        self._load_spectrogram_stats()
        
    def __len__(self):
        if self.multichannel_stacked_spectrograms:
            dataset_length =  self.valid_presets_count
        else:
            dataset_length =  self.valid_presets_count * self.midi_notes_per_preset
        
        return dataset_length
    
    def __getitem__(self, i):
        preset_UID = i // self.midi_notes_per_preset
        midi_pitch, midi_velocity = self.midi_notes[0]
        data = self.get_data_from_file(preset_UID, midi_pitch, midi_velocity)
        waveform, spectrogram, synth_param = data
        return waveform, spectrogram, synth_param, preset_UID
    
    @property
    def midi_notes_per_preset(self):
        """ Number of available midi notes (different pitch and/or velocity) for a given preset. """
        return len(self.midi_notes)
    
    @property
    def valid_presets_count(self):
        """ Total number of presets currently available from this dataset. """
        stats_filename = f'NoisemakerDataset_spectrogram_{self.dataset_config}_full.csv'
        full_stats = pd.read_csv(os.path.join(self.dataset_dir, 'stats', stats_filename))
        return len(full_stats)
    
    def _load_spectrogram_stats(self):
        """
        To be called by the child class, after this parent class construction (because stats file path
        depends on child class constructor arguments).
        """
        file_name = f'NoisemakerDataset_spectrogram_{self.dataset_config}.json'
        file_path = os.path.join(self.dataset_dir, 'stats', file_name)
        with open(file_path, 'r') as f:
            self.spec_stats = json.load(f)
    
    def get_wav_file(self, preset_UID, midi_pitch, midi_velocity):
        file_name = f'preset{preset_UID:06d}_midi{midi_pitch:03d}vel{midi_velocity:03d}.wav'
        file_path = os.path.join(self.dataset_dir, 'wav', file_name)
        waveform = soundfile.read(file_path)[0].astype(np.float32)
        return waveform
    
    def get_spec_file(self, preset_UID, midi_pitch, midi_velocity):
        file_name = f'preset{preset_UID:06d}_midi{midi_pitch:03d}vel{midi_velocity:03d}.pt'
        file_path = os.path.join(self.dataset_dir, 'spectrogram', file_name)
        spectrogram = torch.load(file_path)

        # if self.spectrogram_normalization == 'min_max':
        #     spec_min, spec_max = self.spec_stats['min'], self.spec_stats['max']
        #     spectrogram = -1.0 + (spectrogram - spec_min) / ((spec_max - spec_min) / 2.0)
        # elif self.spectrogram_normalization == 'mean_std':
        #     spectrogram = (spectrogram - self.spec_stats['mean']) / self.spec_stats['std']

        return spectrogram.unsqueeze(0)
    
    def get_data_from_file(self, preset_UID, midi_pitch, midi_velocity):
        synth_param = torch.ones(1)
        waveform = self.get_wav_file(preset_UID, midi_pitch, midi_velocity)
        spectrogram = self.get_spec_file(preset_UID, midi_pitch, midi_velocity)
        return waveform, spectrogram, synth_param