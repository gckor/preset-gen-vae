import sys
sys.path.append('/home/surge/ignore/bpy/src/surge-python')

import os
import json
import torch
import surgepy
import argparse
import soundfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.signal import resample
from utils.audio import MelSpectrogram
from typing import Tuple


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/data1/Music/synth_sound_match/surge')
parser.add_argument('--render_sr', type=int, default=44100, help='sample rate that synth renders')
parser.add_argument('--model_sr', type=int, default=22050, help='sample rate that model receives')
parser.add_argument('--pitch', type=int, default=60, help='0 ~ 127')
parser.add_argument('--velocity', type=int, default=85, help='0 ~ 127')


class SurgeDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_dir: str,
            n_fft: int = 1024,
            fft_hop: int = 256,
            n_mel_bins: int = 257,
            midi_notes: Tuple[Tuple[int, int]] = ((60, 85), ),
            multichannel_stacked_spectrograms: bool = False,
            spectrogram_normalization: str = 'min_max',
            **dataset_kwargs,
        ):
        self.dataset_config = f'nfft{n_fft:04d}hop{fft_hop:04d}mels{n_mel_bins:04d}'
        self.dataset_dir = os.path.join(dataset_dir, 'surge')
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
        stats_filename = f'SurgeDataset_spectrogram_{self.dataset_config}_full.csv'
        full_stats = pd.read_csv(os.path.join(self.dataset_dir, 'stats', stats_filename))
        return len(full_stats)
    
    def _load_spectrogram_stats(self):
        """
        To be called by the child class, after this parent class construction (because stats file path
        depends on child class constructor arguments).
        """
        file_name = f'SurgeDataset_spectrogram_{self.dataset_config}.json'
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

        if self.spectrogram_normalization == 'min_max':
            spec_min, spec_max = self.spec_stats['min'], self.spec_stats['max']
            spectrogram = -1.0 + (spectrogram - spec_min) / ((spec_max - spec_min) / 2.0)
        elif self.spectrogram_normalization == 'mean_std':
            spectrogram = (spectrogram - self.spec_stats['mean']) / self.spec_stats['std']

        return spectrogram.unsqueeze(0)
    
    def get_data_from_file(self, preset_UID, midi_pitch, midi_velocity):
        synth_param = torch.ones(1)
        waveform = self.get_wav_file(preset_UID, midi_pitch, midi_velocity)
        spectrogram = self.get_spec_file(preset_UID, midi_pitch, midi_velocity)
        return waveform, spectrogram, synth_param


if __name__ == '__main__':
    args = parser.parse_args()
    wav_dir = os.path.join(args.dataset_dir, 'wav')
    spec_dir = os.path.join(args.dataset_dir, 'spectrogram')
    stats_dir = os.path.join(args.dataset_dir, 'stats')
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    s = surgepy.createSurge(args.render_sr)
    fd = s.getFactoryDataPath()
    preset_paths = list(Path(fd).rglob('*.fxp'))
    spectrogram = MelSpectrogram(1024, 256, -120.0, 257, args.model_sr)
    full_stats = {
        'UID': np.zeros((len(preset_paths),), dtype=np.int32),
        'min': np.zeros((len(preset_paths),)),
        'max': np.zeros((len(preset_paths),)),
        'mean': np.zeros((len(preset_paths),)),
        'var': np.zeros((len(preset_paths),))
    }

    for preset_UID, path in tqdm(enumerate(preset_paths), total=len(preset_paths)):
        s.loadPatch(str(path))
        onesec = int(s.getSampleRate() / s.getBlockSize())
        buf = s.createMultiBlock(4 * onesec + 1)
        pos = 0

        s.playNote(0, args.pitch, args.velocity, 0)
        s.processMultiBlock(buf, pos, onesec * 3)
        pos = pos + onesec * 3

        s.releaseNote(0, args.pitch, 0 )
        s.processMultiBlock(buf, pos, onesec + 1)
        
        wav = buf[0][:args.render_sr * 4]
        num_samples = int(len(wav) * args.model_sr / args.render_sr)
        resampled_wav = resample(wav, num_samples)

        filename = "preset{:06d}_midi{:03d}vel{:03d}".format(preset_UID, args.pitch, args.velocity)
        soundfile.write(os.path.join(wav_dir, filename + '.wav'), resampled_wav, args.model_sr, subtype='FLOAT')        
        
        tensor_spectrogram = spectrogram(resampled_wav)
        full_stats['UID'][preset_UID] = preset_UID
        full_stats['min'][preset_UID] = torch.min(tensor_spectrogram).item()
        full_stats['max'][preset_UID] = torch.max(tensor_spectrogram).item()
        full_stats['var'][preset_UID] = torch.var(tensor_spectrogram).item()
        full_stats['mean'][preset_UID] = torch.mean(tensor_spectrogram, dim=(0, 1)).item()
        torch.save(tensor_spectrogram, os.path.join(spec_dir, filename + '.pt'))

    print(f'{len(preset_paths)} wav and spectrogram files have been saved')

    dataset_stats = {
        'min': full_stats['min'].min(),
        'max': full_stats['max'].max(),
        'mean': full_stats['mean'].mean(),
        'std': np.sqrt(full_stats['var'].mean())
    }
    full_stats['std'] = np.sqrt(full_stats['var'])
    del full_stats['var']
    full_stats = pd.DataFrame(full_stats)
    full_stats.to_csv(os.path.join(stats_dir, 'SurgeDataset_spectrogram_nfft1024hop0256mels0257_full.csv'))

    with open(os.path.join(stats_dir, 'SurgeDataset_spectrogram_nfft1024hop0256mels0257.json'), 'w') as f:
        json.dump(dataset_stats, f)

    print(f'Statistics from {len(full_stats)} spectrograms has been written to .csv and .json files')
