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
from glob import glob
from pathlib import Path
from scipy.signal import resample
from utils.audio import MelSpectrogram
from torchaudio import transforms as T
from typing import Tuple


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/data1/Music/synth_sound_match/surge_2_257mel')
parser.add_argument('--render_sr', type=int, default=44100, help='sample rate that synth renders')
parser.add_argument('--model_sr', type=int, default=22050, help='sample rate that model receives')
parser.add_argument('--pitch', type=int, default=60, help='0 ~ 127')
parser.add_argument('--velocity', type=int, default=85, help='0 ~ 127')


class SurgeDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_dir: str,
            midi_notes: Tuple[Tuple[int, int]] = ((60, 85), ),
            multichannel_stacked_spectrograms: bool = False,
            spectrogram_normalization: str = 'min_max',
            **dataset_kwargs,
        ):
        self.dataset_dir = os.path.join(dataset_dir, 'surge_2')
        self.midi_notes = midi_notes
        self.multichannel_stacked_spectrograms = multichannel_stacked_spectrograms
        self.spectrogram_normalization = spectrogram_normalization
        
    def __len__(self):
        return self.valid_presets_count
    
    def __getitem__(self, i):
        preset_UID = i // self.midi_notes_per_preset
        midi_pitch, midi_velocity = self.midi_notes[0]
        data = self.get_data_from_file(preset_UID, midi_pitch, midi_velocity)
        waveform, spectrogram, synth_param = data
        return waveform, spectrogram, synth_param, preset_UID, preset_UID
    
    @property
    def midi_notes_per_preset(self):
        """ Number of available midi notes (different pitch and/or velocity) for a given preset. """
        return len(self.midi_notes)
    
    @property
    def valid_presets_count(self):
        wav_dir = os.path.join(self.dataset_dir, 'wav')
        dataset_length = sum(1 for _ in os.scandir(wav_dir))        
        return dataset_length
    
    def get_wav_file(self, preset_UID, midi_pitch, midi_velocity):
        file_name = f'preset{preset_UID:06d}_midi{midi_pitch:03d}vel{midi_velocity:03d}.wav'
        file_path = os.path.join(self.dataset_dir, 'wav', file_name)
        waveform = soundfile.read(file_path)[0].astype(np.float32)
        return waveform
    
    def get_spec_file(self, preset_UID, midi_pitch, midi_velocity):
        file_name = f'preset{preset_UID:06d}_midi{midi_pitch:03d}vel{midi_velocity:03d}.pt'
        file_path = os.path.join(self.dataset_dir, 'spectrogram', file_name)
        spectrogram = torch.load(file_path)
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
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)
    s = surgepy.createSurge(args.render_sr)
    fd = '/usr/share/surge-xt/patches_factory'
    preset_paths = sorted(list(Path(fd).rglob('*.fxp')))

    # spectrogram = MelSpectrogram(1024, 256, -120.0, 257, args.model_sr)
    spectrogram = T.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=128,
        norm='slaney',
    )

    preset_UID = 0
    # preset_UIDs = [584, 170,  73, 575, 552, 255, 619, 261, 438, 351, 303,   0, 344,
    #         609, 437, 302, 378, 308, 160, 101, 618, 193, 518, 628, 125,  65,
    #         460, 246, 591, 615, 583, 123, 613, 207, 368, 573, 234, 534,  45,
    #         191, 576, 601, 139, 133, 269, 586, 563, 554,  19, 105,  89, 590,
    #         561,  67, 228,  86, 284, 382, 183, 214,  57,  98, 436, 622,  94,
    #         566, 621, 556, 145, 154, 429, 274,  88, 158, 553, 581, 558,  70,
    #         120, 571, 182, 327, 100, 550, 328, 545, 199, 577, 236, 441,  24,
    #          21, 432,  93,  15, 270,  80, 263, 129, 398]
    
    for path in tqdm(preset_paths):
    # for preset_UID in tqdm(preset_UIDs):
        s.loadPatch(str(path))
        # s.loadPatch(str(preset_paths[preset_UID]))
        onesec = int(s.getSampleRate() / s.getBlockSize())
        buf = s.createMultiBlock(4 * onesec + 1)
        pos = 0

        s.playNote(0, args.pitch, args.velocity, 0)
        s.processMultiBlock(buf, pos, onesec * 3)
        pos = pos + onesec * 3

        s.releaseNote(0, args.pitch, 0 )
        s.processMultiBlock(buf, pos, onesec + 1)
        
        wav = buf[0][:args.render_sr * 4]

        if np.abs(wav).max() < 1e-4:
            continue

        wav = wav / np.abs(wav).max()
        num_samples = int(len(wav) * args.model_sr / args.render_sr)
        resampled_wav = resample(wav, num_samples)

        filename = "preset{:06d}_midi{:03d}vel{:03d}".format(preset_UID, args.pitch, args.velocity)
        soundfile.write(os.path.join(wav_dir, filename + '.wav'), resampled_wav, args.model_sr, subtype='FLOAT')        
        
        # tensor_spectrogram = spectrogram(resampled_wav)
        tensor_spectrogram = spectrogram(torch.FloatTensor(resampled_wav))
        tensor_spectrogram = torch.clip(torch.log(tensor_spectrogram + 1e-5)/12., -1, 1)
        torch.save(tensor_spectrogram, os.path.join(spec_dir, filename + '.pt'))
        preset_UID += 1

    print(f'{len(preset_paths)} wav and spectrogram files have been saved')
