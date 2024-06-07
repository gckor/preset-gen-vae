import librosa
import numpy as np
from omegaconf import OmegaConf
from pyvirtualdisplay import Display
from matplotlib import pyplot as plt

from data import build
from data.preset import DexedPresetsParams


config = OmegaConf.load('config.yaml')
dataset = build.get_dataset(config)
dataloader = build.get_split_dataloaders(config, dataset)

device = 'cuda'
midi_pitch = 60
midi_velocity = 85
n_fft=1024
fft_hop=256

dataloader_iter = iter(dataloader['train'])
sample = next(dataloader_iter)
x_in, v_in, sample_info = sample[1].to(device), sample[2].to(device), sample[3].to(device)
full_param = DexedPresetsParams(learnable_presets=v_in, dataset=dataset).get_full()[0, :].cpu().numpy()

full_param_change = full_param.copy()

if full_param_change[7] < 0.5:
    full_param_change[7] = min(full_param_change[7] + 0.33, 1)
    change = 'Up'
else:
    full_param_change[7] = max(full_param_change[7] - 0.33, 0)
    change = 'Down'

disp = Display()
disp.start()
wav = dataset._render_audio(full_param, midi_pitch, midi_velocity)[0]
wav_change = dataset._render_audio(full_param_change, midi_pitch, midi_velocity)[0]
disp.stop()

stft = np.abs(librosa.stft(wav, n_fft, fft_hop))
stft_change = np.abs(librosa.stft(wav_change, n_fft, fft_hop))
eps = 1e-4
log_stft = np.log10(np.maximum(stft, eps))
log_stft_change = np.log10(np.maximum(stft_change, eps))

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
librosa.display.specshow(log_stft, shading='flat', ax=axes[0], cmap='magma')
librosa.display.specshow(log_stft_change, shading='flat', ax=axes[1], cmap='magma')
axes[0].set(title='Original')
axes[1].set(title=f'LFO Speed {change}')
fig.tight_layout()
fig.savefig('spectrogram.png')
plt.close(fig)