import data.build
from config import load_config

import h5py
import os
from tqdm import tqdm


'''
Generate synth param in advance due to slow speed of the param extraction.
config.dataset.midi_notes must be [[60, 85]].
Check num_param_bins, vst_params_learned_as_categorical and dataset_name.
'''

config = load_config()
dataset = data.build.get_dataset(config)
os.makedirs(config.dataset.dataset_dir, exist_ok=True)

# Create dataset comprised of waveform, spectrograms, synth parameters, sample info, and labels
with h5py.File(os.path.join(config.dataset.dataset_dir, f'{config.dataset.dataset_name}.h5py'), 'x') as f:
    for i in tqdm(range(len(dataset))):
        synth_param, sample_info, label = dataset.generate_data(i)
        preset_UID, pitch, velocity = sample_info[0], sample_info[1], sample_info[2]
        grp = f.create_group(f'{preset_UID:06d}_{pitch}_{velocity}')
        grp.create_dataset('synth_param', data=synth_param)
        grp.create_dataset('sample_info', data=sample_info)
        grp.create_dataset('label', data=label)
