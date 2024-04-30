import config
import data.build

import h5py
from tqdm import tqdm
import os
import json


config_dict = {'model': config.model.__dict__, 'train': config.train.__dict__}
with open(os.path.join(config.model.dataset_dir, 'config.json'), 'w') as f:
    json.dump(config_dict, f)

dataset = data.build.get_dataset(config.model, config.train)

# Create dataset comprised of waveform, spectrograms, synth parameters, sample info, and labels
with h5py.File(os.path.join(config.model.dataset_dir, 'dataset.h5py'), 'x') as f:
    for i in tqdm(range(len(dataset))):
        waveform, spectrogram, synth_param, sample_info, label = dataset.generate_data(i)
        preset_UID, pitch, velocity = sample_info[0], sample_info[1], sample_info[2]
        grp = f.create_group(f'{preset_UID:06d}_{pitch}_{velocity}')
        grp.create_dataset('waveform', data=waveform)
        grp.create_dataset('spectrogram', data=spectrogram)
        grp.create_dataset('synth_param', data=synth_param)
        grp.create_dataset('sample_info', data=sample_info)
        grp.create_dataset('label', data=label)
