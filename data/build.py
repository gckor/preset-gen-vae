"""
Utility function for building datasets and dataloaders using given configuration arguments.
"""
import torch

from data import dataset
from data.sampler import build_subset_samplers


def get_dataset(config):
    """
    Returns the full (main) dataset.
    """
    full_dataset = dataset.DexedDataset(**config.dataset)
    
    if config.verbosity >= 2:
        print(full_dataset.preset_indexes_helper)
    elif config.verbosity >= 1:
        print(full_dataset.preset_indexes_helper.short_description)

    config.synth_params_count = full_dataset.learnable_params_count
    config.learnable_params_tensor_length = full_dataset.preset_indexes_helper._learnable_preset_size
    return full_dataset


def get_split_dataloaders(config, full_dataset):
    """ Returns a dict of train/validation/test DataLoader instances, and a dict which contains the
    length of each sub-dataset. """\
    # Dataloader easily build from samplers
    subset_samplers = build_subset_samplers(
        full_dataset,
        k_fold=config.train.current_k_fold,
        k_folds_count=config.train.k_folds,
        test_holdout_proportion=config.train.test_holdout_proportion
    )
    dataloaders = dict()
    sub_datasets_lengths = dict()

    for k, sampler in subset_samplers.items():
        # Last train minibatch must be dropped to help prevent training instability.
        # Worst case example, last minibatch contains only 8 elements,
        # mostly sfx: these hard to learn (or generate) item would have a much higher
        # equivalent learning rate because all losses are minibatch-size normalized.
        # No issue for eval though
        drop_last = (k.lower() == 'train')

        # Dataloaders based on previously built samplers
        dataloaders[k] = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=config.train.minibatch_size,
            sampler=sampler,
            drop_last=drop_last,
        )
        sub_datasets_lengths[k] = len(sampler.indices)
        if config.verbosity >= 1:
            print("[data/build.py] Dataset '{}' contains {}/{} samples ({:.1f}%)"
                  .format(k, sub_datasets_lengths[k], len(full_dataset),
                          100.0 * sub_datasets_lengths[k]/len(full_dataset)))
            
    return dataloaders

