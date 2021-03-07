"""
Script that can be edited to configure and run a queue of training runs.
Must be run as main

See the actual training function in train.py
"""


import importlib  # to reload config.py between each run
import numpy as np

import train


# = = = = = = = = = = config.py modifications = = = = = = = = = =
model_config_mods, train_config_mods = list(), list()
"""
Please write two lists of dicts, such that:
- (model|train)_config_mods contains the modifications applied to config.model and config.train, resp.
- each list index corresponds to a training run
- each dict key corresponds to an attribute of config.model.* or config.train.*. Empty dict to indicate
      that no config modification should be performed
"""
# Run 0
model_config_mods.append({})
train_config_mods.append({})
# Run 1
model_config_mods.append({'run_name': '05_vaeflow_slower_warmup'})
train_config_mods.append({'beta_start_value': 0.0, 'beta_warmup_epochs': 100})
"""
# Run 2
model_config_mods.append({'run_name': '20_no_useless_loss'})
train_config_mods.append({})
# Run 3
model_config_mods.append({'run_name': '21_no_useless_loss'})
train_config_mods.append({})
# Run 4
model_config_mods.append({'run_name': '03_less_bn', 'encoder_architecture': 'speccnn8l1_bn'})
train_config_mods.append({})
# Run 5
model_config_mods.append({'run_name': '03-2_less_bn', 'encoder_architecture': 'speccnn8l1_bn'})
train_config_mods.append({})
# Run 6
model_config_mods.append({'run_name': '04_fc_drop_0.3'})
train_config_mods.append({'fc_dropout': 0.3})
# Run 7
model_config_mods.append({'run_name': '04-2_fc_drop_0.3'})
train_config_mods.append({'fc_dropout': 0.3})
"""

# = = = = = = = = = = end of config.py modifications = = = = = = = = = =


if __name__ == "__main__":

    assert len(model_config_mods) == len(train_config_mods)

    for run_index in range(len(model_config_mods)):
        # Force config reload
        import config
        importlib.reload(config)

        print("================================================================")
        print("=============== Enqueued Training Run {}/{} starts ==============="
              .format(run_index+1, len(model_config_mods)))

        # Direct dirty modification of config.py module attributes
        for k, v in model_config_mods[run_index].items():
            config.model.__dict__[k] = v
        for k, v in train_config_mods[run_index].items():
            config.train.__dict__[k] = v

        # This dynamically modified config.py will be used by train.py
        train.train_config()

        print("=============== Enqueued Training Run {}/{} has finished ==============="
              .format(run_index+1, len(model_config_mods)))
        print("======================================================================")