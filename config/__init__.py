from omegaconf import OmegaConf


def load_config():
    default = OmegaConf.load('config/default.yaml')
    encoder = OmegaConf.load(f'config/encoder/{default.model.encoder_architecture}.yaml')
    scheduler = OmegaConf.load(f'config/scheduler/{default.train.scheduler_name}.yaml')
    config = OmegaConf.merge(default, encoder, scheduler)
    return config