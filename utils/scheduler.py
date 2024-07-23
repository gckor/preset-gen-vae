from torch.optim import lr_scheduler


def get_scheduler(scheduler_name, optimizer, **kwargs):
    scheduler = getattr(lr_scheduler, scheduler_name)(optimizer, **kwargs)
    return scheduler