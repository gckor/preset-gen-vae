import torch

class Replaybuffer():
    def __init__(self, n_states: int, n_actions: int, device: str):
        self.actions = torch.zeros((n_states, n_actions), dtype=torch.int64).to(device)
        self.rewards = torch.ones((n_states, 1), dtype=torch.float32).to(device) * -10

    def store(self, preset_UID: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        update = (rewards > self.rewards[preset_UID]).squeeze()
        update_inds = preset_UID[update]
        self.rewards[update_inds] = rewards[update]
        self.actions[update_inds] = actions[update]

    def sample(self, preset_UID: torch.Tensor):
        return self.actions[preset_UID], self.rewards[preset_UID]

    def save(self, path):
        torch.save(self.actions, path.joinpath('actions.pt'))
        torch.save(self.rewards, path.joinpath('rewards.pt'))

    def load(self, path):
        self.actions = torch.load(path.joinpath('actions.pt'))
        self.rewards = torch.load(path.joinpath('rewards.pt'))
