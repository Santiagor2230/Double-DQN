import torch
from torch import nn

class DQN(nn.Module):
  def __init__(self, hidden_size, obs_size, n_actions):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(obs_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
    )

    self.fc_value = nn.Linear(hidden_size, 1)
    self.fc_adv = nn.Linear(hidden_size, n_actions)

  def forward(self, x):
    x = self.net(x.float())
    adv = self.fc_adv(x) #V(s) Value function
    value = self.fc_value(x) #A(s,a) advantage function

    '''Q(s,a) = v(s) + A(s,a) - E(A(s,a'))'''
    return value + adv - torch.mean(adv, dim=1, keepdim=True).float()