import torch
import torch.nn.functional as F
import torch.nn as nn

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, actions_dim):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, actions_dim)

        self.log_std = nn.Parameter(torch.zeros(actions_dim))

        


