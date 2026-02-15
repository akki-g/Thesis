import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, actions_dim):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, actions_dim)

        self.log_std = nn.Parameter(torch.zeros(actions_dim))

        self._init_weights()
        
    def _init_weights(self):

        nn.init.orthogonal_(self.fc1.weight, gain=2**0.5)
        nn.init.orthogonal_(self.fc2.weight, gain=2**0.5)
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, obs):
        
        x = self.fc1(obs)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.tanh(x)


        return x, self.log_std
    

    def get_action_and_log_probs(self, obs, action=None):

        mean, std = self.forward(obs)
        std = torch.exp(std)

        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy
    

    def evaluate_actions(self, obs, actions):

        mean, std = self.forward(obs)
        std = torch.exp(std)

        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, mean
    

    




actor = ActorNetwork(obs_dim=14, hidden_dim=64, actions_dim=2)
obs = torch.randn(32, 14)

actions, log_probs, entropy = actor.get_action_and_log_probs(obs)

print(f"Actions shape: {actions.shape}")
print(f"Log probs shape: {log_probs.shape}")
print(f"Entropy Shape: {entropy.shape}")
print(f"Sample actions: {actions[0]}")



class CriticNetwork(nn.Module):
    def __init__(self, obs_dims, hidden_dims):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 1)
        
        self._init_weights()

    def _init_weights(self):

        nn.init.orthogonal_(self.fc1.weight, gain=2**0.5)
        nn.init.orthogonal_(self.fc2.weight, gain=2**0.5)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)

        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, obs):

        x = self.fc1(obs)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)


        return x


critic = CriticNetwork(obs_dims=14, hidden_dims=64)
values = critic.forward(obs)

print(f"Values shape: {values.shape}")
print(f"Sample value: {values[0]}")