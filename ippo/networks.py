import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, actions_dim):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, actions_dim)

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
        x = F.softmax(x)


        return x
    

    def get_action_and_log_probs(self, obs, action=None):

        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy
    

    def evaluate_actions(self, obs, actions):

        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(actions)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, logits
    



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
