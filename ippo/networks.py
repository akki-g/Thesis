import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical

def _default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, actions_dim, device=None):
        super(ActorNetwork, self).__init__()
        self.device = torch.device(device) if device is not None else _default_device()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, actions_dim)

        self._init_weights()
        self.to(self.device)
        
    def _init_weights(self):

        nn.init.orthogonal_(self.fc1.weight, gain=2**0.5)
        nn.init.orthogonal_(self.fc2.weight, gain=2**0.5)
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device, dtype=torch.float32)
        
        x = self.fc1(obs)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x
    

    def get_action_and_log_probs(self, obs, action=None):

        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()
        else:
            if not torch.is_tensor(action):
                action = torch.as_tensor(action, dtype=torch.long, device=self.device)
            else:
                action = action.to(self.device, dtype=torch.long)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
    

    def evaluate_actions(self, obs, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        else:
            actions = actions.to(self.device, dtype=torch.long)

        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(actions)

        entropy = dist.entropy()

        return log_prob, entropy, logits
    



class CriticNetwork(nn.Module):
    def __init__(self, obs_dims, hidden_dims, device=None):
        super(CriticNetwork, self).__init__()
        self.device = torch.device(device) if device is not None else _default_device()
        self.fc1 = nn.Linear(obs_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 1)
        
        self._init_weights()
        self.to(self.device)

    def _init_weights(self):

        nn.init.orthogonal_(self.fc1.weight, gain=2**0.5)
        nn.init.orthogonal_(self.fc2.weight, gain=2**0.5)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)

        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device, dtype=torch.float32)

        x = self.fc1(obs)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)


    
        return x
