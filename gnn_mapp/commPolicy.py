import torch
import torch.nn as nn
from torch.distributions import Categorical

from graphConv import GraphConv
from observation import ObservationEncoder
from action import ActionHead

class CommPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, F, G, K):
        super(CommPolicy, self).__init__()

        self.obsEncoder = ObservationEncoder(obs_dim, hidden_dim, F)
        self.graphConv = GraphConv(F, G, K)
        self.actionHead = ActionHead(G, hidden_dim, action_dim)


    def forward(self, obs, S):

        obs_encode = self.obsEncoder(obs)
        agg_feats = self.graphConv(obs_encode, S)
        logits = self.actionHead(agg_feats)

        return logits
    
    def get_actions(self, obs, S):

        logits = self.forward(obs, S)

        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy
    
    def evaluate_actions(self, obs, S, actions):
        
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.long)

        logits = self.forward(obs, S)
        dist = Categorical(logits=logits)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, entropy, logits


