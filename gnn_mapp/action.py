import torch
import torch.nn.functional as F
import torch.nn as nn


class ActionHead(nn.Module):    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActionHead, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, G):

        G = self.fc1(G)
        G = F.relu(G)

        G = self.fc2(G)

        return G