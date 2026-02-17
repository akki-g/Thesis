import torch
import torch.nn.functional as F
import torch.nn as nn

class ObservationEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ObservationEncoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x
    
