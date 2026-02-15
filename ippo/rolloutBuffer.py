import torch
import torch.nn.functional as F
import torch.nn as nn


class RolloutBuffer(nn.Module):

    def __init__(self, buffer_size, obs_dim, action_dim, gamma, gae_lambda):
        super(RolloutBuffer, self).__init__()

        