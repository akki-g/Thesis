import torch
import torch.nn.functional as F
import torch.nn as nn
from graph import AdjNorm


class GraphConvLayer(nn.Module):
    def __init__(self):
        super(GraphConvLayer, self).__init__()

        

        