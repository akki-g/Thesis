import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, F, G, K):
        super(GraphConv, self).__init__()
        self.F = F
        self.G = G
        self.K = K

        self.weights = nn.ParameterList(
            [nn.Parameter(torch.eye(F,G)) for _ in range(K)
            ])

        """ for p in self.weights:
            nn.init.xavier_uniform_(p)"""


    def forward(self, X, S):

        Z = X
        accum = X.new_zeros(X.size(0), self.G)

        for k in range(self.K):
            accum += torch.matmul(Z, self.weights[k])

            Z = torch.matmul(S, Z)

        return accum
            


gConv = GraphConv(F=2, G=2, K=3)

X = torch.tensor([[1, 0],
                  [0, 1],
                  [2, 0]], dtype=torch.float32)

S = torch.tensor([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]], dtype=torch.float32)


A = gConv(X, S)
print(f"K: {gConv.K}")
print(A)