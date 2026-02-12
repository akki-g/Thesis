import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import numpy as np
device = "cpu"
if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

dataset = Planetoid(root='/tmp/Cora', name='cora')
data = dataset[0]

def compute_norm_adj(edge_idx, num_nodes):
    """
    computes norm adj matrix 
    return sparse tensor representation of the normed adj matrix
    """

    # add self loops
    # stack twice
    self_loop_edge_idx = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)

    # concate original edges with self loops
    # edge idx is (2, num_edges) and self loop is (2, num_nodes)
    # result is (2, num_edges+num_nodes)
    edge_idx = torch.cat([edge_idx, self_loop_edge_idx], dim=1)

    # compute degree mat
    row, col = edge_idx[0], edge_idx[1]
    deg = torch.bincount(row, minlength=num_nodes).float()


    # compute D^(-1/2)
    D_inv_sqrt = deg.pow(-0.5)
    D_inv_sqrt[D_inv_sqrt == float('inf')] = 0

    # norm edge weights
    edge_weight = D_inv_sqrt[row] * D_inv_sqrt[col]

    num_edges = edge_idx.size(1)
    A_norm = torch.sparse_coo_tensor(
        edge_idx, 
        edge_weight,
        (num_nodes, num_nodes)
    )

    return A_norm


A_norm = compute_norm_adj(data.edge_index, data.num_nodes)


class GCNLayer(nn.Module):
    """single gcn layer essentially implemeting eq 2 of the paper"""

    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()

        ## nn.Linear creates a learnable weight matrix in shape of (in_feat, out_feats)
        # inits weights by default

        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x, A_norm):
        """forward pass of GCN layer
        returns a tensor of transformed features"""

        x = self.linear(x)
        x = torch.sparse.mm(A_norm, x)

        return x
    


class GCN(nn.Module):
    """2 layer network that implements eq 9 from the paper"""

    def __init__(self, in_feats, hidden_dim, num_classes, dropout=0.5):
        super(GCN, self).__init__()

        #layer 1: in_dim -> hidden_dim
        self.gc1 = GCNLayer(in_feats, hidden_dim)

        #layer 2: hidden_dim -> num_classes
        self.gc2 = GCNLayer(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, A_norm):
        """forward pass through network return class logits"""

        x = self.gc1(x, A_norm)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.gc2(x, A_norm)

        return x


def train_gcn():
    hidden_dim = 16
    dropout = 0.5
    lr = 0.01
    weight_decay = 5e-4
    epochs = 200

    feats = data.x
    labels = data.y

    model = GCN(
        in_feats=dataset.num_features, 
        hidden_dim=hidden_dim,
        num_classes=dataset.num_classes,
        dropout=dropout
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(feats, A_norm)

        loss = criterion(out[data.train_mask], labels[data.train_mask])
        loss.backward()

        optimizer.step()

        if (epoch + 1) % 10  == 0:
            model.eval()
            with torch.no_grad():
                out = model(feats, A_norm)

                pred = out.argmax(dim=1)
                val_correct = (pred[data.val_mask] == labels[data.val_mask]).sum().item()
                val_acc = val_correct / data.val_mask.sum().item()

                print(f"Epoch {epoch+1}, Loss {loss.item():.4f}. Val Acc {val_acc:.4f}")
            model.train()

    model.eval()
    with torch.no_grad():
        out = model(feats, A_norm)
        pred = out.argmax(dim=1)
        test_correct = (pred[data.test_mask] == labels[data.test_mask]).sum().item()
        test_acc = test_correct / data.test_mask.sum().item()
        print(f'\nTest Accuracy: {test_acc:.4f}')


model = train_gcn()