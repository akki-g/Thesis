import torch
from torch_geometric.datasets import Planetoid
import scipy.sparse as sp   
import numpy as np

dataset = Planetoid(root='/tmp/cora', name='Cora')
device = "cpu"
if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

torch.device.type = device

def load_sparse(dataset):
    data = dataset[0]
    edges = data.edge_index.numpy().T
    feats = data.x.numpy()
    labels = data.y.numpy()
    num_nodes = feats.shape[0]

    A = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes), 
        dtype=np.float32
    )

    A = A + A.T.multiply(A.t > A) - A.multiply(A.T > A)

    A_norm = compute_normalized_adj(A)

    feats_sparse = sp.csr_matrix(feats, dtype=np.float32)
    labels_onehot = np.eye(dataset.num_classes)[labels]

    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()
    val_mask = data.val_mask.numpy()

    return feats_sparse, A_norm, labels_onehot, train_mask, test_mask, val_mask

def he_init(input_dim, out_dim):
    rng = torch.Generator().manual_seed(0)
    std = (2.0/input_dim) ** 2

    return torch.randn((input_dim, out_dim), generator=rng)


def compute_normalized_adj(A):

    A = A + sp.eye(A.shape[0]) 
    rowsum = np.array(A.sum(axis=1)).flatten()

    D_inv_sqrt = np.power(rowsum, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.

    D_inv_sqrt = sp.diag(D_inv_sqrt)

    return D_inv_sqrt.dot(A).dot(D_inv_sqrt)


def sparse_to_tuple(sp_matrix):
    """conver sp.sparse representation to tf sparse tensor"""
    if not sp.isspmatrix_coo(sp_matrix):
        sp_matrix = sp_matrix.tocoo()


    indicies = np.vstack((sp_matrix.row, sp_matrix.col)).T  
    vals = sp_matrix.data
    shape = sp_matrix.shape

    return indicies, vals, shape