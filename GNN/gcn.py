import jax.numpy as jnp
from collections import defaultdict

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0] 

edges = data.edge_index.numpy().T
node_feats = data.x.numpy()
labels = data.y.numpy()

def he_initialization(in_dim, out_dim):
    rng = jnp.random.default_rng()   
    std = jnp.sqrt(2/in_dim)

    return rng.normal(0.0, std, size=(in_dim, out_dim))

def compute_normalized_adj(graph):
        
    A_a = graph.adj_matrix + jnp.eye(graph.num_nodes)
    degrees = jnp.array(A_a.sum(axis=1)).flatten()

    D_inv_sq = jnp.diag(jnp.power(degrees, -0.5, where=degrees>0))

        
    return D_inv_sq @ A_a @ D_inv_sq


class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

        self.adj_list = defaultdict(list)
        self.adj_matrix = jnp.zeros((num_nodes, num_nodes), dtype=int)

        self.edges = []

    def add_edge(self, u, v):
        if u not in self.adj_list:
            self.adj_list[v].append(u)
            self.adj_matrix[u][v] += 1

        if v not in self.adj_matrix:
            self.adj_list[v].append(u)
            self.adj_matrix[v][u] += 1

    def get_neighbors(self, node):
        return self.adj_list[node]
    
    def degree(self, node):
        return len(self.adj_list[node])
    



class GCN_layer:
    def __init__(self, graph: Graph, input_dim, out_dim):
        self.graph = graph
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.weights = he_initialization(input_dim, out_dim)

        self.norm_adj_mat = compute_normalized_adj(self.graph)

    
    def forward(self, prev_H, final=False):

        return self.norm_adj_mat @ prev_H @ self.weights
        
        

class GCN:
    def __init__(self, graph: Graph, input_dim, hidden_dim, output_dim):
        
        self.graph = graph
        self.l1 = GCN_layer(graph, input_dim, hidden_dim)
        self.l2 = GCN_layer(graph, hidden_dim, hidden_dim)
        self.out = GCN_layer(graph, hidden_dim, output_dim)

    
    def forward(self, X):

        H_1 = self.activate(self.l1.forward(X), 2)
        H_2 = self.activate(self.l2.forward(H_1), 2)
        Z = self.activate(self.out.forward(H_2), 1)

        return Z

    def compute_cross_entropy(self, pred, Y, train_mask):

        preds_train = pred[train_mask]
        Y_train = Y[train_mask]
        true_class_probs = preds_train[jnp.arange(len(Y_train)), Y_train]

        ce_loss = -jnp.sum(jnp.log(true_class_probs))

        return ce_loss / jnp.sum(train_mask)

    def activate(self, H, function):

        if function == 1:
            H = H - jnp.max(H, axis=1, keepdims=True)
            eH = jnp.exp(H)
            return eH / jnp.sum(eH, axis=1, keepdims=True)
        else:
            return H.clip(min=0)
        
    

        