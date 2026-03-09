import torch
import numpy as np
from typing import Literal, Union


METHODS = Literal["symmetric", "random-walk", "laplacian", "raw"]
COMM = Literal["dist", "knn"]
Arr = Union[torch.Tensor, np.ndarray]

def get_agent_pos(env):
    raise NotImplementedError

class AdjNorm:

    def __init__(self, method: METHODS, self_loops: bool = True, eps: float = 1e-12):
        self.method = method
        self.self_loops = self_loops
        self.eps = eps

    # GCN-style normalization: D~^{-1/2} (A + I) D~^{-1/2}
    def symmetric(self, adj: Arr):
        N = adj.shape[0]
        if isinstance(adj, np.ndarray):
            A_hat = adj + np.eye(N, dtype=adj.dtype) if self.self_loops else adj
            deg= A_hat.sum(axis=1)
            inv_sqrt_deg = np.zeros_like(deg, dtype=np.float64)
            np.power(np.clip(deg, self.eps, None), -0.5, out=inv_sqrt_deg)

            return (A_hat * inv_sqrt_deg[:, None]) * inv_sqrt_deg[None, :]
        
        else:
            A_hat = adj + torch.eye(N, device=adj.device, dtype=adj.dtype) if self.self_loops else adj

            deg = A_hat.sum(dim=1)
            inv_sqrt_deg = deg.clamp_min(self.eps).to(torch.float32).pow(-0.5)

            return (A_hat * inv_sqrt_deg.unsqueeze(1)) * inv_sqrt_deg.unsqueeze(0)
    # random-walk norm: A_norm = D~^{-1} (A + I)
    def random_walk(self, adj: Arr):
        N = adj.shape[0]
        if isinstance(adj, np.ndarray):
            A_hat = adj + np.eye(N, dtype=np.float64) if self.self_loops else adj

            deg = A_hat.sum(axis=1)
            inv_deg = np.zeros_like(deg, dtype=np.float64)
            np.divide(1.0, np.clip(deg, self.eps, None), out=inv_deg)

            return A_hat * inv_deg[:, None]
    
        A_hat = adj + torch.eye(N, device=adj.device) if self.self_loops else adj
        deg = A_hat.sum(dim=1)
        inv_deg = deg.clamp_min(self.eps).to(torch.float32).reciprocal()

        return A_hat * inv_deg.unsqueeze(1)

    
    # laplacian norm: L = I - D^{-1/2} A D^{-1/2}
    def laplacian(self, adj: Arr):
        N = adj.shape[0]

        if isinstance(adj, np.ndarray):
            deg = adj.sum(axis=1)

            inv_sqrt_deg = np.zeros_like(deg, dtype=np.float64)
            np.power(np.clip(deg, self.eps, None), -0.5, out=inv_sqrt_deg)

            A_norm = (adj * inv_sqrt_deg[:, None]) * inv_sqrt_deg[None, :]

            return np.eye(N, dtype=A_norm.dtype) - A_norm
        

        deg = adj.sum(dim=1)

        inv_sqrt_deg = deg.clamp_min(self.eps).to(torch.float32).pow(-0.5)
        A_norm = (adj * inv_sqrt_deg.unsqueeze(1)) * inv_sqrt_deg.unsqueeze(0)

        return torch.eye(N, device=adj.device, dtype=A_norm.dtype) - A_norm
    

    def __call__(self, adj: Arr, *args, **kwds):
        
        if self.method == 'symmetric':
            return self.symmetric(adj)
        
        elif self.method == "random-walk":
            return self.random_walk(adj)
        
        elif self.method == "laplacian":
            return self.laplacian(adj)
        
        elif self.method == "raw":
            return adj
        
        else:
            raise ValueError(f"Undefined normilization method: {self.method!r}") 

    

class GraphBuilder:
    def __init__(self, build_method: COMM, norm_method: METHODS, k = 3, r_comm = 1.0, device="cpu", self_loop = True):

        self.build_method = build_method
        self.norm_method = norm_method

        self.k = k
        self.r_comm = r_comm

        self.device = device

        self.normalizer = AdjNorm(method=norm_method, self_loops=self_loop)


    def _pairwise_dist(self, agent_pos):

        dist = agent_pos.unsqueeze(1) - agent_pos.unsqueeze(0)
        return torch.norm(dist, dim=-1)
    
    def _build_dist(self, agent_pos: torch.Tensor):

        N = agent_pos.shape[0]
        dist = self._pairwise_dist(self, agent_pos)

        adj = (dist <= self.r_comm).float()
        adj.fill_diagonal_(0.0)

        return adj
    
    def _build_knn(self, agent_pos: torch.Tensor) -> torch.Tensor:

        N = agent_pos.shape[0]
        k = min(self.k, N-1)

        dist = self._pairwise_dist(agent_pos)

        dist_no_self = dist.clone()
        dist_no_self.fill_diagonal_(float('inf'))

        _, knn_indicies = torch.topk(dist_no_self, k=k, largest=False)

        adj_directed = torch.zeros((N, N), dtype=torch.float32, device=self.device)
        row_indicies = torch.arange(N, device=self.device).unsqueeze(1).expand(N, k)
        adj_directed[row_indicies, knn_indicies] = 1.0

        adj = torch.clamp(adj_directed + adj_directed.T, max=1.0)
        adj.fill_diagonal_(0.0)

        return adj
    

    def __call__(self, env, *args, **kwds):
        pos = env.get_pos()

        if self.build_method == "dist":
            adj = self._build_dist(pos)
        elif self.build_method == "knn":
            adj = self._build_knn(pos)
        else:
            raise NotImplementedError(f"No definition for the build method: {self.build_method}")
        
        adj = self.normalizer(adj)

        return adj


