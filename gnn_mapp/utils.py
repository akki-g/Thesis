import torch


def build_adj(agent_pos, r_comm):   

    diff = agent_pos.unsqueeze(1) - agent_pos.unsqueeze(0)

    dist = torch.norm(diff, dim=-1)

    adj = (dist <= r_comm).float()

    adj.fill_diagonal_(0.0)

    return dist, adj