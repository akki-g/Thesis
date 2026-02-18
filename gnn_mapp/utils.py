import torch
import numpy as np

def build_adj(agent_pos, r_comm):   

    diff = agent_pos.unsqueeze(1) - agent_pos.unsqueeze(0)

    dist = torch.norm(diff, dim=-1)

    adj = (dist <= r_comm).float()

    adj.fill_diagonal_(0.0)

    return dist, adj


def get_agent_pos(env, device): 
    base = getattr(env, "unwrapped", env)
    mpe = base

    if not (hasattr(mpe, "world") and hasattr(mpe.world, "agents")):
        mpe = getattr(base, "env", base)
    if not (hasattr(mpe, "world") and hasattr(mpe.world, "agents")):
        mpe = getattr(getattr(base, "env", None), "unwrapped", base)

    if not (hasattr(mpe, "world") and hasattr(mpe.world, "agents")):
        raise AttributeError(
            "Couldn't find MPE world/agents. "
            "This function is intended for PettingZoo MPE simple_spread."
        )

    # world.agents order corresponds to env.agents naming order in MPE
    pos_np = np.stack([agent.state.p_pos for agent in mpe.world.agents], axis=0)  # (N, 2)

    return torch.as_tensor(pos_np, dtype=torch.float32, device=device)