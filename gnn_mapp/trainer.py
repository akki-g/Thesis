import torch
from torch.optim import Adam

from commPolicy import GraphConv
from rolloutBuffer import GNNRolloutBuffer
from action import ActionHead
from gppoAgent import CriticNetwork
from commPolicy import CommPolicy
from utils import build_adj, get_agent_pos


class GNNTrainer:
    def __init__(self, num_agents, env, obs_dim, hidden_dim, action_dim, F, G, K, lr, gamma, gae_lambda, device):
        self.device = device
        self.num_agents = num_agents
        self.agent_ids = sorted(env.possible_agents)

        self.comm_policy = CommPolicy(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim, F=F, G=G, K=K)
        self.comm_optim = Adam(self.comm_policy.parameters(), lr=lr)

        self.critics = []
        self.critic_optims = []
        for i in range(num_agents):
            self.critics.append(CriticNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, device=self.device))
            self.critic_optims.append(Adam(self.critics[i].parameters(), lr=lr))

        self.buffer = GNNRolloutBuffer(gamma=gamma, gae_lambda=gae_lambda, device=self.device)
        self.env = env
        
    def collect_rollouts(self, num_steps, r_comm=10):

        obs, info = self.env.reset()
        obs_tensor = torch.stack(torch.from_numpy(obs.values()))
        for _ in range(num_steps):
            agent_pos = get_agent_pos(self.env, self.device)
            S = build_adj(agent_pos, r_comm)

            actions, log_probs, entropy = self.comm_policy.get_actions(obs=obs_tensor, S=S)
            values = []
            for i in range(self.num_agents):
                value = self.critics[i](obs_tensor)
                values.append(value.detach())

            values = torch.stack(values)

            actions_pz = {}
            for i, a_id in enumerate(self.agent_ids):
                actions[a_id] = actions[i].cpu().detach()

            next_obs, rewards, dones, truncs, infos = self.env.step(actions) 
            
            rewards_tensor = torch.stack(rewards.values())
            dones_tensor = torch.stack(dones.values())
            self.buffer.add_timestep(obs=obs_tensor, actions=actions, rewards=rewards_tensor, dones=dones_tensor, log_probs=log_probs, A=S)

            if all(dones.values()) or all(truncs.values()):
                obs, info = self.env.reset()

                obs_tensor = torch.stack(torch.from_numpy(obs.values()))
            else:
                obs_tensor = torch.stack(torch.from_numpy(next_obs.values()))

        return obs



