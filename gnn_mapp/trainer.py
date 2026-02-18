import torch
from torch.optim import Adam
import torch.nn.functional as F

from graphConv import GraphConv
from rolloutBuffer import GNNRolloutBuffer
from action import ActionHead
from gppoAgent import CriticNetwork
from commPolicy import CommPolicy
from utils import build_adj, get_agent_pos


class GNNTrainer:
    def __init__(self, num_agents, env, obs_dim, hidden_dim, action_dim, F, G, K, lr, gamma, gae_lambda, clip_eps, value_coef, entropy_coef, device):
        self.device = device
        self.num_agents = num_agents
        self.agent_ids = sorted(env.possible_agents)

        self.clip_eps = clip_eps 
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

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
                value = self.critics[i](obs_tensor[i])
                values.append(value.detach())

            values = torch.stack(values)

            actions_pz = {}
            for i, a_id in enumerate(self.agent_ids):
                actions_pz[a_id] = actions[i].cpu().item()

            next_obs, rewards, dones, truncs, infos = self.env.step(actions_pz) 
            
            rewards_tensor = torch.tensor([rewards[a] for a in self.agent_ids])
            dones_tensor = torch.tensor([dones[a] for a in self.agent_ids])
            rewards_tensor = torch.stack(rewards_tensor)
            dones_tensor = torch.stack(dones_tensor)
            self.buffer.add_timestep(obs=obs_tensor, actions=actions, rewards=rewards_tensor, dones=dones_tensor, log_probs=log_probs, values=values, A=S)

            if all(dones.values()) or all(truncs.values()):
                obs, info = self.env.reset()

                obs_tensor = torch.stack(torch.from_numpy([obs[a] for a in self.agents]))
            else:
                obs_tensor = torch.stack(torch.from_numpy([next_obs[a] for a in self.agents]))

        return obs
    
    def update(self, last_obs, num_epochs=30):

        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)
            last_values = []
            for i in range(self.num_agents):
                value = self.critics[i](last_obs_tensor[i])
                last_values.append(value)

            last_values = torch.stack(last_values)
        self.buffer.compute_advantages(last_values=last_values)
        
        for i in range(num_epochs):
            for obs, actions, old_log_probs, advantages, returns, A in self.buffer.get_batches(B=64):
                new_log_probs, entropy, _ = self.comm_policy.evaluate_actions(obs, A, actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1+self.clip_eps, 1-self.clip_eps)

                policy_loss = -torch.min(surr1, surr2).mean(dim=1)

                values = []
                for i in range(self.num_agents):
                    value = self.critics[i](obs)
                    values.append(value)

                values = torch.stack(values)
                values_loss = F.mse_loss(values, returns)

                entropy_loss = -entropy.mean(dim=1)

                loss = policy_loss + self.value_coef * values_loss + self.entropy_coef * entropy_loss

                for i in range(self.num_agents):
                    self.critic_optims[i].zero_grad()
                
                loss.backwards()






