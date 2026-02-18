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

        self.comm_policy = CommPolicy(
            obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim, F=F, G=G, K=K
        ).to(self.device)
        self.comm_optim = Adam(self.comm_policy.parameters(), lr=lr)

        self.critics = []
        self.critic_optims = []
        for i in range(num_agents):
            self.critics.append(CriticNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, device=self.device))
            self.critic_optims.append(Adam(self.critics[i].parameters(), lr=lr))

        self.buffer = GNNRolloutBuffer(gamma=gamma, gae_lambda=gae_lambda, device=self.device)
        self.env = env
        self._running_episode_return = 0.0
        self.metrics_history = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "mean_bellman_error": [],
            "mean_episode_return": [],
            "mean_episode_rewards": [],
        }

    def _safe_mean(self, values):
        if not values:
            return 0.0
        return float(sum(values) / len(values))
        
    def collect_rollouts(self, num_steps, r_comm=10):

        obs, info = self.env.reset()
        obs_tensor = torch.stack([torch.from_numpy(obs[a]) for a in self.agent_ids]).to(
            device=self.device, dtype=torch.float32
        )
        step_mean_rewards = []
        completed_episode_returns = []
        for _ in range(num_steps):
            agent_pos = get_agent_pos(self.env, self.device)
            S = build_adj(agent_pos, r_comm)

            actions, log_probs, entropy = self.comm_policy.get_actions(obs=obs_tensor, S=S)
            values = []
            for i in range(self.num_agents):
                value = self.critics[i](obs_tensor[i]).detach().squeeze()
                values.append(value)

            values = torch.stack(values)

            actions_pz = {}
            for i, a_id in enumerate(self.agent_ids):
                actions_pz[a_id] = actions[i].cpu().item()

            next_obs, rewards, dones, truncs, infos = self.env.step(actions_pz) 
            
            rewards_tensor = torch.tensor(
                [rewards[a] for a in self.agent_ids], dtype=torch.float32, device=self.device
            )
            dones_tensor = torch.tensor(
                [dones[a] for a in self.agent_ids], dtype=torch.float32, device=self.device
            )
            self.buffer.add_timestep(
                obs=obs_tensor.detach(),
                actions=actions.detach(),
                rewards=rewards_tensor,
                dones=dones_tensor,
                log_probs=log_probs.detach(),
                values=values,
                A=S.detach(),
            )

            step_mean_rewards.append(rewards_tensor.mean().item())
            self._running_episode_return += rewards_tensor.sum().item()

            if all(dones.values()) or all(truncs.values()):
                completed_episode_returns.append(self._running_episode_return)
                self._running_episode_return = 0.0
                obs, info = self.env.reset()

                obs_tensor = torch.stack([torch.from_numpy(obs[a]) for a in self.agent_ids]).to(
                    device=self.device, dtype=torch.float32
                )
            else:
                obs_tensor = torch.stack([torch.from_numpy(next_obs[a]) for a in self.agent_ids]).to(
                    device=self.device, dtype=torch.float32
                )

        rollout_metrics = {
            "mean_episode_return": self._safe_mean(completed_episode_returns)
            if completed_episode_returns
            else float(self._running_episode_return),
            "mean_episode_rewards": self._safe_mean(step_mean_rewards),
        }
        self.metrics_history["mean_episode_return"].append(rollout_metrics["mean_episode_return"])
        self.metrics_history["mean_episode_rewards"].append(rollout_metrics["mean_episode_rewards"])

        return obs_tensor, rollout_metrics
    
    def update(self, last_obs, num_epochs=30, B=64):

        with torch.no_grad():
            last_obs_tensor = last_obs.to(device=self.device, dtype=torch.float32)
            last_values = []
            for i in range(self.num_agents):
                value = self.critics[i](last_obs_tensor[i]).squeeze()
                last_values.append(value)

            last_values = torch.stack(last_values)
        self.buffer.compute_advantages(last_values=last_values)
        policy_losses = []
        entropies = []
        value_losses = []
        bellman_errors = []
        
        for i in range(num_epochs):
            for (obs, actions, old_log_probs, advantages, returns, A) in self.buffer.get_batches(B):
                all_new_log_probs = []
                all_entropy = []
                for t in range(obs.shape[0]):
                    new_lp_t, entropy_t, _ = self.comm_policy.evaluate_actions(obs[t], A[t], actions[t])

                    all_new_log_probs.append(new_lp_t)
                    all_entropy.append(entropy_t)
                
                all_new_log_probs = torch.stack(all_new_log_probs).to(device=self.device, dtype=torch.float32)
                all_entropy = torch.stack(all_entropy).to(device=self.device, dtype=torch.float32)

                ratio = torch.exp(all_new_log_probs - old_log_probs)    
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = all_entropy.mean()

                loss = policy_loss - self.entropy_coef * entropy_loss

                self.comm_optim.zero_grad()
                loss.backward()

                self.comm_optim.step()
                policy_losses.append(policy_loss.item())
                entropies.append(entropy_loss.item())

                for c in range(self.num_agents):
                    agent_obs = obs[:, c, :]
                    agent_returns = returns[:, c]
                    pred_values = self.critics[c](agent_obs).squeeze()
                    td_error = agent_returns - pred_values

                    value_loss = F.mse_loss(pred_values, agent_returns)
                    mean_bellman_error = td_error.abs().mean()

                    self.critic_optims[c].zero_grad()
                    value_loss.backward()
                    self.critic_optims[c].step()
                    value_losses.append(value_loss.item())
                    bellman_errors.append(mean_bellman_error.item())


        self.buffer.clear()
        update_metrics = {
            "policy_loss": self._safe_mean(policy_losses),
            "value_loss": self._safe_mean(value_losses),
            "entropy": self._safe_mean(entropies),
            "mean_bellman_error": self._safe_mean(bellman_errors),
        }
        self.metrics_history["policy_loss"].append(update_metrics["policy_loss"])
        self.metrics_history["value_loss"].append(update_metrics["value_loss"])
        self.metrics_history["entropy"].append(update_metrics["entropy"])
        self.metrics_history["mean_bellman_error"].append(update_metrics["mean_bellman_error"])
        return update_metrics



            



