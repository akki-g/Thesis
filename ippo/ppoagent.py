import torch
import torch.nn.functional as F
import torch.nn as nn

from ippo.networks import ActorNetwork, CriticNetwork
from ippo.rolloutBuffer import RolloutBuffer


class PPOAgent:

    def __init__(self, obs_dim, hidden_dim, action_dim, lr=3e-4, buffer_size = 1024, gamma=0.99, gae_lambda=0.95, 
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        
        #networks
        self.actor = ActorNetwork(obs_dim, hidden_dim, action_dim)
        self.critic = CriticNetwork(obs_dim, hidden_dim)
        #buffer
        self.buffer = RolloutBuffer(buffer_size, obs_dim, action_dim, gamma, gae_lambda)

        #actor/critic optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        #hyperparams
        self.clip_eps = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma


    def select_action(self, obs):

        if not torch.is_tensor(obs):
            obs = torch.stack(obs).squeeze()

        
        actions, log_probs, entropys = self.actor.get_action_and_log_probs(obs)
        value = self.critic.forward(obs)


        return actions.item(), log_probs.item(), value.item()
    

    def update(self, last_obs, num_epochs=10, mini_batch_size=None):

        
        with torch.no_grad():
            last_obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0)
            last_value = self.critic(last_obs_tensor).squeeze()
        self.buffer.compute_returns_and_advantages(last_value)
        obs, actions, old_log_probs, advatages, returns = self.buffer.get()

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(num_epochs):
            new_log_probs, entropy, _ = self.actor.evaluate_actions(obs,actions)

            #compute policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advatages
            surr2 = torch.clamp(ratio, 1.0-self.clip_eps, 1+self.clip_eps)

            #take min
            policy_loss = -torch.min(surr1, surr2).mean()

            #value loss
            values = self.critic(obs)
            values_loss = F.mse_loss(values, returns)

            #entropy bonus 
            entropy_loss = -entropy.mean()

            #total loss 
            loss = -policy_loss + self.value_coef * values_loss - self.entropy_coef * entropy_loss

            #compute gradients
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            
            #clip gradients
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            #update step
            self.actor_optim.step()
            self.critic_optim.step()

            with torch.no_grad():
                values = self.critic(obs).squeeze()
                var_returns = returns.var()
                var_residual = (returns-values).var()
                explained_var = 1 - var_residual / (var_returns + 1e-8)

                rewards_tensor = torch.tensor([self.buffer.rewards[i] for i in range(len(self.buffer.rewards))])
                values_tensor = torch.tensor([self.buffer.values[i] for i in range(len(self.buffer.values))])
                dones_tensor = torch.tensor([self.buffer.dones[i] for i in range(len(self.buffer.dones))])


                next_values = torch.zeros_like(values_tensor)
                next_values[:-1] = values_tensor[1:]
                next_values[-1] = last_value

                td_errors = rewards_tensor + self.gamma * (1 - dones_tensor) * next_values - values_tensor
                mean_bellman_error = td_errors.abs().mean().item()


        self.buffer.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': values_loss.item(),
            'entropy': entropy.mean().item(),
            'explained_variance': explained_var.item(),
            'mean_bellman_error': mean_bellman_error
        }


