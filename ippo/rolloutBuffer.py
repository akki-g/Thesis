import torch
import torch.nn.functional as F
import torch.nn as nn


class RolloutBuffer:
    def __init__(self, buffer_size, obs_dim, action_dim, gamma, gae_lambda):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        self.idx = 0

    def add_rollout(self, obs, action, reward, done, log_prob, value):

        self.obs.append(obs)
        self.actions.append(action)

        self.rewards.append(reward.item() if torch.is_tensor(reward) else reward)
        self.dones.append(done.item() if torch.is_tensor(done) else done)
        self.log_probs.append(log_prob.item() if torch.is_tensor(log_prob) else log_prob)
        self.values.append(value.item() if torch.is_tensor(value) else value)

        self.idx += 1


    def compute_returns_and_advantages(self, last_value):
        rewards = torch.as_tensor(self.rewards).float()
        values = torch.as_tensor(self.values).float()
        dones = torch.as_tensor(self.dones).int()
        """rewards = torch.stack(self.rewards).squeeze()
        values = torch.stack(self.values).squeeze()
        dones = torch.stack(self.dones).squeeze()"""

        advantages = torch.zeros(self.buffer_size)
        last_gae = 0

        for t in reversed(range(self.buffer_size)):

            if t == self.buffer_size-1:
                if torch.is_tensor(last_value):
                    last_value = last_value.item()
                next_value = last_value
            else:
                next_value = values[t+1]

            delta = rewards[t] + self.gamma * (1-dones[t]) * next_value - values[t]

            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

            last_gae = advantages[t]

        returns = advantages + values

        self.advantages = advantages
        self.returns = returns

    def get(self):

        obs = torch.as_tensor(self.obs).float()
        actions = torch.as_tensor(self.actions)
        log_probs = torch.as_tensor(self.log_probs).float()

        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)


        return obs, actions, log_probs, adv, self.returns
    
    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        self.idx = 0




        