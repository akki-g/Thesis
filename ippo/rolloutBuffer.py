import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

class RolloutBuffer:
    def __init__(self, buffer_size, obs_dim, action_dim, gamma, gae_lambda, device):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

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
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(self.values, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.dones, dtype=torch.float32, device=self.device)

        advantages = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        last_gae = 0

        for t in reversed(range(self.buffer_size)):

            if t == self.buffer_size-1:
                if not torch.is_tensor(last_value):
                    next_value = torch.tensor(last_value, dtype=torch.float32, device=self.device)
                else:
                    next_value = last_value.to(self.device, dtype=torch.float32)
            else:
                next_value = values[t+1]

            delta = rewards[t] + self.gamma * (1-dones[t]) * next_value - values[t]

            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

            last_gae = advantages[t]

        returns = advantages + values

        self.advantages = advantages
        self.returns = returns

    def get(self):

        obs = torch.as_tensor(np.asarray(self.obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(self.actions), dtype=torch.long, device=self.device)
        log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device)

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




        
