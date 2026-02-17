import torch


class GNNRolloutBuffer:
    def __init__(self,gamma, gae_lambda, device):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.adj = []

        self.advantages = None
        self.returns = None

        self.device = device

    def add_timestep(self, obs, actions, rewards, dones, log_probs, values, A):

        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.adj.append(A)

    def compute_advantages(self, last_values):
        N = len(self.obs[0])
        buffer_size = len(self.obs)
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(self.values, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.dones, dtype=torch.float32, device=self.device)

        advantages = torch.zeros((buffer_size, N), dtype=torch.float32, device=self.device) 
        last_gae = torch.zeros(len(self.obs[0]), dtype=torch.float32, device=self.device)

        for t in reversed(range(buffer_size)):

            if t == self.buffer_size-1:
                if not torch.is_tensor(last_values):
                    next_value = torch.as_tensor(last_values, dtype=torch.float32, device=self.device)
                else:
                    next_value = last_values.to(device=self.device, dtype=torch.float32)

            else:
                next_value = values[t+1]

            deltas = rewards[t] + self.gamma * (1-dones[t]) * next_value - values[t]
            advantages[t] = deltas + self.gamma * self.gae_lambda * (1-dones[t]) * last_gae

            last_gae = advantages[t]


        returns = advantages + values
        self.advantages = advantages
        self.returns = returns


    def get_batches(self, B):
        perm = torch.randperm(len(self.obs))
        batches = perm.split(B)
        
        obs = torch.stack(self.obs)
        actions = torch.stack(self.actions)
        log_probs = torch.stack(self.log_probs)
        

        for idx in batches:
            m_obs = obs[idx]
            m_actions = actions[idx]
            m_log_probs = log_probs[idx]
            m_advantages = self.advantages[idx]
            m_returns = self.returns[idx]
            m_adj = self.adj[idx]

            yield m_obs, m_actions, m_log_probs, m_advantages, m_returns, m_adj
        

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.adj = []

        self.advantages = None
        self.returns = None



