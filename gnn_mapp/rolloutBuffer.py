import torch


class GNNRolloutBuffer:
    def __init__(self, gamma, gae_lambda):
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


    def add_timestep(self, obs, actions, rewards, dones, log_probs, values, A):

        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.adj.append(A)

    def compute_advantages(self, last_values):
        pass