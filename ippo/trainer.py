import torch
from ppoagent import PPOAgent

import matplotlib.pyplot as plt
import numpy as np

class IPPOTrainer:
    def __init__(self, env, num_agents, obs_dim, hidden_dim, action_dim):

        self.env = env
        self.agents = {}
        self.num_agents = num_agents
        for agent_id in env.possible_agents:
            self.agents[agent_id] = PPOAgent(obs_dim, hidden_dim, action_dim)
        self.metrics_history = {
            'timesteps': [],
            'mean_episode_return': [],
            'mean_episode_length': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'explained_variance': [],
            'mean_bellman_error': []
        }

    
    def collect_rollouts(self, num_steps, obs):
        if obs is None:
            obs, info = self.env.reset()
        episode_returns = {a_id: 0.0 for a_id in self.agents.keys()}
        episode_lengths = {a_id: 0 for a_id in self.agents.keys()}
        completed_episodes = []
        for step in range(num_steps):
            # collect obs from env
            actions = {}
            values = {}
            log_probs = {}
            for a_id, agent in self.agents.items():
                action, log_prob, value = agent.select_action(obs[a_id])
                actions[a_id] = action
                values[a_id] = value
                log_probs[a_id] = log_prob

            next_obs, rewards, dones, trunc, info = self.env.step(actions)
            
            for a_id, agent in self.agents.items():
                agent.buffer.add_rollout(
                    obs[a_id],
                    actions[a_id],
                    rewards[a_id],
                    dones[a_id],
                    log_probs[a_id],
                    values[a_id]
                )

                episode_returns[a_id] += rewards[a_id]
                episode_lengths[a_id] += 1
            

            if all(dones.values()) or all(trunc.values()):
                obs, info = self.env.reset()

                completed_episodes.append({
                    'returns': {a_id: episode_returns[a_id] for a_id in self.agents.keys()},
                    'lengths': {a_id: episode_lengths[a_id] for a_id in self.agents.keys()},
                    'mean_return': sum(episode_returns.values()) / len(episode_returns),
                    'mean_length': sum(episode_lengths.values()) / len(episode_lengths)
                })

                episode_returns = {a_id: 0.0 for a_id in self.agents.keys()}
                episode_lengths = {a_id: 0 for a_id in self.agents.keys()}

            else:
                obs = next_obs

        return obs, completed_episodes

    
    def train(self, total_timesteps, rollout_length):

        timesteps = 0 
        obs = None
        while timesteps < total_timesteps:
            last_obs, completed_episodes = self.collect_rollouts(rollout_length, obs)
            timesteps += rollout_length

            if completed_episodes:
                mean_return = sum(ep['mean_return'] for ep in completed_episodes)
                mean_length = sum(ep['mean_length'] for ep in completed_episodes)

                self.metrics_history['timesteps'].append(timesteps)
                self.metrics_history['mean_episode_return'].append(mean_return)
                self.metrics_history['mean_episode_length'].append(mean_length)

            all_agent_metrics = []
            for a_id, agent in self.agents.items():
                metrics = agent.update(last_obs[a_id])
                all_agent_metrics.append(metrics)

            avg_metrics = {
                key: sum(m[key] for m in all_agent_metrics) / len(all_agent_metrics)
                for key in all_agent_metrics[0].keys()
            }

            for k in ['policy_loss', 'value_loss', 'entropy', 'explained_variance', 'mean_bellman_error']:
                self.metrics_history[k].append(avg_metrics[k])

            if timesteps % (rollout_length * 10) == 0:
                print(f".      Policy Loss: {avg_metrics['policy_loss']:.4f}"
                      f".       Value Loss: {avg_metrics['value_loss']:.4f}"
                      f"           Entropy: {avg_metrics['entropy']:.4f}"
                      f".    Explained Var: {avg_metrics['explained_variance']:.4f}"
                      f"\nMean Bellman Error: {avg_metrics['mean_bellman_error']:.4f}")

            obs = last_obs

    def plot_metrics(self, save_path='training_metrics.png'):
        """
        Plot all tracked metrics.
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('IPPO Training Metrics', fontsize=16)
        
        # Episode metrics
        if self.metrics_history['mean_episode_return']:
            axes[0, 0].plot(self.metrics_history['timesteps'], 
                           self.metrics_history['mean_episode_return'])
            axes[0, 0].set_title('Mean Episode Return')
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Return')
            axes[0, 0].grid(True)
        
        if self.metrics_history['mean_episode_length']:
            axes[0, 1].plot(self.metrics_history['timesteps'], 
                           self.metrics_history['mean_episode_length'])
            axes[0, 1].set_title('Mean Episode Length')
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True)
        
        # Learning metrics (use update count for x-axis since they're logged per update)
        update_steps = list(range(len(self.metrics_history['policy_loss'])))
        
        axes[0, 2].plot(update_steps, self.metrics_history['policy_loss'])
        axes[0, 2].set_title('Policy Loss')
        axes[0, 2].set_xlabel('Updates')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True)
        
        axes[1, 0].plot(update_steps, self.metrics_history['value_loss'])
        axes[1, 0].set_title('Value Loss')
        axes[1, 0].set_xlabel('Updates')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(update_steps, self.metrics_history['entropy'])
        axes[1, 1].set_title('Entropy')
        axes[1, 1].set_xlabel('Updates')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].grid(True)
        
        axes[2, 1].plot(update_steps, self.metrics_history['explained_variance'])
        axes[2, 1].set_title('Explained Variance')
        axes[2, 1].set_xlabel('Updates')
        axes[2, 1].set_ylabel('Explained Var')
        axes[2, 1].grid(True)
        
        axes[2, 2].plot(update_steps, self.metrics_history['mean_bellman_error'])
        axes[2, 2].set_title('Mean Bellman Error')
        axes[2, 2].set_xlabel('Updates')
        axes[2, 2].set_ylabel('TD Error')
        axes[2, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"Metrics plot saved to {save_path}")



