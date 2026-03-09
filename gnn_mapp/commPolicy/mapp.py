from pettingzoo.mpe import simple_spread_v3
from trainer import GNNTrainer

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
NUM_AGENTS = 3
MAX_CYCLES = 100

env = simple_spread_v3.parallel_env(N=3, max_cycles=MAX_CYCLES)
obs, info = env.reset()
obs_dim = obs["agent_0"].shape[0]
action_dim = 5


F = 64
G = 64
K = 2
hidden_dim = 64

lr = 3e-4
gamma = 0.99
gae_lambda = 0.85
clip_eps = 0.2
value_coef = 0.5
entropy_coef = 0.01

TOTAL_TIMESTEPS = 1_000_000
ROLLOUT_LENGTH  = 2048
BATCH_SIZE      = 64
NUM_EPOCHS      = 10
R_COMM          = 1.0

device = "cpu"

if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
print(f"Using: {device}")
trainer = GNNTrainer(
    num_agents=NUM_AGENTS, env=env,
    obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim,
    F=F, G=G, K=K, lr=lr, gamma=gamma, gae_lambda=gae_lambda,
    clip_eps=clip_eps, value_coef=value_coef, entropy_coef=entropy_coef,
    device=device
)

num_iterations = TOTAL_TIMESTEPS // ROLLOUT_LENGTH

for iteration in range(num_iterations):
    last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=ROLLOUT_LENGTH, r_comm=R_COMM)
    update_metrics = trainer.update(last_obs, num_epochs=NUM_EPOCHS, B=BATCH_SIZE)

    if iteration % 10 == 0:
        print(
            f"Iteration {iteration}/{num_iterations} | "
            f"policy_loss={update_metrics['policy_loss']:.4f} | "
            f"value_loss={update_metrics['value_loss']:.4f} | "
            f"entropy={update_metrics['entropy']:.4f} | "
            f"mean_bellman_error={update_metrics['mean_bellman_error']:.4f} | "
            f"mean_episode_return={rollout_metrics['mean_episode_return']:.4f} | "
            f"mean_episode_rewards={rollout_metrics['mean_episode_rewards']:.4f}"
        )

metrics_to_plot = [
    "policy_loss",
    "value_loss",
    "entropy",
    "mean_bellman_error",
    "mean_episode_return",
    "mean_episode_rewards",
]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for i, metric_name in enumerate(metrics_to_plot):
    values = trainer.metrics_history.get(metric_name, [])
    x = range(1, len(values) + 1)
    axes[i].plot(x, values, linewidth=1.8)
    axes[i].set_title(metric_name)
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel(metric_name)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "training_metrics.png")
plt.savefig(plot_path, dpi=180)
print(f"Saved metrics plot to {plot_path}")
