from trainer import IPPOTrainer
from pettingzoo.mpe import simple_spread_v3

NUM_AGENTS = 3

env = simple_spread_v3.parallel_env(N=NUM_AGENTS, max_cycles=100)

obs, info = env.reset()
print(obs['agent_0'].shape)

trainer = IPPOTrainer(env, NUM_AGENTS, obs['agent_0'].shape[0], 64, 5)
trainer.train(total_timesteps=100000, rollout_length=2048)

trainer.plot_metrics('ippo_training.png')