from trainer import IPPOTrainer
from pettingzoo.mpe import simple_spread_v3
from datetime import date

NUM_AGENTS = 5

env = simple_spread_v3.parallel_env(N=NUM_AGENTS, max_cycles=100, render_mode="human")

obs, info = env.reset()
print(obs['agent_0'].shape)

trainer = IPPOTrainer(env, NUM_AGENTS, obs['agent_0'].shape[0], 64, 5)
trainer.train(total_timesteps=10000000, rollout_length=2048)

trainer.plot_metrics(f'ippo_training_{date.today()}.png')
