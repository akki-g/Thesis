"""
vmas_guarded_adapter.py — Adapter to connect the Guarded Territory VMAS
scenario to your existing GNN-MAPPO training loop.

This handles the interface differences between PettingZoo MPE (which your
current code targets) and VMAS, so you can reuse your CommPolicy,
CriticNetwork, RolloutBuffer, and PPOTrainer with minimal changes.

Key differences this adapter handles:
1. VMAS returns tuples of tensors (one per agent) vs PettingZoo's dict
2. VMAS steps all agents simultaneously vs PettingZoo's AEC iteration
3. VMAS batches across num_envs vs single-env in PettingZoo
4. Adjacency matrices must be built from defender positions only
   (intruders are scripted and NOT part of the GNN communication graph)

Usage:
    from vmas_guarded_adapter import GuardedTerritoryAdapter
    from guarded_territory import Scenario

    adapter = GuardedTerritoryAdapter(
        num_envs=64,
        device="cuda",
        n_scouts=3,
        n_interceptors=3,
        n_intruders=3,
        n_zones=2,
        max_steps=200,
    )

    obs, positions = adapter.reset()
    adj = adapter.build_adj(positions, r_comm=1.5)
    # obs shape: (num_envs, n_defenders, obs_dim)
    # positions shape: (num_envs, n_defenders, 2)
    # adj shape: (num_envs, n_defenders, n_defenders)
"""

import torch
import vmas
from guarded_territory import Scenario, get_obs_dim, SCOUT, INTERCEPTOR, INTRUDER


class GuardedTerritoryAdapter:
    """
    Wraps the VMAS Guarded Territory scenario to produce outputs
    compatible with your GNN-MAPPO training loop.

    The adapter:
    - Filters out intruder agents (scripted, not learned)
    - Stacks defender observations into (num_envs, n_defenders, obs_dim)
    - Provides positions for adjacency matrix construction
    - Returns a single team reward (shared across defenders for CTDE)
    - Handles the global done flag (broadcasts to all defenders)
    """

    def __init__(
        self,
        num_envs: int = 64,
        device: str = "cpu",
        n_scouts: int = 3,
        n_interceptors: int = 3,
        n_intruders: int = 3,
        n_zones: int = 2,
        max_steps: int = 200,
        **kwargs,
    ):
        self.num_envs = num_envs
        self.device = device
        self.n_scouts = n_scouts
        self.n_interceptors = n_interceptors
        self.n_intruders = n_intruders
        self.n_defenders = n_scouts + n_interceptors
        self.n_zones = n_zones

        # Create the VMAS environment
        self.env = vmas.make_env(
            scenario=Scenario(),
            num_envs=num_envs,
            device=device,
            continuous_actions=True,
            max_steps=max_steps,
            n_scouts=n_scouts,
            n_interceptors=n_interceptors,
            n_intruders=n_intruders,
            n_zones=n_zones,
            **kwargs,
        )

        # Compute obs dim for network construction
        self.obs_dim = get_obs_dim(n_scouts, n_interceptors, n_intruders, n_zones)

        # Identify which indices in env.agents are defenders vs intruders
        self.defender_indices = []
        self.intruder_indices = []
        for i, agent in enumerate(self.env.agents):
            if hasattr(agent, "agent_type"):
                if agent.agent_type in (SCOUT, INTERCEPTOR):
                    self.defender_indices.append(i)
                elif agent.agent_type == INTRUDER:
                    self.intruder_indices.append(i)

        assert len(self.defender_indices) == self.n_defenders, (
            f"Expected {self.n_defenders} defenders, found {len(self.defender_indices)}"
        )

        # Store agent type info for potential type-conditioned processing
        self.agent_types = []  # 0=scout, 1=interceptor
        for idx in self.defender_indices:
            agent = self.env.agents[idx]
            self.agent_types.append(agent.type_id)
        self.agent_types = torch.tensor(self.agent_types, device=device)

    def reset(self):
        """
        Reset the environment.

        Returns:
            obs: (num_envs, n_defenders, obs_dim) — stacked defender observations
            positions: (num_envs, n_defenders, 2) — defender positions for adj matrix
        """
        all_obs = self.env.reset()  # tuple of (num_envs, obs_dim_i) tensors

        # Stack defender observations
        defender_obs = torch.stack(
            [all_obs[i] for i in self.defender_indices], dim=1
        )  # (num_envs, n_defenders, obs_dim)

        # Extract positions from observations (pos is at indices 2:4 in obs)
        positions = defender_obs[:, :, 2:4].clone()  # (num_envs, n_defenders, 2)

        return defender_obs, positions

    def step(self, defender_actions: torch.Tensor):
        """
        Step the environment with learned defender actions.
        Intruder actions are handled internally by the scenario's process_action.

        Args:
            defender_actions: (num_envs, n_defenders, action_dim)
                             Continuous actions for each defender.

        Returns:
            obs: (num_envs, n_defenders, obs_dim)
            rewards: (num_envs, n_defenders) — per-defender rewards
            done: (num_envs,) — global done flag (bool)
            info: dict with extra logging info
            positions: (num_envs, n_defenders, 2) — for adj matrix construction
        """
        # Build full action list for all agents (defenders + intruders)
        # Intruders get dummy actions — process_action overrides them
        all_actions = []
        defender_action_idx = 0

        for i in range(len(self.env.agents)):
            if i in self.defender_indices:
                # Map from defender index to position in defender_actions
                local_idx = self.defender_indices.index(i)
                all_actions.append(defender_actions[:, local_idx])
            else:
                # Intruder: provide zero action (will be overwritten by script)
                all_actions.append(
                    torch.zeros(self.num_envs, 2, device=self.device)
                )

        # Step VMAS
        all_obs, all_rewards, dones, all_infos = self.env.step(all_actions)

        # Extract defender observations
        defender_obs = torch.stack(
            [all_obs[i] for i in self.defender_indices], dim=1
        )

        # Extract defender rewards
        defender_rewards = torch.stack(
            [all_rewards[i] for i in self.defender_indices], dim=1
        )  # (num_envs, n_defenders)

        # Positions for adjacency matrix
        positions = defender_obs[:, :, 2:4].clone()

        # Aggregate info
        info = {}
        if len(all_infos) > 0 and self.defender_indices:
            first_def_info = all_infos[self.defender_indices[0]]
            if isinstance(first_def_info, dict):
                info = first_def_info

        return defender_obs, defender_rewards, dones, info, positions

    def build_adj(self, positions: torch.Tensor, r_comm: float) -> torch.Tensor:
        """
        Build adjacency matrices from defender positions.

        This is the CRITICAL function for your GNN ablations.
        The adjacency matrix determines the communication graph —
        only defenders within r_comm of each other get edges.

        IMPORTANT: This builds a SEPARATE adjacency matrix for each
        environment in the batch, since agent positions differ.

        Args:
            positions: (num_envs, n_defenders, 2)
            r_comm: communication radius

        Returns:
            adj: (num_envs, n_defenders, n_defenders) — row-normalized
        """
        # Pairwise distances: (num_envs, n_defenders, n_defenders)
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff, dim=-1)

        # Adjacency: 1 if within r_comm (includes self-loops)
        adj = (dist <= r_comm).float()

        # Row-normalize (Kipf & Welling renormalization)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj = adj / deg

        return adj

    def get_team_reward(self, defender_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute a single team reward by averaging across defenders.
        Use this for the shared critic in CTDE.

        Args:
            defender_rewards: (num_envs, n_defenders)

        Returns:
            team_reward: (num_envs,)
        """
        return defender_rewards.mean(dim=1)

    @property
    def action_dim(self) -> int:
        """Action dimension for defenders (continuous 2D force)."""
        return 2

    @property
    def n_agents(self) -> int:
        """Number of learned agents (defenders only)."""
        return self.n_defenders


# ─────────────────────────────────────────────────────────────────
# Example: integration with your existing training loop
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Guarded Territory Adapter — Integration Test")
    print("=" * 60)

    adapter = GuardedTerritoryAdapter(
        num_envs=8,
        device="cpu",
        n_scouts=3,
        n_interceptors=3,
        n_intruders=3,
        n_zones=2,
        max_steps=200,
    )

    print(f"\nEnvironment configuration:")
    print(f"  Defenders (learned): {adapter.n_defenders}")
    print(f"    - Scouts: {adapter.n_scouts}")
    print(f"    - Interceptors: {adapter.n_interceptors}")
    print(f"  Intruders (scripted): {adapter.n_intruders}")
    print(f"  Observation dim: {adapter.obs_dim}")
    print(f"  Action dim: {adapter.action_dim}")
    print(f"  Agent types: {adapter.agent_types}")

    # ── Reset ──────────────────────────────────────────────────
    obs, positions = adapter.reset()
    print(f"\nAfter reset:")
    print(f"  obs shape: {obs.shape}")
    print(f"  positions shape: {positions.shape}")

    # ── Build adjacency matrix ─────────────────────────────────
    for r_comm in [0.5, 1.0, 1.5, 2.0]:
        adj = adapter.build_adj(positions, r_comm=r_comm)
        avg_degree = (adj > 0).float().sum(dim=-1).mean().item()
        print(f"  r_comm={r_comm}: adj shape={adj.shape}, avg_degree={avg_degree:.2f}")

    # ── Step with random actions ───────────────────────────────
    print(f"\nRunning 50-step rollout with random actions...")
    total_team_reward = torch.zeros(8)
    n_done = 0

    for step in range(50):
        # Random defender actions: (num_envs, n_defenders, 2)
        actions = torch.randn(8, adapter.n_defenders, 2) * 0.5

        obs, rewards, dones, info, positions = adapter.step(actions)
        team_reward = adapter.get_team_reward(rewards)
        total_team_reward += team_reward

        # Rebuild adj each step (dynamic graph!)
        adj = adapter.build_adj(positions, r_comm=1.5)

        if dones.any():
            n_done += dones.sum().item()

    print(f"  Total team reward: {total_team_reward.mean().item():.3f}")
    print(f"  Episodes completed: {n_done}")
    print(f"  Final obs shape: {obs.shape}")
    print(f"  Final adj shape: {adj.shape}")

    # ── Show how this maps to your existing code ───────────────
    print(f"\n{'=' * 60}")
    print("INTEGRATION WITH YOUR GNN-MAPPO:")
    print("=" * 60)
    print("""
    # In your training loop, replace the PettingZoo env with:

    adapter = GuardedTerritoryAdapter(num_envs=64, device="cuda", ...)

    # Your CommPolicy stays the same, just update dimensions:
    policy = CommPolicy(
        obs_dim=adapter.obs_dim,   # auto-computed
        hidden_dim=64,
        action_dim=adapter.action_dim,  # 2 (continuous)
        F_dim=64, G_dim=64, K_hops=K,
    )

    # Rollout collection:
    obs, positions = adapter.reset()
    adj = adapter.build_adj(positions, r_comm=R_COMM)

    for t in range(ROLLOUT_LENGTH):
        # obs[:, i, :] is agent i's observation
        # adj is (num_envs, n_defenders, n_defenders)
        actions, log_probs, entropy = policy.get_actions(obs, adj)
        obs, rewards, dones, info, positions = adapter.step(actions)
        adj = adapter.build_adj(positions, r_comm=R_COMM)
        # Store (obs, actions, rewards, adj, ...) in rollout buffer

    # PPO update uses stored adj matrices (not current ones!)
    # This is the same principle you already have.
    """)