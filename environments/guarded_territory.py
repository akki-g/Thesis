import torch
import typing
from vmas.simulator.core import Agent, World, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

# Agent Type definitions
SCOUT = "scout"
INTERCEPTOR = "interceptor"
INTRUDER = "intruder"


class Scenario(BaseScenario):
    """
    Guarded Territory: heterogenus cooperative-competitive MARL scenario

    Specifically designed for our GNN use case with
    - Heterogenus observation spaces (scouts see further, interceptors see closer)
    - Communication contraints (GNN is the only communication channel between agents)
    - Mixed cooperative-competitive dynamics (defenters cooperate, intruders compete)

    Key Principle: Interceptor CANNOT succeed without scout communication
    Ensures that r_comm and K ablations produce strong, interpretable signals
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Agent / World Config
        self.n_scouts = kwargs.get("n_scouts", 3)
        self.n_interceptors = kwargs.get("n_interceptors", 3)
        self.n_intruders = kwargs.get("n_intruders", 3)
        self.n_zones = kwargs.get("n_zones", 2)
        self.world_size = kwargs.get("world_size", 5.0)

        # Observation Radii - core asymmetry
        self.scout_fov = kwargs.get("scout_fov", 1.0)
        self.interceptor_fov = kwargs.get("interceptor_fov", 0.5)
        self.tag_radius = kwargs.get("tag_radius", 0.1)

        # Speeds
        self.intruder_speed = kwargs.get("intruder_speed", 0.5)
        self.defender_speed = kwargs.get("defender_speed", 0.8)

        self.n_defenders = self.n_scouts + self.n_interceptors

        # Use Scripted Intruder (default)
        self.intruder_script = kwargs.get("scripted_intruder", True)

        # Create World
        world = World(
            batch_dim=batch_dim,
            device=device,
            dt=0.1,
            drag=0.25,
            dim_c=0, # no build in comms
            x_semidim=self.world_size,
            y_semidim=self.world_size
        )

        # Create Scouts
        self.scouts = []
        for i in range(self.n_scouts):
            agent = Agent(
                name=f"scout_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.075),
                max_speed=self.defender_speed,
                color=Color.BLUE,
                u_range=1.0
            )
            agent.agent_type = SCOUT
            agent.type_id = 0
            world.add_agent(agent=agent)
            self.scouts.append(agent)


        # Create Interceptors
        self.intercpetors = []
        for i in range(self.n_interceptors):
            agent = Agent(
                name=f"interceptor_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.09),
                max_speed=self.defender_speed,
                color=Color.GREEN,
                u_range=1.0
            )
            agent.agent_type = INTERCEPTOR
            agent.type_id = 1
            world.add_agent(agent)
            self.intercpetors.append(agent)

        # Create Interuders (Scripted)
        self.intruders = []
        for i in range(self.n_intruders):
            intruder = Agent(
                name=f"intruder_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.075),
                max_speed=self.intruder_speed,
                color=Color.RED,
                u_range=1.0,
                action_script=self._intruder_action_script if self.intruder_script else None
            )
            intruder.agent_type = INTRUDER
            intruder.type_id = 2
            world.add_agent(intruder)
            self.intruders.append(intruder)


        self.defenders = self.scouts + self.intercpetors

        # Create Target Zones / Landmarks
        self.zones = []
        for i in range(self.n_zones):
            zone = Landmark(
                name=f"zone_{i}",
                collide=False,
                movable=False,
                shape=Sphere(radius=0.2),
                color=Color.LIGHT_GREEN
            )
            world.add_landmark(zone)
            self.zones.append(zone)


        # Tracking Tensors (allocated in reset)
        self._intruder_tagged = None
        self._zone_breached = None
        self._tag_count = None

        return world
    

    def reset_world_at(self, env_index: typing.Optional[int] = None):
        """
        Spawns agents and landmarks during env.reset()

        Layout:
        - Zones placed in the inner region
        - Defenders spawn near zones
        - Intruders spawn on the outer edges
        """
        batch = self.world.batch_dim
        device = self.world.device

        if env_index is None:
            self._intruder_tagged = torch.zeros(
                batch, self.n_intruders, dtype=torch.bool, device=device
            )
            self._zone_breached = torch.zeros(
                batch, self.n_zones, dtype=torch.bool, device=device
            )
            self._tag_count = torch.zeros(batch, dtype=torch.float32, device=device)
        else:
            self._intruder_tagged[env_index] = False
            self._zone_breached[env_index] = False
            self._tag_count[env_index] = 0.0
        
        # spawn zones in inner region
        for i, zone in enumerate(self.zones):
            pos = torch.zeros(
                (1,2) if env_index is not None else (batch, 2),
                dtype=torch.float32,
                device=device
            )

            # spread zones along x-axis near center
            angle = 2 * torch.pi * i / self.n_zones
            radius = 0.3
            pos[..., 0] = radius * torch.cos(torch.tensor(angle))
            pos[...,1] = radius * torch.sin(torch.tensor(angle))

            # add small noise
            pos += 0.1 * torch.randn_like(pos)
            zone.set_pos(pos, batch_index=env_index)

        for i, defender in enumerate(self.defenders):
            pos = torch.zeros(
                (1,2) if env_index is not None else (batch,2),
                dtype=torch.float32,
                device=device
            )
            angle = 2 * torch.pi * i / self.n_defenders
            radius = 0.5 + 0.2 * torch.rand(pos.shape[0], 1, device=device)
            pos[...,0:1] = radius * torch.cos(torch.tensor(angle))
            pos[...,1:2] = radius * torch.sin(torch.tensor(angle))
            pos += 0.05 * torch.randn_like(pos)
            defender.set_pos(pos, batch_index=env_index)

        # spawn intruders on outer edge
        for i, intruder in enumerate(self.intruders):
            pos = torch.zeros(
                (1, 2) if env_index is not None else (batch, 2),
                dtype=torch.float32,
                device=device,
            )
            angle = 2 * torch.pi * i / self.n_intruders + torch.pi  # opposite side
            radius = self.world_size * 0.85
            pos[..., 0] = radius * torch.cos(torch.tensor(angle))
            pos[..., 1] = radius * torch.sin(torch.tensor(angle))
            pos += 0.1 * torch.randn_like(pos)
            intruder.set_pos(pos, batch_index=env_index)

    # Scripted Intruder Behavior
    def _intruder_action_script(self, world: World):
        """
        Simple rule-based intruder policy: move toward nearest unbreached zone.
        Adds noise for unpredictability. Called automatically by VMAS for
        agents with action_script set.

        This is intentionally simple — start here, then optionally make
        intruders learned agents later as a thesis extension.
        """
        agent = world.agents[
            [a.name for a in world.agents].index(
                [a.name for a in self.intruders if a.action_callback == self._intruder_action_script][-1]
            )
        ]

    def _get_intruder_actions(self, intruder: Agent) -> torch.Tensor:
        """
        Compute scripted intruder action: navigate to nearest zone with noise.
        Returns action tensor of shape (batch, 2).
        """
        device = self.world.device
        batch = self.world.batch_dim

        # Find nearest zone
        min_dist = torch.full((batch,), float("inf"), device=device)
        target_pos = self.zones[0].state.pos.clone()

        for zone in self.zones:
            dist = torch.linalg.vector_norm(
                intruder.state.pos - zone.state.pos, dim=-1
            )
            closer = dist < min_dist
            min_dist = torch.where(closer, dist, min_dist)
            target_pos = torch.where(closer.unsqueeze(-1), zone.state.pos, target_pos)

        # Direction toward target + exploration noise
        direction = target_pos - intruder.state.pos
        direction = direction / (torch.linalg.vector_norm(direction, dim=-1, keepdim=True) + 1e-6)
        noise = 0.2 * torch.randn(batch, 2, device=device)
        action = self.intruder_speed * (direction + noise)

        return action
    
    def process_action(self, agent: Agent):
        """
        Override to inject scripted actions for intruders.
        For defenders, actions come from the learned policy (no-op here).
        """
        if hasattr(agent, "agent_type") and agent.agent_type == INTRUDER:
            action = self._get_intruder_actions(agent)
            agent.action.u = action


    def observation(self, agent: Agent) -> torch.Tensor:
        """
        Build obs vector for agent 
        Differs by agent type
        
        Scouts (large FOV):
            [own_vel(2), own_pos(2), type_one_hot(2),
             zone_rel_pos(n_zones*2),
             visible_intruders_rel_pos(n_intruders*2) <- sees more
             visible_intruders_vel(n_intruders*2)
             nearby_defenders_rel_pos(n_defenders-1)*2] 

        Interceptors (small FOV):
            [own_vel(2), own_pos(2), type_one_hot(2),
             zone_rel_pos(n_zones*2),
             visible_intruders_rel_pos(n_intruders*2) <- sees less (masked)
             visible_intruders_vel(n_intruders*2),
             nearby_defenders_rel_pos((n_defenders-1)*2)]

        CRITICAL: Intruder visibility is masked by FOV radius.
        Agents outside FOV get zero-filled observations.
        This is what makes GNN communication essential — scouts see
        intruders that interceptors cannot, and must relay this info.

        NOTE: We make obs dim identical across agent types by zero masking out of range entities
        """
        batch = self.world.batch_dim
        device = self.world.device

        if hasattr(agent, "agent_type") and agent.agent_type == SCOUT:
            fov = self.scout_fov
            type_oh = torch.tensor([1.0,0.0], device=device).expand(batch, 2)
        elif hasattr(agent, "agent_type") and agent.agent_type == INTERCEPTOR:
            fov = self.interceptor_fov
            type_oh = torch.tensor([0.0,1.0], device=device).expand(batch,2)
        else:
            # intruders get dubby obs since scripted
            return torch.zeros(batch,2,device=device) 
        
        obs_parts = []
        obs_parts.append(agent.state.vel) # (batch,2)
        obs_parts.append(agent.state.pos) # (batch,2)
        obs_parts.append(type_oh)         # (batch,2)

        # zone relative positions (always visible)
        for zone in self.zones:
            obs_parts.append(zone.state.pos - agent.state.pos)  # (batch,2)

        # intruder obs (fov masked)
        for intruder in self.intruders:
            rel_pos = intruder.state.pos - agent.state.pos
            dist = torch.linalg.vector_norm(rel_pos, dim=-1, keepdim=True) 
            visible = (dist <= fov).float()

            obs_parts.append(rel_pos * visible)
            obs_parts.append(intruder.state.vel * visible)
        
        # other defender rel pos (fov masked)
        for other in self.defenders:
            if other is agent:
                continue
            rel_pos = other.state.pos - agent.state.pos
            dist = torch.linalg.vector_norm(rel_pos, dim=-1, keepdim=True)
            visible = (dist <= fov).float()

            obs_parts.append(rel_pos*visible)

        return torch.cat(obs_parts, dim=-1)


    def reward(self, agent: Agent) -> torch.Tensor:
        """
        Compute reward for the given agent. Return shape (batch,)

        intruders are scripted so return 0s
        Defender reward structure (shared team reward + individual shaping):

        1. TEAM REWARD (same for all defenders — enables CTDE):
           - Large penalty when intruder reaches a zone (zone breached)
           - Large bonus when interceptor tags an intruder

        2. INDIVIDUAL SHAPING (differs by type):
           Scouts:
             - Small reward for being near intruders (encourages scouting)
             - Small reward for being within comm range of interceptors
               (encourages information relay positioning)
           Interceptors:
             - Distance-based shaping toward nearest untagged intruder
             - Bonus for successful tags

        This reward design means scouts are incentivized to position
        themselves as communication bridges, not just passive observers.
        """

        if hasattr(agent, "agent_type") and agent.agent_type == INTRUDER:
            return torch.zeros(self.world.batch_dim, device=self.world.device)
        
        batch = self.world.batch_dim
        device = self.world.device
        rew = torch.zeros(batch, device=device)

        # compute tagging events
        for j, intruder in enumerate(self.intruders):
            if self._intruder_tagged is None:
                break
            already_tagged = self._intruder_tagged[:, j]

            for interceptor in self.intercpetors:
                dist = torch.linalg.vector_norm(
                    interceptor.state.pos - intruder.state.pos, dim=-1
                )

                just_tagged = (~already_tagged) & (dist < self.tag_radius)
                self._intruder_tagged[:,j] = self._intruder_tagged[:, j] | just_tagged

                self._tag_count += just_tagged.float()


        # compute zone breach events
        for k, zone in enumerate(self.zones):
            for j, intruder in enumerate(self.intruders):
                if self._intruder_tagged in None:
                    break
                tagged = self._intruder_tagged[:, j]
                dist_to_zone = torch.linalg.vector_norm(
                    intruder.state.pos - zone.state.pos, dim=-1
                )
                breached = (~tagged) & (dist_to_zone < 0.15)
                # only count first breach per zone per ep
                new_breached = breached & (~self._zone_breached[:,k])
                self._zone_breached[:, k] = self._zone_breached[:,k] | breached

                rew -= 5.0 * new_breached.float()



        # team bonus: per-step differential, reward the tag in the step it happens
        for j, intruder in enumerate(self.intruders):
            dist = torch.linalg.vector_norm(
                interceptor.state.pos - intruder.state.pos, dim=-1
            )
            tagged = self._intruder_tagged[:, j]

            #give team bonus only on the step the tag occurs
            close_to_untagged = (~tagged) & (dist < self.tag_radius)
            rew += 3.0 * close_to_untagged.float()


        # individual shaping
        if hasattr(agent, "agent_type") and agent.agent_type == SCOUT:
            # Scouts: reward proximity to intruders (scouting)
            for intruder in self.intruders:
                dist = torch.linalg.vector_norm(
                    agent.state.pos - intruder.state.pos, dim=-1
                )
                # Reward for keeping intruders in FOV
                in_fov = (dist < self.scout_fov).float()
                rew += 0.1 * in_fov

            # Scouts: reward being within potential comm range of interceptors
            # This encourages relay positioning behavior
            for interceptor in self.interceptors:
                dist = torch.linalg.vector_norm(
                    agent.state.pos - interceptor.state.pos, dim=-1
                )
                # Reward for being at a "useful" relay distance
                # Not too close (wasting the FOV advantage), not too far
                good_relay_dist = (dist > 0.2) & (dist < 1.0)
                rew += 0.05 * good_relay_dist.float()

        elif hasattr(agent, "agent_type") and agent.agent_type == INTERCEPTOR:
            # Interceptors: shaped reward toward nearest untagged intruder
            min_dist = torch.full((batch,), float("inf"), device=device)
            for j, intruder in enumerate(self.intruders):
                if self._intruder_tagged is not None:
                    tagged = self._intruder_tagged[:, j]
                else:
                    tagged = torch.zeros(batch, dtype=torch.bool, device=device)

                dist = torch.linalg.vector_norm(
                    agent.state.pos - intruder.state.pos, dim=-1
                )
                # Only consider untagged intruders
                effective_dist = torch.where(tagged, torch.tensor(float("inf"), device=device), dist)
                min_dist = torch.minimum(min_dist, effective_dist)

            # Negative distance shaping (closer = better)
            # Clamp to avoid inf when all intruders tagged
            min_dist = torch.clamp(min_dist, max=5.0)
            rew -= 0.1 * min_dist

        return rew
    

    # done condition
    def done(self) -> torch.Tensor:
        """
        Episode ends when:
        - All intruders are tagged (defenders win), OR
        - All zones are breached (intruders win)

        Returns (batch,) bool tensor.
        """
        if self._intruder_tagged is None or self._zone_breached is None:
            return torch.zeros(
                self.world.batch_dim, dtype=torch.bool, device=self.world.device
            )

        all_tagged = self._intruder_tagged.all(dim=-1)   # (batch,)
        all_breached = self._zone_breached.all(dim=-1)   # (batch,)
        return all_tagged | all_breached
    
    def info(self, agent: Agent) -> dict:
        """Return extra info for logging during training."""
        info = {}
        if self._intruder_tagged is not None:
            info["n_tagged"] = self._intruder_tagged.sum(dim=-1).float()
        if self._zone_breached is not None:
            info["n_breached"] = self._zone_breached.sum(dim=-1).float()
        return info




def get_obs_dim(n_scouts=3, n_interceptors=3, n_intruders=3, n_zones=2):
    """
    Compute the observation dimension for defenders.
    Useful for constructing your GNN-MAPPO policy networks.

    obs = [vel(2) + pos(2) + type_oh(2)
        + zones(n_zones*2)
        + intruders_rel_pos(n_intruders*2) + intruders_vel(n_intruders*2)
        + other_defenders_rel_pos((n_scouts+n_interceptors-1)*2)]
    """
    n_defenders = n_scouts + n_interceptors
    obs_dim = (
        2       # vel
        + 2     # pos
        + 2     # type one-hot
        + n_zones * 2              # zone relative positions
        + n_intruders * 2          # intruder relative positions (masked)
        + n_intruders * 2          # intruder velocities (masked)
        + (n_defenders - 1) * 2    # other defenders relative positions
    )
    return obs_dim

