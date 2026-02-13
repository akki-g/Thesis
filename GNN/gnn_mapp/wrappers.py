import torch


class GNNSpreadWrapper:
    def __init__(self, env, comm_radius=None, graph_type='distance_based'):
        self.env = env
        self.comm_radius = comm_radius
        self.graph_type = graph_type


    def build_adj_matrix(self, agent_pos):
        n_agents = len(agent_pos)

        if self.graph_type == 'distance_based':
            adj = torch.zeros((n_agents, n_agents))

            for i in range(n_agents):
                for j in range(i+1, n_agents):
                    dist = torch.linalg.norm(agent_pos[i] - agent_pos[j])
                    if dist <= self.comm_radius:
                        adj[i][j] = adj[j][i] = 1
            return adj
        else:
            return torch.ones((n_agents, n_agents)) - torch.eye(n_agents)
        

    def step(self, actions):
        obs, rewards, dones, truns, info = self.env.step(actions)

        agent_pos = self.get_agent_pos()
        adj = self.build_adj_matrix(agent_pos)

        for agent_id in info:
            info[agent_id]['adjacency'] = adj

        return obs, rewards, dones, truns, info
    def get_agent_pos(self):
        return torch.tensor([agent.state.p_pos for agent in self.env.world.agents])
    