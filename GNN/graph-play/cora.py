import numpy as np 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import deque, defaultdict

from torch_geometric.datasets import Planetoid


dataset = Planetoid(root='/tmp/Cora', name='Cora')

data= dataset[0]


edges = data.edge_index.numpy().T
node_features = data.x.numpy()
labels = data.y.numpy()

print(f"num nodes: {node_features.shape[0]}")
print(f"num edges: {edges.shape[0]}")
print(f"feat dim: {node_features.shape[1]}")


class Graph:
    """Graph class w multi representation"""

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

        self.adj_list = defaultdict(list)
        self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        self.edges = []

        self.node_attributes = {}

    def add_edge(self, u, v, directed=True):
        
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)

        self.adj_matrix[u][v] = 1

        self.edges.append((u,v))

        if not directed:
            if u not in self.adj_list[v]:
                self.adj_list[v].append(u)

            self.adj_matrix[v][u] = 1

    def get_neighbors(self, node):
        return self.adj_list[node]
    
    def get_degree(self, node):
        return len(self.adj_list[node])
    
    def set_node_attribute(self, node, attr_name, value):
        if node not in self.node_attributes:
            self.node_attributes[node] = {}

        self.node_attributes[node][attr_name] = value
    
    def get_node_attributes(self, node, attr_name, default=None):
        return self.node_attributes.get(node, {}).get(attr_name, default)
    
    # viz methods

    def show_ascii(self, max_nodes = 20):
        """simple ASCII representation of the graph"""

        print(f"\n{'='*50}")
        print(f"Graph: {self.num_nodes} nodes, {len(self.edges)} edges")
        print(f"\n{'='*50}")

        nodes_to_show = min(max_nodes, self.num_nodes)

        for node in range(nodes_to_show):
            neighbors = self.get_neighbors(node)
            degree = len(neighbors)

            attrs = []
            if node in self.node_attributes:
                for k,v in self.node_attributes[node].items():
                    attrs.append(f"{k}={v}")
            attr_str = f"[{', '.join(attrs)}]" if attrs else ""
            
            print(f"Node {node:3d}{attr_str}")
            print(f" |__Degree: {degree}")

            if neighbors:
                neighbor_str = ', '.join(map(str, neighbors[:10]))
                suffix = "..." if len(neighbors) > 0 else ''
                print(f"|__ Neighbors:{neighbor_str}{suffix}")
            else:
                print(f"|__ Neighbors: none (isolated node)")
            print()

        if nodes_to_show < self.num_nodes:
            print(f"... and {self.num_nodes - nodes_to_show} more nodes")
        print(f"{'='*50}")

    def show_matrix(self, max_size=20):
        """viz to show adj matrix as heatmap
        good for seeing connection patterns"""

        size = min(max_size, self.num_nodes)

        plt.figure(figsize=(10,8))
        plt.imshow(self.adj_matrix[:size, :size], cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Edge Exists')
        plt.xlabel('Node ID')
        plt.ylabel('NodeID')
        plt.title(f'Adjacency Matrix (first {size}x{size} nodes)')

        plt.grid(False)

        step = max(1, size//10)
        plt.xticks(range(0, size, step))
        plt.yticks(range(0, size, step))

        plt.tight_layout()
        plt.savefig('graph_adj_matrix.png', dpi=150)
        plt.show()

        print(f"Sparcity: {100 * (1 - np.count_nonzero(self.adj_matrix) / self.adj_matrix.size):.2f}")

    
    def show_subgraph(self, center_node, radius=2, layout='spring', show_labels=True, save_path=None):
        """
        Viz local neighborhood of a node to understand local graph structures
        center_node: node to center viz around
        radius: how many hops to include
        """

        visited = {center_node}
        layers = [[center_node]]
        curr_layer = [center_node]

        for _ in range(radius):
            next_layer = []
            for node in curr_layer:
                for neighbor in self.get_neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.append(neighbor)

            if next_layer:
                layers.append(next_layer)
                curr_layer = next_layer
            else:
                break
        
        subgraph_nodes = visited
        G = nx.DiGraph()
        G.add_nodes_from(subgraph_nodes)

        for u, v in self.edges:
            if u in subgraph_nodes and v in subgraph_nodes:
                G.add_edge(u, v)

        if layout == 'spring':
            pos = nx.spring_layout(G, k=1.5, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'shell':
            pos = nx.shell_layout(G, nlist=layers)
        else:
            pos = nx.spring_layout(G)

        node_colors = []

        for node in G.nodes():
            if node == center_node:
                node_colors.append('red')
            else:
                for i, layer in enumerate(layers):
                    if node in layer:
                        intensity = i / len(layers)
                        node_colors.append(plt.cm.RdYlBu(intensity))
                        break

        plt.figure(figsize=(14,10))

        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, alpha=0.5, width=1.5, connectionstyle='arc3,rad=0.1')

        nx.draw_networkx_nodes(G, pos, nodelist=[center_node], node_color='darkred', node_size=800, node_shape='*')

        if show_labels:
            labels = {node: str(node) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        plt.title(f"Subgraph around Node {center_node} (radius={radius})", fontsize=14, fontweight='bold')

        legend_elements = [Patch(facecolor='darkred', label=f'Center (node {center_node})')]
        for i in range(1, len(layers)):
            color = plt.cm.RdYlBu(i/len(layers))
            legend_elements.append(Patch(facecolor=color, label=f'{i}-hop neighbor'))
        
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(f'graph_subgraph_node_{center_node}.png', dpi=150, bbox_inches='tight')
        
        plt.show()

        print(f"\nSubgraph Statistics:")
        print(f"  Center node: {center_node}")
        for i, layer in enumerate(layers):
            print(f"  {i}-hop: {len(layer)} nodes")

    def show_full_graph(self, max_nodes=100, layout='spring', 
                       node_colors=None, node_sizes=None,
                       save_path=None):
        """
        Visualize entire graph (or subset if too large)
        
        Args:
            max_nodes: Maximum nodes to visualize
            layout: Layout algorithm
            node_colors: List/array of colors for each node
            node_sizes: List/array of sizes for each node
            save_path: Path to save figure
        """
        if self.num_nodes > max_nodes:
            print(f"⚠️  Graph has {self.num_nodes} nodes. Showing first {max_nodes}...")
            nodes_to_show = set(range(max_nodes))
        else:
            nodes_to_show = set(range(self.num_nodes))
        
        # Create NetworkX graph
        G = nx.DiGraph()
        G.add_nodes_from(nodes_to_show)
        
        for u, v in self.edges:
            if u in nodes_to_show and v in nodes_to_show:
                G.add_edge(u, v)
        
        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Node colors
        if node_colors is None:
            # Color by degree
            degrees = [self.get_degree(node) for node in G.nodes()]
            node_colors = degrees
            cmap = plt.cm.viridis
        else:
            # Use provided colors
            node_colors = [node_colors[node] if node < len(node_colors) else 0 
                          for node in G.nodes()]
            cmap = plt.cm.tab10
        
        # Node sizes
        if node_sizes is None:
            # Size by degree
            degrees = [self.get_degree(node) for node in G.nodes()]
            node_sizes = [50 + d * 20 for d in degrees]
        
        # Draw
        plt.figure(figsize=(16, 12))
        
        nx.draw_networkx_edges(G, pos, 
                              edge_color='lightgray',
                              arrows=True,
                              arrowsize=10,
                              alpha=0.3,
                              width=0.5)
        
        nodes = nx.draw_networkx_nodes(G, pos,
                                       node_color=node_colors,
                                       node_size=node_sizes,
                                       cmap=cmap,
                                       alpha=0.8)
        
        plt.colorbar(nodes, label='Node Degree')
        
        plt.title(f'Graph Visualization\n{len(G.nodes())} nodes, {len(G.edges())} edges',
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig('graph_full_visualization.png', dpi=150, bbox_inches='tight')
        
        plt.show()

    def show_degree_dist(self, save_path=None):
        """Viz degree dist"""

        degrees = [self.get_degree(i) for i in range(self.num_nodes)]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].hist(degrees, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Degree', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Degree Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Log-log plot (checks for power law)
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        axes[1].scatter(unique_degrees, counts, alpha=0.6, s=50)
        axes[1].set_xlabel('Degree (log scale)', fontsize=12)
        axes[1].set_ylabel('Count (log scale)', fontsize=12)
        axes[1].set_title('Log-Log Degree Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].grid(alpha=0.3)

        # Cumulative distribution
        sorted_degrees = np.sort(degrees)
        cumulative = np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
        axes[2].plot(sorted_degrees, cumulative, linewidth=2, color='darkgreen')
        axes[2].set_xlabel('Degree', fontsize=12)
        axes[2].set_ylabel('Cumulative Probability', fontsize=12)
        axes[2].set_title('Cumulative Degree Distribution', fontsize=14, fontweight='bold')
        axes[2].grid(alpha=0.3)
        
        # Add statistics
        stats_text = f"Mean: {np.mean(degrees):.2f}\n"
        stats_text += f"Median: {np.median(degrees):.2f}\n"
        stats_text += f"Max: {np.max(degrees)}\n"
        stats_text += f"Min: {np.min(degrees)}"
        
        axes[0].text(0.95, 0.95, stats_text,
                    transform=axes[0].transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig('graph_degree_distribution.png', dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return degrees
    
    def show_interactive(self, max_nodes=100):
        """
        Create an interactive HTML visualization using pyvis
        Requires: pip install pyvis
        """
        try:
            from pyvis.network import Network
        except ImportError:
            print("⚠️  pyvis not installed. Install with: pip install pyvis")
            return
        
        nodes_to_show = min(max_nodes, self.num_nodes)
        
        # Create network
        net = Network(height='750px', width='100%', 
                     bgcolor='#222222', font_color='white',
                     directed=True)
        
        # Set physics options for better layout
        net.set_options('''
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
          }
        }
        ''')
        
        # Add nodes
        for node in range(nodes_to_show):
            degree = self.get_degree(node)
            
            # Size and color by degree
            size = 10 + degree * 2
            color = f"#{int(255 * min(degree/10, 1)):02x}4444"
            
            # Title shows up on hover
            title = f"Node {node}<br>Degree: {degree}"
            
            # Add node attributes if they exist
            if node in self.node_attributes:
                for key, val in self.node_attributes[node].items():
                    title += f"<br>{key}: {val}"
            
            net.add_node(node, label=str(node), title=title, 
                        size=size, color=color)
        
        # Add edges
        for u, v in self.edges:
            if u < nodes_to_show and v < nodes_to_show:
                net.add_edge(u, v)
        
        # Save and show
        output_file = 'graph_interactive.html'
        net.show(output_file)
        print(f"✅ Interactive visualization saved to: {output_file}")
        print(f"   Open this file in your web browser!")

    def __repr__(self):
        return f"Graph(nodes={self.num_nodes}, egdes={len(self.edges)})"
    

cora_graph = Graph(num_nodes=node_features.shape[0])

for u,v in edges:
    cora_graph.add_edge(u, v, directed=True)

category_names = ["Neural_Networks", "Rule_Learning", "Reinforcement_Learning",
                      "Probabilistic_Methods", "Theory", "Genetic_Algorithms", "Case_Based"]
    
for node in range(cora_graph.num_nodes):
    cora_graph.set_node_attribute(node, 'category', category_names[labels[node]])
    cora_graph.set_node_attribute(node, 'category_id', int(labels[node]))

print(cora_graph)

def visulaize(cora_graph):
     # 1. Simple ASCII representation
    print("\n1. ASCII Representation:")
    cora_graph.show_ascii(max_nodes=10)
    
    # 2. Adjacency matrix heatmap
    print("\n2. Adjacency Matrix:")
    cora_graph.show_matrix(max_size=50)
    
    # 3. Subgraph around a specific node
    print("\n3. Subgraph Visualization:")
    cora_graph.show_subgraph(center_node=0, radius=2, layout='spring')
    
    # 4. Degree distribution
    print("\n4. Degree Distribution:")
    cora_graph.show_degree_dist()
    
    # 5. Full graph (colored by category)
    print("\n5. Full Graph Visualization:")
    cora_graph.show_full_graph(max_nodes=200, node_colors=labels)
    
    # 6. Interactive visualization
    print("\n6. Interactive Visualization:")
    cora_graph.show_interactive(max_nodes=100)


# visulaize(cora_graph=cora_graph)



def dijakstra(graph, start_node, end_node=None):

    distances = {i: float('inf') for i in range(graph.num_nodes)}
    distances[start_node] = 0

    predecessors = {i: None for i in range(graph.num_nodes)}

    visited = set()

    import heapq
    pq = [(0, start_node)]

    while pq:
        curr_dist, curr_node = heapq.heappop(pq)

        if curr_node in visited:
            continue

        if end_node is not None and curr_node == end_node:
            break

        for neighbor in graph.get_neighbors(curr_node):
            edge_weight = 1

            distance = curr_dist + edge_weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = curr_node
                heapq.heappush(pq, (distance, neighbor))

    return distances, predecessors

def reconstruct_path(predecessors, start_node, end_node):
    """ reconstructs a path fron start to end using predecessors"""

    path = []   
    current = end_node

    while current is not None:
        path.append(current)
        current = predecessors[current]

    path.reverse()

    if path[0] != start_node:
        return None
    
    return path


start = 0
end = 100

distances, predecessors = dijakstra(cora_graph, start, end)

if distances[end] != float('inf'):
    path = reconstruct_path(predecessors, start, end)

    print(f"Shortest Path from paper {start} to paper {end}: ")
    print(f"Path: {path}")
    print(f"Lenght: {len(path)}")
    print(f"\nInterpretation: Paper {end} cites a paper which cites another, which cites another, eventually leading to paper {start}")

else:
    print(f"No citation path exists from {start} for {end}")


# good questions to explore about a graph

def graph_diameter(graph, sample_size=100):

    import random

    max_dist = 0
    nodes = random.sample(range(graph.num_nodes), min(sample_size, graph.num_nodes))

    for n in nodes:
        distances, _ = dijakstra(graph, n)

        reachable_dists = [d for d in distances if d != float('inf')]

        if reachable_dists:
            max_dist = max(max_dist, max(reachable_dists))


    return max_dist


print(f"Approx Diameter: {graph_diameter(cora_graph)}")

def count_connected_components(graph):
    visited = set()

    num_components = 0

    for start in range(graph.num_nodes):
        if start in visited:
            continue

        num_components += 1
        q = deque([start])
        visited.add(start)

        while q:
            node = q.popleft()

            for neighbor in graph.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)

    return num_components

print(f"Connected Components: {count_connected_components(cora_graph)}")

# these can all be connected to MARL as:
# the shortest path between nodes can be interpreted as the shorted number of communication hops between agents
# diameter: can be interpretes as the worst case communication delay
# connected components: can be interpreted as seperate agents teams/squads

# more standard graph practice

def bfs_layers(graph, start_node, max_depth=None):
    """BFA that returns nodes at each depth level
    this mirrors how GNNs aggregate information layer-by-layer"""


    visited = {start_node}
    layers = [[start_node]]
    curr_layer = [start_node]

    depth = 0
    while curr_layer and (max_depth is None or max_depth > depth):
        next_layer = []

        for node in curr_layer:
            for neighbor in graph.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_layer.append(neighbor)
        if next_layer:
            depth += 1
            layers.append(next_layer)
            curr_layer = next_layer
        else:
            break

    return layers

layers = bfs_layers(cora_graph, start, max_depth=3)

print(f"receptive field from node {start}:")
for i, layer in enumerate(layers):
    print(f"    Depth {i}: {len(layer)} nodes")
    if i <= 2:
        print(f"    Nodes: {layer[:10]}...")

print(f"interpretation: a 3-layer GNN can see at node {start} information from:")
print(f"{sum(len(l) for l in layers)} nodes total out of {cora_graph.num_nodes} nodes")


# who are the most important papers

def degree_centrality(graph):
    degrees = [graph.get_degree(i) for i in range(graph.num_nodes)]

    max_degree = max(degrees)

    return [d/ max_degree for d in degrees]


def between_centrality_approx(graph, sample_size=100):
    """papers that are on many shortes paths 
    these are bridge papers connecting different research areas"""

    import random

    betweenness = defaultdict(int)
    nodes = random.sample(range(graph.num_nodes), min(sample_size, graph.num_nodes))

    for start in nodes:
        distances, predecessors = dijakstra(graph, node)

        for end in nodes:
            if end == start or distances[end] == float('inf'):
                continue

            path = reconstruct_path(graph, start, end)

            if path:
                for node in path[1:-1]:
                    betweenness[node] += 1

    values = list(betweenness.values())

    if values: 
        max_val = max(values)
        return {k: v/max_val for k,v in betweenness.items()}
    
    else:
        return {}
    

deg_cent = degree_centrality(cora_graph)
between_cent = between_centrality_approx(cora_graph)

most_cited = np.argsort(deg_cent)[-10:][::-1]
most_brigde = sorted(between_cent.items(), key= lambda x: x[1], reverse=True)[:10]

print("Most cited papers (degree centralilty)")
for node in most_cited:
    print(f"Node {node}: degree = {cora_graph.get_degree(node)}")

print("\n Most bridged papers:")
for node, cent in most_brigde:
    print(f"Node {node}: betweenness {cent:.3f}")

# connections to MARL Space:
# degree centrality: which agents communicate w most others
# betweenness: which agents are critical for team coordination
# w formation control, high-betweenness agents are single points of failure

