import networkx as nx
import matplotlib.pyplot as plt
import random
import heapq
import time
from matplotlib.animation import FuncAnimation
import numpy as np

figsize = (6,6)

# Generate a random geometric graph
def generate_graph(num_nodes=10, connection_radius=0.3):
    G = nx.random_geometric_graph(num_nodes, connection_radius)
    pos = nx.get_node_attributes(G, 'pos')
    
    for u, v in G.edges():
        dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
        G[u][v]['weight'] = dist
    
    return G, pos

# Dijkstra's algorithm implementation with animation support
def dijkstra_animated(G, pos, source, target,animate=False):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Priority queue
    pq = [(0, source)]
    distances = {node: float('inf') for node in G.nodes}
    distances[source] = 0
    predecessors = {node: None for node in G.nodes}
    visited = set()
    steps = []  # Store steps for animation
    
    while pq:
        curr_dist, curr_node = heapq.heappop(pq)
        if curr_node in visited:
            continue
        visited.add(curr_node)
        
        # Store the step for visualization
        steps.append((curr_node, dict(distances)))
        
        if curr_node == target:
            break
        
        for neighbor in G.neighbors(curr_node):
            weight = G[curr_node][neighbor]['weight']
            new_dist = curr_dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = curr_node
                heapq.heappush(pq, (new_dist, neighbor))
    
    # Extract shortest path
    path = []
    step = target
    while step is not None:
        path.append(step)
        step = predecessors[step]
    path.reverse()
    
    # Animation function
    def update(num):
        ax.clear()
        nx.draw(G, pos, with_labels=False, node_color='lightgray', edge_color='gray', ax=ax)
        visited_nodes = [s[0] for s in steps[:num+1]]
        nx.draw_networkx_nodes(G, pos, nodelist=visited_nodes, node_color='blue', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color='green', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[target], node_color='red', ax=ax)
        if num < len(steps):
            ax.set_title(f"Visiting: {steps[num][0]}, Distances: {steps[num][1]}")
        if num == len(steps) - 1:
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red', ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=[(path[i], path[i+1]) for i in range(len(path)-1)], edge_color='red', width=2, ax=ax)
    
    ani=None
    if animate: 
        ani = FuncAnimation(fig, update, frames=len(steps), interval=10, repeat=False)
    else :
        ax.clear()
        nx.draw(G, pos, with_labels=False, node_color='lightgray', edge_color='gray', ax=ax)
        visited_nodes = [s[0] for s in steps]
        nx.draw_networkx_nodes(G, pos, nodelist=visited_nodes, node_color='blue', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color='green', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[target], node_color='red', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red', ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=[(path[i], path[i+1]) for i in range(len(path)-1)], edge_color='red', width=2, ax=ax)
    
    plt.show()
    return ani
  
# Generate and visualize the graph
if __name__ == "__main__":
    random.seed(42)
    G, pos = generate_graph(num_nodes=100, connection_radius=0.2)
    source, target = random.sample(list(G.nodes), 2)
    ani = dijkstra_animated(G, pos, source, target, animate = True)
