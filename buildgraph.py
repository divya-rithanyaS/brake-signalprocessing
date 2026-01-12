import numpy as np
import networkx as nx

# Load discretized symbols
S = np.load("data/discrete_symbols.npy")   # shape (156, 50001)

num_signals = S.shape[0]

node_counts = []
edge_counts = []

for i in range(num_signals):
    sym = S[i]
    G = nx.DiGraph()

    # Build transitions
    for t in range(len(sym) - 1):
        u = int(sym[t])
        v = int(sym[t+1])
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    node_counts.append(n_nodes)
    edge_counts.append(n_edges)

    print(f"Signal {i+1}: Nodes = {n_nodes}, Edges = {n_edges}")

print("\n--- Summary over all 156 signals ---")
print("Nodes: min =", min(node_counts), 
      ", max =", max(node_counts), 
      ", mean =", np.mean(node_counts))

print("Edges: min =", min(edge_counts), 
      ", max =", max(edge_counts), 
      ", mean =", np.mean(edge_counts))
