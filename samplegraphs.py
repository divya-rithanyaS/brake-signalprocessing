import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Load discretized symbols
S = np.load("data/discrete_symbols.npy")   # shape (156, T)

# Pick one sample from each state
samples = {
    "State 1": S[29],     
    "State 2": S[52],    # first of state 2
    "State 3": S[104]    # first of state 3
}

plt.figure(figsize=(12,4))

for idx, (name, seq) in enumerate(samples.items(), 1):
    G = nx.DiGraph()

    # Build transition graph
    for i in range(len(seq)-1):
        u, v = int(seq[i]), int(seq[i+1])
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)

    # Draw
    plt.subplot(1,3,idx)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, arrows=True)
    plt.title(name + f"\nNodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")

plt.suptitle("Sample Transition Graphs from Each State")
plt.tight_layout()
plt.show()
