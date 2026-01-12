import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# Load discretized symbols
S = np.load("data/discrete_symbols.npy")   # shape (156, 50001)

# Output folders
base_folder = "graph"
state1 = os.path.join(base_folder, "state-1")
state2 = os.path.join(base_folder, "state-2")
state3 = os.path.join(base_folder, "state-3")

os.makedirs(state1, exist_ok=True)
os.makedirs(state2, exist_ok=True)
os.makedirs(state3, exist_ok=True)

num_signals = S.shape[0]

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

    # Decide state folder
    if i < 52:
        folder = state1
        state_name = "State 1"
    elif i < 104:
        folder = state2
        state_name = "State 2"
    else:
        folder = state3
        state_name = "State 3"

    # Draw graph
    plt.figure(figsize=(4,4))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=8)
    plt.title(f"{state_name} - Signal {i+1}")

    # Save
    filename = os.path.join(folder, f"signal_{i+1}.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")

print("All graphs saved by state.")
