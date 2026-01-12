import numpy as np
import networkx as nx
from math import log
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Load discretized symbols
# ===============================
S = np.load("data/discrete_symbols.npy")   # (156, T)

# ===============================
# Feature Extraction
# ===============================
features = []

for seq in S:
    G = nx.DiGraph()

    for i in range(len(seq)-1):
        u, v = int(seq[i]), int(seq[i+1])
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)

    N = G.number_of_nodes()
    M = G.number_of_edges()

    density = nx.density(G)

    degrees = [d for _, d in G.degree()]
    mean_deg = np.mean(degrees)
    std_deg = np.std(degrees)

    self_loops = sum(1 for u,v in G.edges() if u==v)
    self_loop_ratio = self_loops / M if M>0 else 0

    num_scc = nx.number_strongly_connected_components(G)

    total_w = sum(G[u][v]["weight"] for u,v in G.edges())
    entropy = 0
    for u,v in G.edges():
        p = G[u][v]["weight"] / total_w
        entropy -= p * log(p)

    features.append([
        N, M, density, mean_deg, std_deg,
        self_loop_ratio, num_scc, entropy
    ])

features = np.array(features)
np.save("data/graph_features.npy", features)

print("Feature matrix shape:", features.shape)
print("First 10 rows:\n", features[:10])

# ===============================
# Convert to Table
# ===============================
columns = [
    "Num_Nodes","Num_Edges","Density",
    "Mean_Degree","Std_Degree",
    "Self_Loop_Ratio","Num_SCC","Transition_Entropy"
]

df = pd.DataFrame(features, columns=columns)
states = ["State 1"]*52 + ["State 2"]*52 + ["State 3"]*52
df["State"] = states

df.to_excel("data/graph_features_table.xlsx", index=False)
print("Saved: data/graph_features_table.xlsx")
print(df.head(10))

# ===============================
# Summary Table (Mean ± Std)
# ===============================
summary = []
for f in columns:
    row = [f]
    for s in ["State 1","State 2","State 3"]:
        vals = df[df["State"]==s][f]
        row.append(f"{vals.mean():.3f} ± {vals.std():.3f}")
    summary.append(row)

summary_df = pd.DataFrame(summary, columns=["Feature","State 1","State 2","State 3"])
print("\nSummary Table (Mean ± Std):\n")
print(summary_df)

summary_df.to_excel("data/feature_summary_mean_std.xlsx", index=False)
print("Saved: data/feature_summary_mean_std.xlsx")

# ===============================
# Visualization: Boxplots
# ===============================
for f in columns:
    plt.figure()
    df.boxplot(column=f, by="State")
    plt.title(f"{f} by State")
    plt.suptitle("")
    plt.ylabel(f)
    plt.show()

# ===============================
# Visualization: Scatter Example
# ===============================
plt.figure()
for state in ["State 1","State 2","State 3"]:
    sub = df[df["State"]==state]
    plt.scatter(sub["Num_Nodes"], sub["Transition_Entropy"], label=state)

plt.xlabel("Num_Nodes")
plt.ylabel("Transition_Entropy")
plt.legend()
plt.title("Nodes vs Entropy")
plt.show()
