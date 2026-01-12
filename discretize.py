import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# -----------------------------
# Load dataset
# -----------------------------
data = sio.loadmat("data/phdiff.mat")
A = data["A"]   # shape (156, 50001)

# Pick one sample
idx = 0   # 0 = State 1, 52 = State 2, 104 = State 3
phi = A[idx]

# -----------------------------
# Discretize this sample only (for visualization)
# -----------------------------
num_bins = 16
phi_min = phi.min()
phi_max = phi.max()
bins = np.linspace(phi_min, phi_max, num_bins + 1)

sym = np.digitize(phi, bins) - 1
sym[sym == num_bins] = num_bins - 1  # safety fix

print("Unique bin values used:", np.unique(sym))

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,6))

# Original signal
plt.subplot(2,1,1)
plt.plot(phi)
plt.title(f"Original Phase Difference (Sample {idx})")
plt.ylabel("Phase Difference")

# Discretized signal (step plot = honest)
plt.subplot(2,1,2)
plt.step(range(len(sym)), sym, where="post")
plt.yticks(range(0,16))
plt.title("Discretized Signal (Bin Index 0 to 15)")
plt.ylabel("Bin Index")
plt.xlabel("Time Index")

plt.tight_layout()
plt.show()
