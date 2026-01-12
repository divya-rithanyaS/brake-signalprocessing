import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Load dataset
data = sio.loadmat("data/phdiff.mat")
A = data["A"]   # shape should be (156, 50001)

print("Dataset shape:", A.shape)

# Plot one sample from each synchronization state
plt.figure(figsize=(10, 7))

# State 1: first samplea
plt.subplot(3, 1, 1)
plt.plot(A[0])
plt.title("Sample from Synchronization State 1")
plt.ylabel("Phase Difference")

# State 2: first sample of second block
plt.subplot(3, 1, 2)
plt.plot(A[52])
plt.title("Sample from Synchronization State 2")
plt.ylabel("Phase Difference")

# State 3: first sample of third block
plt.subplot(3, 1, 3)
plt.plot(A[104])
plt.title("Sample from Synchronization State 3")
plt.xlabel("Time Index")
plt.ylabel("Phase Difference")

plt.tight_layout()
plt.show()
