import numpy as np
import scipy.io as sio

data = sio.loadmat("data/phdiff.mat")
A = data["A"]   # (156, 50001)

phi_min = A.min()
phi_max = A.max()

num_bins = 16
bins = np.linspace(phi_min, phi_max, num_bins+1)

S = np.digitize(A, bins) - 1
S[S == num_bins] = num_bins - 1

np.save("data/discrete_symbols.npy", S)

print("Saved discretized symbols.")
print("Shape:", S.shape)
print("Min bin:", S.min(), "Max bin:", S.max())
